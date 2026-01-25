"""Particle tracking using neural flow field integration."""

import logging
from typing import Tuple, List, Optional, Callable
import numpy as np
import torch
from tqdm import tqdm

from .model import NeuralFlowField


logger = logging.getLogger(__name__)


def euler_step(model: NeuralFlowField, pos: torch.Tensor, t: torch.Tensor,
              dt: float, device: torch.device) -> torch.Tensor:
    """
    Single Euler integration step.
    
    Args:
        model: Neural flow field model
        pos: Current positions, shape (N, 2) where pos[:, 0] is x and pos[:, 1] is y
        t: Current time, scalar or shape (N,)
        dt: Time step size
        device: Device for computation
        
    Returns:
        New positions after step, shape (N, 2)
    """
    model.eval()
    with torch.no_grad():
        # Ensure inputs are on correct device
        pos = pos.to(device)
        if isinstance(t, (int, float)):
            t = torch.full((len(pos),), t, device=device)
        else:
            t = t.to(device)
        
        # Get velocity at current position
        velocity = model.predict_velocity(pos, t)
        
        # Euler step: pos_new = pos + v * dt
        new_pos = pos + velocity * dt
    
    return new_pos


def rk4_step(model: NeuralFlowField, pos: torch.Tensor, t: torch.Tensor,
            dt: float, device: torch.device) -> torch.Tensor:
    """
    Single RK4 (4th order Runge-Kutta) integration step.
    
    Args:
        model: Neural flow field model
        pos: Current positions, shape (N, 2)
        t: Current time, scalar or shape (N,)
        dt: Time step size
        device: Device for computation
        
    Returns:
        New positions after step, shape (N, 2)
    """
    model.eval()
    with torch.no_grad():
        # Ensure inputs are on correct device
        pos = pos.to(device)
        if isinstance(t, (int, float)):
            t_tensor = torch.full((len(pos),), t, device=device)
        else:
            t_tensor = t.to(device)
        
        # k1 = f(t, y)
        k1 = model.predict_velocity(pos, t_tensor)
        
        # k2 = f(t + dt/2, y + k1*dt/2)
        pos_k2 = pos + k1 * (dt / 2)
        t_k2 = t_tensor + dt / 2
        k2 = model.predict_velocity(pos_k2, t_k2)
        
        # k3 = f(t + dt/2, y + k2*dt/2)
        pos_k3 = pos + k2 * (dt / 2)
        t_k3 = t_tensor + dt / 2
        k3 = model.predict_velocity(pos_k3, t_k3)
        
        # k4 = f(t + dt, y + k3*dt)
        pos_k4 = pos + k3 * dt
        t_k4 = t_tensor + dt
        k4 = model.predict_velocity(pos_k4, t_k4)
        
        # Combine: y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        new_pos = pos + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    
    return new_pos


def track_forward(model: NeuralFlowField, start_positions: np.ndarray,
                 start_time: float, num_steps: int, dt: float,
                 device: torch.device, method: str = 'rk4',
                 show_progress: bool = True) -> np.ndarray:
    """
    Track particles forward in time.
    
    Args:
        model: Trained neural flow field model
        start_positions: Starting positions, shape (N, 2)
        start_time: Starting time
        num_steps: Number of integration steps
        dt: Time step size
        device: Device for computation
        method: Integration method ('euler' or 'rk4')
        show_progress: Whether to show progress bar
        
    Returns:
        Trajectories array of shape (num_steps+1, N, 2)
    """
    if method not in ['euler', 'rk4']:
        raise ValueError(f"Unknown integration method: {method}. Use 'euler' or 'rk4'.")
    
    logger.info(f"Forward tracking {len(start_positions)} particles for {num_steps} steps using {method}")
    
    # Initialize trajectory storage
    num_particles = len(start_positions)
    trajectories = np.zeros((num_steps + 1, num_particles, 2))
    trajectories[0] = start_positions
    
    # Convert to tensor
    pos = torch.tensor(start_positions, dtype=torch.float32, device=device)
    
    # Integration function
    if method == 'euler':
        step_fn = euler_step
    else:
        step_fn = rk4_step
    
    # Integrate
    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator, desc="Forward tracking")
    
    for step in iterator:
        t = start_time + step * dt
        pos = step_fn(model, pos, t, dt, device)
        trajectories[step + 1] = pos.cpu().numpy()
    
    return trajectories


def track_backward(model: NeuralFlowField, end_positions: np.ndarray,
                  end_time: float, num_steps: int, dt: float,
                  device: torch.device, method: str = 'rk4',
                  show_progress: bool = True) -> np.ndarray:
    """
    Track particles backward in time.
    
    Args:
        model: Trained neural flow field model
        end_positions: Ending positions, shape (N, 2)
        end_time: Ending time
        num_steps: Number of integration steps
        dt: Time step size (will be negated for backward tracking)
        device: Device for computation
        method: Integration method ('euler' or 'rk4')
        show_progress: Whether to show progress bar
        
    Returns:
        Trajectories array of shape (num_steps+1, N, 2) from end to start
    """
    logger.info(f"Backward tracking {len(end_positions)} particles for {num_steps} steps using {method}")
    
    # Track backward by negating dt
    trajectories = track_forward(
        model, end_positions, end_time, num_steps, -dt,
        device, method, show_progress
    )
    
    return trajectories


def find_source(model: NeuralFlowField, particle_positions: np.ndarray,
               current_time: float, lookback_time: float, dt: float,
               device: torch.device, method: str = 'rk4') -> np.ndarray:
    """
    Find source positions of particles by tracking backward.
    
    Args:
        model: Trained neural flow field model
        particle_positions: Current positions, shape (N, 2)
        current_time: Current time
        lookback_time: How far back to track
        dt: Time step size
        device: Device for computation
        method: Integration method ('euler' or 'rk4')
        
    Returns:
        Source positions, shape (N, 2)
    """
    num_steps = int(lookback_time / dt)
    trajectories = track_backward(
        model, particle_positions, current_time, num_steps, dt,
        device, method, show_progress=False
    )
    
    # Return final positions (which are the source positions)
    source_positions = trajectories[-1]
    
    return source_positions


def batch_track_forward(model: NeuralFlowField, start_positions: np.ndarray,
                       start_times: np.ndarray, num_steps: int, dt: float,
                       device: torch.device, method: str = 'rk4',
                       batch_size: int = 1000) -> List[np.ndarray]:
    """
    Track multiple particles starting at different times (batched for efficiency).
    
    Args:
        model: Trained neural flow field model
        start_positions: Starting positions, shape (N, 2)
        start_times: Starting times, shape (N,)
        num_steps: Number of integration steps
        dt: Time step size
        device: Device for computation
        method: Integration method ('euler' or 'rk4')
        batch_size: Number of particles to process at once
        
    Returns:
        List of trajectory arrays, one per particle
    """
    num_particles = len(start_positions)
    trajectories_list = []
    
    for i in tqdm(range(0, num_particles, batch_size), desc="Batch tracking"):
        batch_end = min(i + batch_size, num_particles)
        batch_positions = start_positions[i:batch_end]
        batch_times = start_times[i:batch_end]
        
        # Track each particle in batch (they have different start times)
        for j, (pos, t0) in enumerate(zip(batch_positions, batch_times)):
            traj = track_forward(
                model, pos[np.newaxis, :], t0, num_steps, dt,
                device, method, show_progress=False
            )
            trajectories_list.append(traj[:, 0, :])  # Remove batch dimension
    
    return trajectories_list


def compute_trajectory_error(pred_trajectories: np.ndarray,
                            true_trajectories: np.ndarray) -> dict:
    """
    Compute error metrics for predicted trajectories.
    
    Args:
        pred_trajectories: Predicted trajectories, shape (num_steps, N, 2)
        true_trajectories: True trajectories, shape (num_steps, N, 2)
        
    Returns:
        Dictionary containing error metrics
    """
    # Ensure same shape
    assert pred_trajectories.shape == true_trajectories.shape
    
    # Position error at each timestep
    position_error = np.sqrt(np.sum((pred_trajectories - true_trajectories) ** 2, axis=-1))
    
    # Mean error over time
    mean_error_per_timestep = np.mean(position_error, axis=1)
    
    # Final position error
    final_error = position_error[-1]
    
    # Velocity error (finite differences)
    pred_vel = np.diff(pred_trajectories, axis=0)
    true_vel = np.diff(true_trajectories, axis=0)
    velocity_error = np.sqrt(np.sum((pred_vel - true_vel) ** 2, axis=-1))
    
    metrics = {
        'mean_position_error': float(np.mean(position_error)),
        'std_position_error': float(np.std(position_error)),
        'max_position_error': float(np.max(position_error)),
        'final_position_error_mean': float(np.mean(final_error)),
        'final_position_error_std': float(np.std(final_error)),
        'mean_velocity_error': float(np.mean(velocity_error)),
        'position_error_per_timestep': mean_error_per_timestep.tolist()
    }
    
    return metrics


def validate_tracking_accuracy(model: NeuralFlowField, test_trajectories: np.ndarray,
                              test_times: np.ndarray, device: torch.device,
                              method: str = 'rk4', dt: float = 0.1) -> dict:
    """
    Validate tracking accuracy against ground truth trajectories.
    
    Args:
        model: Trained neural flow field model
        test_trajectories: Ground truth trajectories, shape (num_particles, num_steps, 2)
        test_times: Time values for each step, shape (num_steps,)
        device: Device for computation
        method: Integration method
        dt: Time step size
        
    Returns:
        Dictionary of validation metrics
    """
    num_particles, num_steps, _ = test_trajectories.shape
    
    logger.info(f"Validating tracking accuracy on {num_particles} particles")
    
    # Track forward from initial positions
    start_positions = test_trajectories[:, 0, :]
    start_time = test_times[0]
    
    pred_trajectories = track_forward(
        model, start_positions, start_time, num_steps - 1, dt,
        device, method, show_progress=True
    )
    
    # Transpose to match expected shape (num_particles, num_steps, 2)
    pred_trajectories = np.transpose(pred_trajectories, (1, 0, 2))
    
    # Compute metrics
    metrics = compute_trajectory_error(
        pred_trajectories.transpose(1, 0, 2),  # (num_steps, num_particles, 2)
        test_trajectories.transpose(1, 0, 2)
    )
    
    logger.info(f"Validation results: Mean error = {metrics['mean_position_error']:.4f}, "
                f"Final error = {metrics['final_position_error_mean']:.4f}")
    
    return metrics


def streamline_integration(model: NeuralFlowField, seed_points: np.ndarray,
                          time: float, max_steps: int, dt: float,
                          device: torch.device, method: str = 'rk4',
                          bounds: Optional[Tuple[float, float, float, float]] = None) -> List[np.ndarray]:
    """
    Compute streamlines from seed points.
    
    Args:
        model: Trained neural flow field model
        seed_points: Seed point positions, shape (N, 2)
        time: Fixed time for streamline computation
        max_steps: Maximum integration steps per streamline
        dt: Step size
        device: Device for computation
        method: Integration method
        bounds: Optional (x_min, x_max, y_min, y_max) to stop integration at boundaries
        
    Returns:
        List of streamlines, each with shape (num_steps, 2)
    """
    streamlines = []
    
    for seed in tqdm(seed_points, desc="Computing streamlines"):
        positions = [seed]
        pos = torch.tensor(seed, dtype=torch.float32, device=device).unsqueeze(0)
        
        for _ in range(max_steps):
            # Integration step
            if method == 'euler':
                new_pos = euler_step(model, pos, time, dt, device)
            else:
                new_pos = rk4_step(model, pos, time, dt, device)
            
            new_pos_np = new_pos.cpu().numpy()[0]
            
            # Check bounds
            if bounds is not None:
                x_min, x_max, y_min, y_max = bounds
                if (new_pos_np[0] < x_min or new_pos_np[0] > x_max or
                    new_pos_np[1] < y_min or new_pos_np[1] > y_max):
                    break
            
            positions.append(new_pos_np)
            pos = new_pos
            
            # Check for convergence (very small movement)
            if np.linalg.norm(new_pos_np - positions[-2]) < 1e-6:
                break
        
        streamlines.append(np.array(positions))
    
    return streamlines


def compute_particle_ftle(model: NeuralFlowField, positions: np.ndarray,
                         time: float, integration_time: float, dt: float,
                         device: torch.device, epsilon: float = 1e-3,
                         method: str = 'rk4') -> np.ndarray:
    """
    Compute Finite-Time Lyapunov Exponent (FTLE) field.
    
    Args:
        model: Trained neural flow field model
        positions: Grid positions to compute FTLE, shape (N, 2)
        time: Starting time
        integration_time: Total integration time
        dt: Time step size
        device: Device for computation
        epsilon: Perturbation size for computing flow map gradient
        method: Integration method
        
    Returns:
        FTLE values at each position, shape (N,)
    """
    num_steps = int(abs(integration_time) / dt)
    
    # Track particles forward
    final_positions = track_forward(
        model, positions, time, num_steps, dt, device, method, show_progress=False
    )[-1]
    
    # Compute flow map gradient using finite differences
    ftle_values = []
    
    for i, pos in enumerate(positions):
        # Perturb in x and y directions
        pos_dx = pos + np.array([epsilon, 0])
        pos_dy = pos + np.array([0, epsilon])
        
        # Track perturbed particles
        final_dx = track_forward(
            model, pos_dx[np.newaxis, :], time, num_steps, dt, device, method, show_progress=False
        )[-1, 0]
        
        final_dy = track_forward(
            model, pos_dy[np.newaxis, :], time, num_steps, dt, device, method, show_progress=False
        )[-1, 0]
        
        # Compute gradient of flow map
        final_pos = final_positions[i]
        grad_x = (final_dx - final_pos) / epsilon
        grad_y = (final_dy - final_pos) / epsilon
        
        # Jacobian of flow map
        J = np.array([grad_x, grad_y]).T
        
        # Cauchy-Green strain tensor
        C = J.T @ J
        
        # FTLE = 1/(2T) * log(max eigenvalue of C)
        max_eigenvalue = np.max(np.linalg.eigvals(C))
        ftle = (1 / (2 * abs(integration_time))) * np.log(max_eigenvalue)
        
        ftle_values.append(ftle.real)
    
    return np.array(ftle_values)
