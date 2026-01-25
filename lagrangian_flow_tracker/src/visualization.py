"""Visualization tools for flow fields, trajectories, and analysis."""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
import torch
from tqdm import tqdm

from .model import NeuralFlowField


logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_trajectories(trajectories: np.ndarray, frame: Optional[np.ndarray] = None,
                     particle_ids: Optional[List[int]] = None,
                     save_path: Optional[str] = None, title: str = "Particle Trajectories",
                     colormap: str = 'tab10', show: bool = False) -> None:
    """
    Plot particle trajectories overlaid on frame.
    
    Args:
        trajectories: DataFrame or array with columns/fields: particle_id, x, y (or shape (N, timesteps, 2))
        frame: Optional background frame image
        particle_ids: Optional list of particle IDs to plot
        save_path: Path to save figure
        title: Plot title
        colormap: Matplotlib colormap name
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot background frame if provided
    if frame is not None:
        ax.imshow(frame, cmap='gray', alpha=0.5)
    
    # Handle different trajectory formats
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
        # Shape: (num_particles, num_timesteps, 2)
        num_particles = len(trajectories)
        colors = plt.cm.get_cmap(colormap, num_particles)
        
        for i in range(num_particles):
            if particle_ids is not None and i not in particle_ids:
                continue
            
            traj = trajectories[i]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors(i), alpha=0.7, linewidth=1.5)
            ax.scatter(traj[0, 0], traj[0, 1], color=colors(i), marker='o', s=50, edgecolors='black', linewidths=1, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=colors(i), marker='s', s=50, edgecolors='black', linewidths=1, zorder=5)
    
    else:
        # Assume DataFrame-like with particle_id, x, y columns
        import pandas as pd
        if not isinstance(trajectories, pd.DataFrame):
            trajectories = pd.DataFrame(trajectories)
        
        unique_ids = trajectories['particle_id'].unique()
        if particle_ids is not None:
            unique_ids = [pid for pid in unique_ids if pid in particle_ids]
        
        colors = plt.cm.get_cmap(colormap, len(unique_ids))
        
        for i, pid in enumerate(unique_ids):
            particle_data = trajectories[trajectories['particle_id'] == pid].sort_values('frame')
            x = particle_data['x'].values
            y = particle_data['y'].values
            
            ax.plot(x, y, '-', color=colors(i), alpha=0.7, linewidth=1.5, label=f'Particle {pid}')
            ax.scatter(x[0], y[0], color=colors(i), marker='o', s=50, edgecolors='black', linewidths=1, zorder=5)
            ax.scatter(x[-1], y[-1], color=colors(i), marker='s', s=50, edgecolors='black', linewidths=1, zorder=5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.legend(['Start', 'End'], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved trajectory plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_velocity_field(model: NeuralFlowField, time: float,
                       bounds: Tuple[float, float, float, float],
                       resolution: int = 30, device: Optional[torch.device] = None,
                       save_path: Optional[str] = None, title: Optional[str] = None,
                       show: bool = False, scale: Optional[float] = None) -> None:
    """
    Plot velocity field as quiver plot at a specific time.
    
    Args:
        model: Trained neural flow field model
        time: Time to evaluate velocity field
        bounds: (x_min, x_max, y_min, y_max)
        resolution: Number of grid points in each dimension
        device: Device for computation
        save_path: Path to save figure
        title: Plot title
        show: Whether to display the plot
        scale: Quiver scale (None for auto)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_min, x_max, y_min, y_max = bounds
    
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, time)
    
    # Predict velocities
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device)
        t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)
        
        velocity = model(x_tensor, y_tensor, t_tensor)
        vx = velocity[:, 0].cpu().numpy()
        vy = velocity[:, 1].cpu().numpy()
    
    # Reshape
    VX = vx.reshape(X.shape)
    VY = vy.reshape(Y.shape)
    
    # Compute velocity magnitude
    speed = np.sqrt(VX**2 + VY**2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Velocity magnitude as background
    im = ax.contourf(X, Y, speed, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    # Quiver plot
    if scale is None:
        q = ax.quiver(X, Y, VX, VY, color='black', alpha=0.7, width=0.003)
    else:
        q = ax.quiver(X, Y, VX, VY, color='black', alpha=0.7, width=0.003, scale=scale)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if title is None:
        title = f'Velocity Field at t={time:.2f}'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved velocity field plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_streamlines(model: NeuralFlowField, time: float,
                    bounds: Tuple[float, float, float, float],
                    num_seeds: int = 20, device: Optional[torch.device] = None,
                    save_path: Optional[str] = None, title: Optional[str] = None,
                    show: bool = False) -> None:
    """
    Plot streamlines of velocity field.
    
    Args:
        model: Trained neural flow field model
        time: Time to evaluate streamlines
        bounds: (x_min, x_max, y_min, y_max)
        num_seeds: Number of seed points for streamlines
        device: Device for computation
        save_path: Path to save figure
        title: Plot title
        show: Whether to display the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_min, x_max, y_min, y_max = bounds
    
    # Create velocity field grid
    resolution = 50
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Predict velocities
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, time)
    
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device)
        t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device)
        
        velocity = model(x_tensor, y_tensor, t_tensor)
        vx = velocity[:, 0].cpu().numpy().reshape(X.shape)
        vy = velocity[:, 1].cpu().numpy().reshape(Y.shape)
    
    # Compute speed for background
    speed = np.sqrt(vx**2 + vy**2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Background: velocity magnitude
    im = ax.contourf(X, Y, speed, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    # Streamlines
    ax.streamplot(X, Y, vx, vy, color='white', linewidth=1.5, density=1.5, arrowsize=1.5, arrowstyle='->')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if title is None:
        title = f'Streamlines at t={time:.2f}'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved streamlines plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def animate_tracking(model: NeuralFlowField, particles: np.ndarray,
                    time_range: Tuple[float, float], dt: float,
                    bounds: Tuple[float, float, float, float],
                    device: Optional[torch.device] = None,
                    save_path: Optional[str] = None, fps: int = 30) -> None:
    """
    Create animation of particle tracking.
    
    Args:
        model: Trained neural flow field model
        particles: Initial particle positions, shape (N, 2)
        time_range: (start_time, end_time)
        dt: Time step
        bounds: (x_min, x_max, y_min, y_max)
        device: Device for computation
        save_path: Path to save animation (MP4)
        fps: Frames per second for animation
    """
    from .tracking import track_forward
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time, end_time = time_range
    num_steps = int((end_time - start_time) / dt)
    
    # Track particles
    logger.info(f"Computing trajectories for animation ({num_steps} steps)")
    trajectories = track_forward(model, particles, start_time, num_steps, dt, device, method='rk4')
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    x_min, x_max, y_min, y_max = bounds
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], c='red', s=20, alpha=0.8, edgecolors='black', linewidths=0.5)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Tracking Animation')
    ax.grid(True, alpha=0.3)
    
    # Trail lines
    trail_lines = [ax.plot([], [], 'r-', alpha=0.3, linewidth=0.8)[0] for _ in range(len(particles))]
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        for line in trail_lines:
            line.set_data([], [])
        return [scatter, time_text] + trail_lines
    
    def update(frame):
        # Current positions
        current_positions = trajectories[frame]
        scatter.set_offsets(current_positions)
        
        # Update time text
        current_time = start_time + frame * dt
        time_text.set_text(f't = {current_time:.2f}')
        
        # Update trails (last 20 frames)
        trail_length = 20
        start_frame = max(0, frame - trail_length)
        for i, line in enumerate(trail_lines):
            trail = trajectories[start_frame:frame+1, i, :]
            line.set_data(trail[:, 0], trail[:, 1])
        
        return [scatter, time_text] + trail_lines
    
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=num_steps,
                                  interval=1000/fps, blit=True, repeat=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        logger.info(f"Saving animation to {save_path} (this may take a while...)")
        Writer = animation.writers['pillow' if save_path.endswith('.gif') else 'ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        logger.info(f"Saved animation to {save_path}")
    
    plt.close()


def plot_error_analysis(predicted: np.ndarray, actual: np.ndarray,
                       save_path: Optional[str] = None, show: bool = False) -> None:
    """
    Plot error analysis between predicted and actual trajectories.
    
    Args:
        predicted: Predicted trajectories, shape (num_steps, N, 2) or (N, 2)
        actual: Actual trajectories, same shape as predicted
        save_path: Path to save figure
        show: Whether to display the plot
    """
    # Compute errors
    if predicted.ndim == 3:
        # Time series of positions
        position_error = np.sqrt(np.sum((predicted - actual) ** 2, axis=-1))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error over time
        mean_error = np.mean(position_error, axis=1)
        std_error = np.std(position_error, axis=1)
        timesteps = np.arange(len(mean_error))
        
        axes[0, 0].plot(timesteps, mean_error, 'b-', linewidth=2, label='Mean Error')
        axes[0, 0].fill_between(timesteps, mean_error - std_error, mean_error + std_error,
                               alpha=0.3, label='±1 Std Dev')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Position Error')
        axes[0, 0].set_title('Error vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 1].hist(position_error.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Position Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].axvline(np.mean(position_error), color='r', linestyle='--', label=f'Mean: {np.mean(position_error):.4f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # X component error
        x_error = predicted[:, :, 0] - actual[:, :, 0]
        axes[1, 0].hist(x_error.flatten(), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('X Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('X Component Error Distribution')
        axes[1, 0].axvline(np.mean(x_error), color='r', linestyle='--', label=f'Mean: {np.mean(x_error):.4f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y component error
        y_error = predicted[:, :, 1] - actual[:, :, 1]
        axes[1, 1].hist(y_error.flatten(), bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Y Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Y Component Error Distribution')
        axes[1, 1].axvline(np.mean(y_error), color='r', linestyle='--', label=f'Mean: {np.mean(y_error):.4f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        # Single positions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(actual[:, 0], actual[:, 1], c='blue', label='Actual', alpha=0.6, s=30)
        axes[0].scatter(predicted[:, 0], predicted[:, 1], c='red', marker='x', label='Predicted', alpha=0.6, s=30)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Predicted vs Actual Positions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        # Error magnitude histogram
        error = np.sqrt(np.sum((predicted - actual) ** 2, axis=-1))
        axes[1].hist(error, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Position Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Error Distribution (Mean: {np.mean(error):.4f})')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved error analysis plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_divergence_field(model: NeuralFlowField, time: float,
                         bounds: Tuple[float, float, float, float],
                         resolution: int = 30, device: Optional[torch.device] = None,
                         save_path: Optional[str] = None, show: bool = False) -> None:
    """
    Plot divergence field at a specific time.
    
    Args:
        model: Trained neural flow field model
        time: Time to evaluate divergence
        bounds: (x_min, x_max, y_min, y_max)
        resolution: Number of grid points in each dimension
        device: Device for computation
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_min, x_max, y_min, y_max = bounds
    
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute divergence
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, time)
    
    logger.info(f"Computing divergence field at {resolution}x{resolution} points")
    
    model.eval()
    divergence_values = []
    
    # Batch processing for efficiency
    batch_size = 100
    for i in tqdm(range(0, len(x_flat), batch_size), desc="Computing divergence"):
        batch_end = min(i + batch_size, len(x_flat))
        x_batch = torch.tensor(x_flat[i:batch_end], dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_flat[i:batch_end], dtype=torch.float32, device=device)
        t_batch = torch.tensor(t_flat[i:batch_end], dtype=torch.float32, device=device)
        
        div = model.compute_divergence(x_batch, y_batch, t_batch)
        divergence_values.append(div.detach().cpu().numpy())
    
    divergence = np.concatenate(divergence_values)
    DIV = divergence.reshape(X.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use symmetric colormap centered at zero
    vmax = np.max(np.abs(DIV))
    im = ax.contourf(X, Y, DIV, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Divergence')
    
    # Add zero contour
    ax.contour(X, Y, DIV, levels=[0], colors='black', linewidths=2, linestyles='--')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Divergence Field at t={time:.2f}')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved divergence field plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None,
                         show: bool = False) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Metrics
    if 'val_mae' in history:
        axes[1].plot(epochs, history['val_mae'], 'g-', label='Val MAE', linewidth=2)
    if 'val_rmse' in history:
        axes[1].plot(epochs, history['val_rmse'], 'm-', label='Val RMSE', linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
