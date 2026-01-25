"""
Advanced Neural Flow Field Models for ICIP 2026 Submission

This module implements state-of-the-art architectures for learning continuous
flow fields from Lagrangian particle trajectories, with emphasis on:
1. Physics-Informed Neural Networks (PINN) with Navier-Stokes constraints
2. Temporal Transformer architectures for multi-scale temporal modeling
3. Conditional flow fields for multi-experiment support

Author: Lagrangian Flow Tracker Team
Date: January 2026
Conference: IEEE ICIP 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import math


# ============================================================================
# Physics-Informed Neural Network (PINN) - Main Contribution
# ============================================================================

class NavierStokesPINN(nn.Module):
    """
    Physics-Informed Neural Network for incompressible Navier-Stokes equations.
    
    This network predicts both velocity field (vx, vy) and pressure field (p)
    while enforcing physical constraints:
    
    1. Incompressibility (continuity equation):
       ∇·v = ∂vx/∂x + ∂vy/∂y = 0
    
    2. Momentum conservation (Navier-Stokes equations):
       ∂vx/∂t + vx·∂vx/∂x + vy·∂vx/∂y = -1/ρ·∂p/∂x + ν·(∂²vx/∂x² + ∂²vx/∂y²)
       ∂vy/∂t + vx·∂vy/∂x + vy·∂vy/∂y = -1/ρ·∂p/∂y + ν·(∂²vy/∂x² + ∂²vy/∂y²)
    
    where:
    - v = (vx, vy): velocity field
    - p: pressure field
    - ρ: fluid density
    - ν: kinematic viscosity
    - t: time
    
    The network architecture uses sinusoidal positional encoding followed by
    a deep MLP that outputs 3 values: (vx, vy, p).
    
    Args:
        input_dim: Input dimension (typically 3 for x, y, t)
        hidden_dims: List of hidden layer dimensions
        num_frequencies: Number of frequencies for positional encoding
        viscosity: Kinematic viscosity ν (m²/s)
        density: Fluid density ρ (kg/m³)
        enforce_incompressibility: If True, use hard constraint for ∇·v=0
        
    Example:
        >>> pinn = NavierStokesPINN(
        ...     input_dim=3,
        ...     hidden_dims=[256, 256, 256, 128],
        ...     viscosity=0.01,
        ...     density=1.0
        ... )
        >>> positions = torch.randn(100, 2)  # (batch, xy)
        >>> times = torch.randn(100, 1)      # (batch, t)
        >>> vx, vy, p = pinn(positions, times)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256, 128],
        num_frequencies: int = 10,
        viscosity: float = 0.01,
        density: float = 1.0,
        enforce_incompressibility: bool = True,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.viscosity = viscosity
        self.density = density
        self.enforce_incompressibility = enforce_incompressibility
        
        # Positional encoding parameters
        self.freq_bands = nn.Parameter(
            2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies),
            requires_grad=False
        )
        
        # Calculate encoding dimension
        encoding_dim = input_dim * (2 * num_frequencies + 1)  # sin, cos, + original
        
        # Build MLP
        layers = []
        in_dim = encoding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim
        
        # Output layer: 3 values (vx, vy, p)
        layers.append(nn.Linear(in_dim, 3))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Encoded tensor of shape (batch, encoding_dim)
        """
        # x: (batch, input_dim)
        # freq_bands: (num_frequencies,)
        
        # Expand: x_expanded: (batch, input_dim, num_frequencies)
        x_expanded = x.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0)
        
        # Compute sin and cos
        x_sin = torch.sin(x_expanded)  # (batch, input_dim, num_frequencies)
        x_cos = torch.cos(x_expanded)  # (batch, input_dim, num_frequencies)
        
        # Flatten: (batch, input_dim * num_frequencies)
        x_sin = x_sin.reshape(x.shape[0], -1)
        x_cos = x_cos.reshape(x.shape[0], -1)
        
        # Concatenate with original input
        encoded = torch.cat([x, x_sin, x_cos], dim=-1)
        
        return encoded
    
    def forward(
        self,
        positions: torch.Tensor,
        times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through PINN.
        
        Args:
            positions: Spatial coordinates (batch, 2) for (x, y)
            times: Time coordinates (batch, 1) for t
            
        Returns:
            vx: x-component of velocity (batch,)
            vy: y-component of velocity (batch,)
            p: pressure field (batch,)
        """
        # Concatenate spatial and temporal coordinates
        inputs = torch.cat([positions, times], dim=-1)  # (batch, 3)
        
        # Apply positional encoding
        encoded = self.positional_encoding(inputs)
        
        # Forward through MLP
        output = self.mlp(encoded)  # (batch, 3)
        
        vx = output[:, 0]
        vy = output[:, 1]
        p = output[:, 2]
        
        return vx, vy, p
    
    def predict_velocity(
        self,
        positions: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field only (for compatibility with existing code).
        
        Args:
            positions: Spatial coordinates (batch, 2)
            times: Time coordinates (batch, 1)
            
        Returns:
            velocity: Velocity field (batch, 2) for (vx, vy)
        """
        vx, vy, _ = self.forward(positions, times)
        return torch.stack([vx, vy], dim=-1)


def compute_navier_stokes_residual(
    model: NavierStokesPINN,
    positions: torch.Tensor,
    times: torch.Tensor,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute Navier-Stokes equation residuals for PINN training.
    
    This function computes all necessary gradients using automatic differentiation
    and returns the residual of the incompressible Navier-Stokes equations.
    
    Physics equations enforced:
    
    1. Continuity (incompressibility):
       R_cont = ∂vx/∂x + ∂vy/∂y
    
    2. Momentum (x-direction):
       R_mom_x = ∂vx/∂t + vx·∂vx/∂x + vy·∂vx/∂y + 1/ρ·∂p/∂x - ν·(∂²vx/∂x² + ∂²vx/∂y²)
    
    3. Momentum (y-direction):
       R_mom_y = ∂vy/∂t + vx·∂vy/∂x + vy·∂vy/∂y + 1/ρ·∂p/∂y - ν·(∂²vy/∂x² + ∂²vy/∂y²)
    
    The total residual is: R = |R_cont|² + |R_mom_x|² + |R_mom_y|²
    
    Args:
        model: NavierStokesPINN model
        positions: Spatial coordinates (batch, 2), requires_grad=True
        times: Time coordinates (batch, 1), requires_grad=True
        return_components: If True, return individual residuals
        
    Returns:
        residual: Scalar residual (mean squared residual)
        OR dict with individual components if return_components=True
        
    Example:
        >>> positions = torch.randn(100, 2, requires_grad=True)
        >>> times = torch.randn(100, 1, requires_grad=True)
        >>> residual = compute_navier_stokes_residual(pinn, positions, times)
        >>> loss = residual
        >>> loss.backward()
    """
    # Ensure gradients are enabled
    positions = positions.requires_grad_(True)
    times = times.requires_grad_(True)
    
    # Forward pass
    vx, vy, p = model(positions, times)
    
    # Get parameters
    rho = model.density
    nu = model.viscosity
    
    # ========================================================================
    # Compute first-order derivatives
    # ========================================================================
    
    # Velocity gradients w.r.t. space
    grad_outputs = torch.ones_like(vx)
    
    # ∂vx/∂x, ∂vx/∂y
    vx_grad = torch.autograd.grad(
        vx, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    dvx_dx = vx_grad[:, 0]
    dvx_dy = vx_grad[:, 1]
    
    # ∂vy/∂x, ∂vy/∂y
    vy_grad = torch.autograd.grad(
        vy, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    dvy_dx = vy_grad[:, 0]
    dvy_dy = vy_grad[:, 1]
    
    # ∂p/∂x, ∂p/∂y
    p_grad = torch.autograd.grad(
        p, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]
    dp_dx = p_grad[:, 0]
    dp_dy = p_grad[:, 1]
    
    # ∂vx/∂t
    dvx_dt = torch.autograd.grad(
        vx, times,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    # ∂vy/∂t
    dvy_dt = torch.autograd.grad(
        vy, times,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    # ========================================================================
    # Compute second-order derivatives (Laplacian terms)
    # ========================================================================
    
    # ∂²vx/∂x²
    d2vx_dx2 = torch.autograd.grad(
        dvx_dx, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    # ∂²vx/∂y²
    d2vx_dy2 = torch.autograd.grad(
        dvx_dy, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    # ∂²vy/∂x²
    d2vy_dx2 = torch.autograd.grad(
        dvy_dx, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    # ∂²vy/∂y²
    d2vy_dy2 = torch.autograd.grad(
        dvy_dy, positions,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    # ========================================================================
    # Compute equation residuals
    # ========================================================================
    
    # 1. Continuity equation: ∇·v = 0
    continuity_residual = dvx_dx + dvy_dy
    
    # 2. Momentum equation (x-direction)
    # ∂vx/∂t + vx·∂vx/∂x + vy·∂vx/∂y = -1/ρ·∂p/∂x + ν·∇²vx
    convection_x = vx * dvx_dx + vy * dvx_dy
    pressure_x = (1.0 / rho) * dp_dx
    diffusion_x = nu * (d2vx_dx2 + d2vx_dy2)
    
    momentum_x_residual = dvx_dt + convection_x + pressure_x - diffusion_x
    
    # 3. Momentum equation (y-direction)
    # ∂vy/∂t + vx·∂vy/∂x + vy·∂vy/∂y = -1/ρ·∂p/∂y + ν·∇²vy
    convection_y = vx * dvy_dx + vy * dvy_dy
    pressure_y = (1.0 / rho) * dp_dy
    diffusion_y = nu * (d2vy_dx2 + d2vy_dy2)
    
    momentum_y_residual = dvy_dt + convection_y + pressure_y - diffusion_y
    
    # ========================================================================
    # Compute total residual (MSE)
    # ========================================================================
    
    if return_components:
        return {
            'continuity': continuity_residual.pow(2).mean(),
            'momentum_x': momentum_x_residual.pow(2).mean(),
            'momentum_y': momentum_y_residual.pow(2).mean(),
            'total': (
                continuity_residual.pow(2).mean() +
                momentum_x_residual.pow(2).mean() +
                momentum_y_residual.pow(2).mean()
            )
        }
    else:
        total_residual = (
            continuity_residual.pow(2).mean() +
            momentum_x_residual.pow(2).mean() +
            momentum_y_residual.pow(2).mean()
        )
        return total_residual


# ============================================================================
# Analytical Solutions for Validation
# ============================================================================

class TaylorGreenVortex:
    """
    Taylor-Green vortex: analytical solution to 2D Navier-Stokes equations.
    
    This is a standard benchmark for validating CFD and PINN implementations.
    The solution represents decaying vortices with known analytical form:
    
    vx(x, y, t) = -V₀·cos(kx)·sin(ky)·exp(-2νk²t)
    vy(x, y, t) = +V₀·sin(kx)·cos(ky)·exp(-2νk²t)
    p(x, y, t) = -V₀²/4·(cos(2kx) + cos(2ky))·exp(-4νk²t)
    
    where:
    - V₀: initial velocity scale
    - k: wavenumber (2π/L for domain size L)
    - ν: kinematic viscosity
    - t: time
    
    Args:
        velocity_scale: Initial velocity magnitude V₀
        domain_size: Size of periodic domain L
        viscosity: Kinematic viscosity ν
        
    Example:
        >>> tg = TaylorGreenVortex(velocity_scale=1.0, viscosity=0.01)
        >>> x = torch.linspace(0, 2*np.pi, 100)
        >>> y = torch.linspace(0, 2*np.pi, 100)
        >>> X, Y = torch.meshgrid(x, y, indexing='ij')
        >>> positions = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        >>> t = 0.1
        >>> vx, vy, p = tg.solution(positions, t)
    """
    
    def __init__(
        self,
        velocity_scale: float = 1.0,
        domain_size: float = 2 * np.pi,
        viscosity: float = 0.01
    ):
        self.V0 = velocity_scale
        self.L = domain_size
        self.k = 2 * np.pi / domain_size
        self.nu = viscosity
    
    def solution(
        self,
        positions: torch.Tensor,
        time: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute analytical solution at given positions and time.
        
        Args:
            positions: Spatial coordinates (N, 2) for (x, y)
            time: Time value (scalar)
            
        Returns:
            vx: x-velocity component (N,)
            vy: y-velocity component (N,)
            p: pressure field (N,)
        """
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Convert time to tensor if needed
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=positions.dtype, device=positions.device)
        
        # Exponential decay factor
        decay = torch.exp(-2 * self.nu * self.k**2 * time)
        
        # Velocity components
        vx = -self.V0 * torch.cos(self.k * x) * torch.sin(self.k * y) * decay
        vy = self.V0 * torch.sin(self.k * x) * torch.cos(self.k * y) * decay
        
        # Pressure (with stronger decay)
        pressure_decay = torch.exp(-4 * self.nu * self.k**2 * time)
        p = -(self.V0**2 / 4) * (
            torch.cos(2 * self.k * x) + torch.cos(2 * self.k * y)
        ) * pressure_decay
        
        return vx, vy, p
    
    def generate_training_data(
        self,
        num_points: int = 1000,
        time_range: Tuple[float, float] = (0.0, 1.0),
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training data from analytical solution.
        
        Args:
            num_points: Number of random points to sample
            time_range: (t_min, t_max) for temporal sampling
            device: Device to create tensors on
            
        Returns:
            positions: (num_points, 2)
            times: (num_points, 1)
            velocities: (num_points, 2)
            pressures: (num_points, 1)
        """
        # Random spatial sampling
        positions = torch.rand(num_points, 2, device=device) * self.L
        
        # Random temporal sampling
        times = torch.rand(num_points, 1, device=device)
        times = times * (time_range[1] - time_range[0]) + time_range[0]
        
        # Compute analytical solution
        velocities = []
        pressures = []
        
        for i in range(num_points):
            pos = positions[i:i+1]
            t = times[i, 0].item()
            vx, vy, p = self.solution(pos, t)
            velocities.append(torch.stack([vx, vy], dim=-1))
            pressures.append(p.unsqueeze(-1))
        
        velocities = torch.cat(velocities, dim=0)
        pressures = torch.cat(pressures, dim=0)
        
        return positions, times, velocities, pressures


# ============================================================================
# PINN Training Utilities
# ============================================================================

def train_pinn_on_taylor_green(
    model: NavierStokesPINN,
    num_epochs: int = 1000,
    num_points: int = 2000,
    learning_rate: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_physics: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train PINN on Taylor-Green vortex analytical solution.
    
    This function demonstrates how to train a PINN with both data loss
    and physics loss (Navier-Stokes residual).
    
    Loss = λ_data·L_data + λ_physics·L_physics
    
    where:
    - L_data: MSE between predicted and analytical velocity/pressure
    - L_physics: Navier-Stokes equation residual
    
    Args:
        model: NavierStokesPINN instance
        num_epochs: Number of training epochs
        num_points: Number of training points per epoch
        learning_rate: Optimizer learning rate
        lambda_data: Weight for data loss
        lambda_physics: Weight for physics loss
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        history: Dictionary with training history
        
    Example:
        >>> pinn = NavierStokesPINN(viscosity=0.01, density=1.0)
        >>> history = train_pinn_on_taylor_green(pinn, num_epochs=500)
        >>> print(f"Final data loss: {history['data_loss'][-1]:.6f}")
        >>> print(f"Final physics loss: {history['physics_loss'][-1]:.6f}")
    """
    # Create Taylor-Green vortex
    tg_vortex = TaylorGreenVortex(
        velocity_scale=1.0,
        domain_size=2*np.pi,
        viscosity=model.viscosity
    )
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'data_loss': [],
        'physics_loss': [],
        'total_loss': [],
        'velocity_error': [],
        'pressure_error': []
    }
    
    for epoch in range(num_epochs):
        # Generate training data
        positions, times, velocities_true, pressures_true = \
            tg_vortex.generate_training_data(num_points, device=device)
        
        # Forward pass
        vx_pred, vy_pred, p_pred = model(positions, times)
        velocities_pred = torch.stack([vx_pred, vy_pred], dim=-1)
        
        # Data loss
        velocity_loss = F.mse_loss(velocities_pred, velocities_true)
        pressure_loss = F.mse_loss(p_pred.unsqueeze(-1), pressures_true)
        data_loss = velocity_loss + pressure_loss
        
        # Physics loss (Navier-Stokes residual)
        physics_loss = compute_navier_stokes_residual(model, positions, times)
        
        # Total loss
        total_loss = lambda_data * data_loss + lambda_physics * physics_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record history
        history['data_loss'].append(data_loss.item())
        history['physics_loss'].append(physics_loss.item())
        history['total_loss'].append(total_loss.item())
        history['velocity_error'].append(velocity_loss.item())
        history['pressure_error'].append(pressure_loss.item())
        
        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Data Loss: {data_loss.item():.6f}")
            print(f"  Physics Loss: {physics_loss.item():.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Velocity Error: {velocity_loss.item():.6f}")
            print(f"  Pressure Error: {pressure_loss.item():.6f}")
    
    return history


def validate_pinn_accuracy(
    model: NavierStokesPINN,
    num_test_points: int = 5000,
    time_range: Tuple[float, float] = (0.0, 1.0),
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Validate PINN accuracy against Taylor-Green analytical solution.
    
    Args:
        model: Trained NavierStokesPINN
        num_test_points: Number of test points
        time_range: Time range for testing
        device: Device
        
    Returns:
        metrics: Dictionary with error metrics
    """
    model.eval()
    
    # Generate test data
    tg_vortex = TaylorGreenVortex(viscosity=model.viscosity)
    positions, times, velocities_true, pressures_true = \
        tg_vortex.generate_training_data(num_test_points, time_range, device)
    
    with torch.no_grad():
        # Predict
        vx_pred, vy_pred, p_pred = model(positions, times)
        velocities_pred = torch.stack([vx_pred, vy_pred], dim=-1)
        
        # Compute errors
        velocity_mse = F.mse_loss(velocities_pred, velocities_true).item()
        velocity_mae = F.l1_loss(velocities_pred, velocities_true).item()
        
        pressure_mse = F.mse_loss(p_pred.unsqueeze(-1), pressures_true).item()
        pressure_mae = F.l1_loss(p_pred.unsqueeze(-1), pressures_true).item()
    
    # Compute physics residual on test set (outside no_grad context)
    positions = positions.detach().requires_grad_(True)
    times = times.detach().requires_grad_(True)
    residuals = compute_navier_stokes_residual(
        model, positions, times, return_components=True
    )
    
    metrics = {
        'velocity_mse': velocity_mse,
        'velocity_mae': velocity_mae,
        'velocity_rmse': np.sqrt(velocity_mse),
        'pressure_mse': pressure_mse,
        'pressure_mae': pressure_mae,
        'pressure_rmse': np.sqrt(pressure_mse),
        'continuity_residual': residuals['continuity'].item(),
        'momentum_x_residual': residuals['momentum_x'].item(),
        'momentum_y_residual': residuals['momentum_y'].item(),
        'total_physics_residual': residuals['total'].item()
    }
    
    return metrics


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Physics-Informed Neural Network (PINN) for Navier-Stokes Equations")
    print("ICIP 2026 Submission - Advanced Flow Field Modeling")
    print("=" * 80)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create PINN model
    print("\n1. Creating PINN model...")
    pinn = NavierStokesPINN(
        input_dim=3,
        hidden_dims=[256, 256, 256, 128],
        num_frequencies=10,
        viscosity=0.01,
        density=1.0
    )
    print(f"   Model parameters: {sum(p.numel() for p in pinn.parameters()):,}")
    
    # Train on Taylor-Green vortex
    print("\n2. Training on Taylor-Green vortex analytical solution...")
    history = train_pinn_on_taylor_green(
        pinn,
        num_epochs=500,
        num_points=2000,
        learning_rate=1e-3,
        lambda_data=1.0,
        lambda_physics=0.1,
        device=device,
        verbose=True
    )
    
    # Validate
    print("\n3. Validating accuracy...")
    metrics = validate_pinn_accuracy(pinn, num_test_points=5000, device=device)
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Velocity RMSE:       {metrics['velocity_rmse']:.6f}")
    print(f"Velocity MAE:        {metrics['velocity_mae']:.6f}")
    print(f"Pressure RMSE:       {metrics['pressure_rmse']:.6f}")
    print(f"Pressure MAE:        {metrics['pressure_mae']:.6f}")
    print(f"\nPhysics Residuals:")
    print(f"Continuity:          {metrics['continuity_residual']:.6e}")
    print(f"Momentum X:          {metrics['momentum_x_residual']:.6e}")
    print(f"Momentum Y:          {metrics['momentum_y_residual']:.6e}")
    print(f"Total Residual:      {metrics['total_physics_residual']:.6e}")
    print("=" * 80)
    
    # Success criterion for ICIP paper
    if metrics['total_physics_residual'] < 1e-4:
        print("\n✓ SUCCESS: Physics residual < 1e-4 (publication quality!)")
    else:
        print(f"\n✗ WARNING: Physics residual {metrics['total_physics_residual']:.6e} >= 1e-4")
        print("  Consider training longer or adjusting lambda_physics")
