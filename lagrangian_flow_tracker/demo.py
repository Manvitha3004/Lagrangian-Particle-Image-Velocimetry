"""Quick start example for Lagrangian Flow Tracker.

This script demonstrates basic usage of the neural flow field model.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import modules
from src.model import NeuralFlowField
from src.utils import set_seed, get_device
from src.tracking import track_forward, rk4_step
from src.visualization import plot_velocity_field, plot_trajectories


def quick_demo():
    """Run a quick demonstration of the system."""
    
    print("=" * 80)
    print("Lagrangian Flow Tracker - Quick Demo")
    print("=" * 80)
    
    # Setup
    set_seed(42)
    device = get_device()
    print(f"\n✓ Using device: {device}")
    
    # Create a simple model
    print("\n1. Creating neural flow field model...")
    model = NeuralFlowField(
        num_frequencies=6,
        hidden_dims=[128, 128, 64],
        activation='relu'
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {num_params:,} parameters")
    
    # Test forward pass
    print("\n2. Testing velocity prediction...")
    with torch.no_grad():
        x = torch.tensor([0.5], device=device)
        y = torch.tensor([0.5], device=device)
        t = torch.tensor([0.0], device=device)
        
        velocity = model(x, y, t)
        print(f"   Velocity at (0.5, 0.5, 0.0): [{velocity[0, 0]:.4f}, {velocity[0, 1]:.4f}]")
    
    # Create synthetic flow field for demonstration
    print("\n3. Creating synthetic vortex flow for demonstration...")
    
    # Define a simple vortex flow
    def vortex_flow(x, y, t, strength=1.0):
        """Clockwise vortex centered at (0.5, 0.5)."""
        cx, cy = 0.5, 0.5
        dx, dy = x - cx, y - cy
        r = torch.sqrt(dx**2 + dy**2 + 1e-8)
        vx = -strength * dy / r
        vy = strength * dx / r
        return torch.stack([vx, vy], dim=-1)
    
    # Generate training data
    num_samples = 1000
    x_train = torch.rand(num_samples, device=device)
    y_train = torch.rand(num_samples, device=device)
    t_train = torch.zeros(num_samples, device=device)
    v_train = vortex_flow(x_train, y_train, t_train)
    
    print(f"   Generated {num_samples} training samples")
    
    # Quick training loop
    print("\n4. Training model (10 iterations)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Predict
        v_pred = model(x_train, y_train, t_train)
        
        # Loss
        loss = torch.mean((v_pred - v_train) ** 2)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/10, Loss: {loss.item():.6f}")
    
    print("   ✓ Training completed")
    
    # Track particles
    print("\n5. Tracking particles...")
    num_particles = 20
    start_positions = np.random.uniform(0.2, 0.8, size=(num_particles, 2))
    
    trajectories = track_forward(
        model, start_positions, start_time=0.0,
        num_steps=50, dt=0.02, device=device,
        method='rk4', show_progress=False
    )
    
    print(f"   ✓ Tracked {num_particles} particles for 50 steps")
    print(f"   Trajectory shape: {trajectories.shape}")
    
    # Visualize
    print("\n6. Creating visualizations...")
    
    # Create output directory
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Plot velocity field
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    x_grid = np.linspace(0, 1, 20)
    y_grid = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Predict velocities
    with torch.no_grad():
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
        t_flat = torch.zeros_like(x_flat)
        
        v = model(x_flat, y_flat, t_flat)
        vx = v[:, 0].cpu().numpy().reshape(X.shape)
        vy = v[:, 1].cpu().numpy().reshape(Y.shape)
    
    # Plot
    speed = np.sqrt(vx**2 + vy**2)
    im = ax.contourf(X, Y, speed, levels=15, cmap='viridis', alpha=0.6)
    ax.quiver(X, Y, vx, vy, color='white', alpha=0.8, scale=15)
    
    # Overlay trajectories
    for i in range(num_particles):
        traj = trajectories[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Neural Flow Field - Vortex Demo')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    save_path = output_dir / 'demo_flow_field.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {save_path}")
    plt.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Prepare your data (flow files or videos)")
    print("  2. Update configs/config.yaml with your data paths")
    print("  3. Run: python train.py --config configs/config.yaml")
    print("  4. Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth")
    print("  5. Track: python track_particles.py --checkpoint checkpoints/best_model.pth")
    print("\nFor detailed instructions, see README.md")
    print("=" * 80)


if __name__ == '__main__':
    try:
        quick_demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        import traceback
        traceback.print_exc()
