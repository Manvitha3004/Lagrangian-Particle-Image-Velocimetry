"""Script for tracking particles using trained neural flow field model."""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_config, setup_logger, get_device, DataNormalizer, load_checkpoint
)
from src.model import create_model
from src.tracking import track_forward, track_backward, find_source
from src.visualization import plot_trajectories, animate_tracking


def main():
    """Main particle tracking function."""
    parser = argparse.ArgumentParser(description='Track Particles using Neural Flow Field')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--start_positions', type=str, default=None,
                       help='CSV file with starting positions (columns: x, y)')
    parser.add_argument('--num_particles', type=int, default=100,
                       help='Number of random particles to track (if no start_positions provided)')
    parser.add_argument('--mode', type=str, default='forward', choices=['forward', 'backward', 'source'],
                       help='Tracking mode: forward, backward, or source finding')
    parser.add_argument('--start_time', type=float, default=0.0,
                       help='Starting time for tracking')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of integration steps')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step size (in normalized time)')
    parser.add_argument('--method', type=str, default='rk4', choices=['euler', 'rk4'],
                       help='Integration method')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='data/results/tracking',
                       help='Output directory for tracking results')
    parser.add_argument('--animate', action='store_true',
                       help='Generate animation of particle tracking')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logger('track_particles', log_file='logs/track_particles.log')
    logger.info("=" * 80)
    logger.info("Particle Tracking with Neural Flow Field")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Integration method: {args.method}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = get_device(config.get('device'))
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config, device)
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Load normalizer stats
    checkpoint_dir = Path(args.checkpoint).parent.parent
    normalizer_stats_path = checkpoint_dir / 'data' / 'processed' / 'normalizer_stats.json'
    
    if not normalizer_stats_path.exists():
        # Try alternative location
        normalizer_stats_path = Path('data/processed/normalizer_stats.json')
    
    if normalizer_stats_path.exists():
        with open(normalizer_stats_path, 'r') as f:
            stats = json.load(f)
        normalizer = DataNormalizer(
            stats['spatial_bounds'],
            stats['temporal_bounds']
        )
        logger.info("Loaded normalizer statistics")
        bounds = stats['spatial_bounds']
    else:
        logger.warning("Normalizer stats not found, using config values")
        spatial_bounds = config['data']['normalization']['spatial_bounds']
        temporal_bounds = config['data']['normalization']['temporal_bounds']
        normalizer = DataNormalizer(spatial_bounds, temporal_bounds)
        bounds = spatial_bounds
    
    logger.info(f"Spatial bounds: {bounds}")
    
    # Get starting positions
    if args.start_positions and os.path.exists(args.start_positions):
        logger.info(f"Loading starting positions from {args.start_positions}")
        positions_df = pd.read_csv(args.start_positions)
        start_positions = positions_df[['x', 'y']].values
    else:
        logger.info(f"Generating {args.num_particles} random starting positions")
        x_min, x_max, y_min, y_max = bounds
        start_positions = np.random.uniform(
            low=[x_min, y_min],
            high=[x_max, y_max],
            size=(args.num_particles, 2)
        )
    
    logger.info(f"Starting positions: {len(start_positions)} particles")
    
    # Normalize positions and time
    start_positions_norm = normalizer.normalize_position(
        start_positions[:, 0], start_positions[:, 1]
    )
    start_positions_norm = np.stack(start_positions_norm, axis=1)
    start_time_norm = normalizer.normalize_time(np.array([args.start_time]))[0]
    
    # Track particles
    if args.mode == 'forward':
        logger.info(f"Tracking {len(start_positions)} particles forward for {args.steps} steps")
        trajectories = track_forward(
            model, start_positions_norm, start_time_norm,
            num_steps=args.steps, dt=args.dt,
            device=device, method=args.method, show_progress=True
        )
    
    elif args.mode == 'backward':
        logger.info(f"Tracking {len(start_positions)} particles backward for {args.steps} steps")
        trajectories = track_backward(
            model, start_positions_norm, start_time_norm,
            num_steps=args.steps, dt=args.dt,
            device=device, method=args.method, show_progress=True
        )
    
    elif args.mode == 'source':
        lookback_time = args.steps * args.dt
        logger.info(f"Finding source positions (lookback time: {lookback_time:.2f})")
        source_positions = find_source(
            model, start_positions_norm, start_time_norm,
            lookback_time=lookback_time, dt=args.dt,
            device=device, method=args.method
        )
        
        # Denormalize
        source_x, source_y = normalizer.denormalize_position(
            source_positions[:, 0], source_positions[:, 1]
        )
        
        # Save results
        results_df = pd.DataFrame({
            'start_x': start_positions[:, 0],
            'start_y': start_positions[:, 1],
            'source_x': source_x,
            'source_y': source_y
        })
        
        save_path = output_dir / 'source_positions.csv'
        results_df.to_csv(save_path, index=False)
        logger.info(f"Saved source positions to {save_path}")
        
        # Plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(start_positions[:, 0], start_positions[:, 1], 
                  c='red', label='Current', s=50, alpha=0.7)
        ax.scatter(source_x, source_y, 
                  c='blue', label='Source', s=50, alpha=0.7)
        
        # Draw arrows
        for i in range(len(start_positions)):
            ax.arrow(source_x[i], source_y[i],
                    start_positions[i, 0] - source_x[i],
                    start_positions[i, 1] - source_y[i],
                    alpha=0.3, head_width=2, length_includes_head=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Source Finding Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / 'source_finding.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved source finding plot to {plot_path}")
        plt.close()
        
        logger.info("Source finding completed!")
        return
    
    # Denormalize trajectories
    trajectories_denorm = np.zeros_like(trajectories)
    for t in range(len(trajectories)):
        x_denorm, y_denorm = normalizer.denormalize_position(
            trajectories[t, :, 0], trajectories[t, :, 1]
        )
        trajectories_denorm[t, :, 0] = x_denorm
        trajectories_denorm[t, :, 1] = y_denorm
    
    logger.info(f"Trajectory shape: {trajectories_denorm.shape}")
    
    # Save trajectories
    save_path = output_dir / f'trajectories_{args.mode}.npy'
    np.save(save_path, trajectories_denorm)
    logger.info(f"Saved trajectories to {save_path}")
    
    # Save as CSV
    rows = []
    for particle_id in range(trajectories_denorm.shape[1]):
        for time_step in range(trajectories_denorm.shape[0]):
            rows.append({
                'particle_id': particle_id,
                'time_step': time_step,
                'time': args.start_time + time_step * args.dt * normalizer.t_range,
                'x': trajectories_denorm[time_step, particle_id, 0],
                'y': trajectories_denorm[time_step, particle_id, 1]
            })
    
    trajectories_df = pd.DataFrame(rows)
    csv_path = output_dir / f'trajectories_{args.mode}.csv'
    trajectories_df.to_csv(csv_path, index=False)
    logger.info(f"Saved trajectories to {csv_path}")
    
    # Visualize
    logger.info("Generating trajectory plot...")
    plot_path = output_dir / f'trajectories_{args.mode}.png'
    
    # Transpose for plotting: (num_particles, num_timesteps, 2)
    trajectories_plot = np.transpose(trajectories_denorm, (1, 0, 2))
    
    plot_trajectories(
        trajectories_plot,
        frame=None,
        save_path=str(plot_path),
        title=f'Particle Trajectories ({args.mode.capitalize()})',
        show=False
    )
    logger.info(f"Saved trajectory plot to {plot_path}")
    
    # Generate animation if requested
    if args.animate:
        logger.info("Generating animation...")
        
        # Denormalize time range
        end_time_norm = start_time_norm + args.steps * args.dt
        end_time_denorm = normalizer.denormalize_time(np.array([end_time_norm]))[0]
        start_time_denorm = normalizer.denormalize_time(np.array([start_time_norm]))[0]
        
        animation_path = output_dir / f'tracking_animation_{args.mode}.gif'
        
        # Use subset of particles for animation (for performance)
        num_particles_animate = min(50, len(start_positions_norm))
        
        animate_tracking(
            model,
            start_positions_norm[:num_particles_animate],
            time_range=(start_time_norm, end_time_norm),
            dt=args.dt,
            bounds=normalizer.get_stats()['spatial_bounds'],
            device=device,
            save_path=str(animation_path),
            fps=30
        )
        logger.info(f"Saved animation to {animation_path}")
    
    logger.info("=" * 80)
    logger.info("Particle tracking completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
