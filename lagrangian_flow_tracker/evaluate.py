"""Evaluation script for trained neural flow field model."""

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
    load_config, setup_logger, get_device, DataNormalizer,
    load_checkpoint, compute_metrics
)
from src.model import create_model
from src.data_processing import FlowDataset
from src.tracking import validate_tracking_accuracy, compute_trajectory_error
from src.visualization import (
    plot_velocity_field, plot_streamlines, plot_error_analysis,
    plot_divergence_field
)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Neural Flow Field Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/processed/trajectories.csv',
                       help='Path to test trajectory data')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='data/results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logger('evaluate', log_file='logs/evaluate.log')
    logger.info("=" * 80)
    logger.info("Neural Flow Field Model Evaluation")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = get_device(config.get('device'))
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'velocity_fields').mkdir(exist_ok=True)
    (output_dir / 'trajectories').mkdir(exist_ok=True)
    (output_dir / 'metrics').mkdir(exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config, device)
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    trajectories = pd.read_csv(args.test_data)
    
    # Load normalizer stats
    normalizer_stats_path = Path(args.test_data).parent / 'normalizer_stats.json'
    if normalizer_stats_path.exists():
        with open(normalizer_stats_path, 'r') as f:
            stats = json.load(f)
        normalizer = DataNormalizer(
            stats['spatial_bounds'],
            stats['temporal_bounds']
        )
        logger.info("Loaded normalizer statistics")
    else:
        logger.warning("Normalizer stats not found, using config values")
        spatial_bounds = config['data']['normalization']['spatial_bounds']
        temporal_bounds = config['data']['normalization']['temporal_bounds']
        normalizer = DataNormalizer(spatial_bounds, temporal_bounds)
    
    # Create test dataset
    test_dataset = FlowDataset(trajectories, normalizer)
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluate velocity prediction
    logger.info("Evaluating velocity predictions...")
    
    all_pred_velocities = []
    all_true_velocities = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            x = sample['x'].unsqueeze(0).to(device)
            y = sample['y'].unsqueeze(0).to(device)
            t = sample['t'].unsqueeze(0).to(device)
            
            pred_vel = model(x, y, t)
            
            all_pred_velocities.append(pred_vel.cpu().numpy())
            all_true_velocities.append(
                np.array([[sample['vx'].item(), sample['vy'].item()]])
            )
    
    all_pred_velocities = np.concatenate(all_pred_velocities, axis=0)
    all_true_velocities = np.concatenate(all_true_velocities, axis=0)
    
    # Compute metrics
    velocity_metrics = compute_metrics(all_pred_velocities, all_true_velocities)
    
    logger.info("Velocity Prediction Metrics:")
    for metric, value in velocity_metrics.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics' / 'velocity_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(velocity_metrics, f, indent=2)
    logger.info(f"Saved velocity metrics to {metrics_path}")
    
    # Trajectory-based evaluation
    logger.info("Evaluating trajectory tracking accuracy...")
    
    # Get unique particles and create trajectory arrays
    particle_ids = trajectories['particle_id'].unique()[:10]  # Evaluate first 10 particles
    
    for pid in particle_ids:
        particle_traj = trajectories[trajectories['particle_id'] == pid].sort_values('frame')
        
        if len(particle_traj) < 5:
            continue
        
        # Ground truth
        true_positions = particle_traj[['x', 'y']].values
        
        # Predict trajectory
        start_pos = true_positions[0:1]
        start_time = particle_traj['frame'].values[0]
        dt = 1.0
        
        from src.tracking import track_forward
        
        # Normalize start position
        start_pos_norm = normalizer.normalize_position(start_pos[:, 0], start_pos[:, 1])
        start_pos_norm = np.stack(start_pos_norm, axis=1)
        start_time_norm = normalizer.normalize_time(np.array([start_time]))[0]
        
        pred_traj = track_forward(
            model, start_pos_norm, start_time_norm,
            num_steps=len(true_positions)-1, dt=dt/normalizer.t_range,
            device=device, method='rk4', show_progress=False
        )
        
        # Denormalize predictions
        pred_traj_denorm = np.zeros_like(pred_traj[:, 0, :])
        for t in range(len(pred_traj)):
            x_denorm, y_denorm = normalizer.denormalize_position(
                pred_traj[t, 0, 0], pred_traj[t, 0, 1]
            )
            pred_traj_denorm[t] = [x_denorm, y_denorm]
    
    # Visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        
        # Get bounds (denormalized)
        bounds = normalizer.get_stats()['spatial_bounds']
        
        # Velocity field at different times
        times_to_plot = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in times_to_plot:
            save_path = output_dir / 'velocity_fields' / f'velocity_field_t{t:.2f}.png'
            plot_velocity_field(
                model, t, bounds, resolution=30, device=device,
                save_path=str(save_path), show=False
            )
            logger.info(f"Saved velocity field at t={t:.2f}")
        
        # Streamlines
        save_path = output_dir / 'velocity_fields' / 'streamlines_t0.50.png'
        plot_streamlines(
            model, 0.5, bounds, num_seeds=20, device=device,
            save_path=str(save_path), show=False
        )
        logger.info("Saved streamlines plot")
        
        # Divergence field
        save_path = output_dir / 'velocity_fields' / 'divergence_t0.50.png'
        plot_divergence_field(
            model, 0.5, bounds, resolution=30, device=device,
            save_path=str(save_path), show=False
        )
        logger.info("Saved divergence field plot")
        
        # Error analysis
        save_path = output_dir / 'trajectories' / 'velocity_error_analysis.png'
        plot_error_analysis(
            all_pred_velocities, all_true_velocities,
            save_path=str(save_path), show=False
        )
        logger.info("Saved error analysis plot")
    
    logger.info("=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
