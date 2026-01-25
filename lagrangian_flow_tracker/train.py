"""Main training script for neural flow field model."""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_config, set_seed, get_device, setup_logger,
    DataNormalizer, count_parameters
)
from src.model import create_model
from src.data_processing import (
    load_flow_sequence, sample_particles_from_flow,
    create_datasets, create_dataloaders
)
from src.training import train_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Neural Flow Field Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing flow files (overrides config)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['flow_dir'] = args.data_dir
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logger('train', log_file='logs/train.log')
    logger.info("=" * 80)
    logger.info("Starting Neural Flow Field Training")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Get device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = get_device(config.get('device'))
    logger.info(f"Using device: {device}")
    
    # Load or process data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectories_path = output_dir / 'trajectories.csv'
    
    if trajectories_path.exists():
        logger.info(f"Loading existing trajectories from {trajectories_path}")
        trajectories = pd.read_csv(trajectories_path)
    else:
        logger.info("Processing flow data to extract trajectories")
        
        # Load flow sequence
        flow_dir = config['data'].get('flow_dir', '../Dataset/backstep')
        max_files = config['data'].get('max_flow_files', 50)
        
        logger.info(f"Loading flow files from {flow_dir}")
        flow_sequence, file_paths = load_flow_sequence(flow_dir, pattern='*.flo', max_files=max_files)
        
        # Sample particles from flow
        num_particles = config['data'].get('num_particles', 1000)
        num_timesteps = config['data'].get('num_timesteps', 10)
        
        logger.info(f"Sampling {num_particles} particles for {num_timesteps} timesteps")
        trajectories = sample_particles_from_flow(
            flow_sequence, num_particles=num_particles, num_timesteps=num_timesteps
        )
        
        # Save trajectories
        trajectories.to_csv(trajectories_path, index=False)
        logger.info(f"Saved trajectories to {trajectories_path}")
    
    logger.info(f"Total trajectory points: {len(trajectories)}")
    logger.info(f"Unique particles: {trajectories['particle_id'].nunique()}")
    
    # Create normalizer
    spatial_bounds = config['data']['normalization']['spatial_bounds']
    temporal_bounds = config['data']['normalization']['temporal_bounds']
    
    # Update temporal bounds based on actual data
    temporal_bounds = (trajectories['frame'].min(), trajectories['frame'].max())
    
    # Update spatial bounds if using flow data
    if 'x' in trajectories.columns and 'y' in trajectories.columns:
        spatial_bounds = (
            trajectories['x'].min(), trajectories['x'].max(),
            trajectories['y'].min(), trajectories['y'].max()
        )
    
    logger.info(f"Spatial bounds: {spatial_bounds}")
    logger.info(f"Temporal bounds: {temporal_bounds}")
    
    normalizer = DataNormalizer(spatial_bounds, temporal_bounds)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        trajectories,
        train_ratio=0.7,
        val_ratio=0.15,
        normalizer=normalizer,
        random_seed=seed
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(config, device)
    num_params = count_parameters(model)
    logger.info(f"Model created with {num_params:,} trainable parameters")
    
    # Train model
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs/tensorboard'
    
    logger.info("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resume_checkpoint=args.resume
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to {checkpoint_dir}/best_model.pth")
    
    # Save normalizer stats
    import json
    stats_path = output_dir / 'normalizer_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(normalizer.get_stats(), f, indent=2)
    logger.info(f"Saved normalizer stats to {stats_path}")
    
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
