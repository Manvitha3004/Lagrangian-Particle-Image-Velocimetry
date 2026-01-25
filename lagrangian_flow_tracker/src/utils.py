"""Utility functions for configuration, device management, and metrics computation."""

import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import yaml


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("Configuration file is empty")
        
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device (GPU if available, else CPU).
    
    Args:
        device_str: Device string ('cuda', 'cpu', or None for auto-detection)
        
    Returns:
        PyTorch device object
    """
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.info("Using CPU")
    
    return device


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics between predictions and targets.
    
    Args:
        pred: Predicted values, shape (N, ...)
        target: Target values, shape (N, ...)
        
    Returns:
        Dictionary containing MSE, RMSE, MAE, and MAPE metrics
    """
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape}, target {target.shape}"
    
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    
    # Mean Absolute Percentage Error (avoid division by zero)
    epsilon = 1e-8
    mape = np.mean(np.abs((target - pred) / (np.abs(target) + epsilon))) * 100
    
    # R-squared score
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2)
    }


class DataNormalizer:
    """Normalizer for spatial and temporal data."""
    
    def __init__(self, spatial_bounds: Tuple[float, float, float, float],
                 temporal_bounds: Tuple[float, float]):
        """
        Initialize normalizer with data bounds.
        
        Args:
            spatial_bounds: (x_min, x_max, y_min, y_max)
            temporal_bounds: (t_min, t_max)
        """
        self.x_min, self.x_max, self.y_min, self.y_max = spatial_bounds
        self.t_min, self.t_max = temporal_bounds
        
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.t_range = self.t_max - self.t_min
        
        # Avoid division by zero
        self.x_range = max(self.x_range, 1e-8)
        self.y_range = max(self.y_range, 1e-8)
        self.t_range = max(self.t_range, 1e-8)
    
    def normalize_position(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize spatial coordinates to [0, 1]."""
        x_norm = (x - self.x_min) / self.x_range
        y_norm = (y - self.y_min) / self.y_range
        return x_norm, y_norm
    
    def denormalize_position(self, x_norm: np.ndarray, y_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize spatial coordinates from [0, 1]."""
        x = x_norm * self.x_range + self.x_min
        y = y_norm * self.y_range + self.y_min
        return x, y
    
    def normalize_time(self, t: np.ndarray) -> np.ndarray:
        """Normalize time to [0, 1]."""
        return (t - self.t_min) / self.t_range
    
    def denormalize_time(self, t_norm: np.ndarray) -> np.ndarray:
        """Denormalize time from [0, 1]."""
        return t_norm * self.t_range + self.t_min
    
    def normalize_velocity(self, vx: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize velocity components."""
        # Velocity normalization: v_norm = v * dt / dx
        vx_norm = vx / self.x_range
        vy_norm = vy / self.y_range
        return vx_norm, vy_norm
    
    def denormalize_velocity(self, vx_norm: np.ndarray, vy_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Denormalize velocity components."""
        vx = vx_norm * self.x_range
        vy = vy_norm * self.y_range
        return vx, vy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        # Convert numpy types to native Python types for JSON serialization
        return {
            'spatial_bounds': (float(self.x_min), float(self.x_max), float(self.y_min), float(self.y_max)),
            'temporal_bounds': (float(self.t_min), float(self.t_max)),
            'x_range': float(self.x_range),
            'y_range': float(self.y_range),
            't_range': float(self.t_range)
        }


def create_directory_structure(base_dir: str) -> None:
    """
    Create project directory structure.
    
    Args:
        base_dir: Base directory path
    """
    dirs = [
        'data/raw_videos',
        'data/processed',
        'data/results',
        'checkpoints',
        'logs',
        'results/velocity_fields',
        'results/trajectories',
        'results/animations',
        'results/metrics'
    ]
    
    for d in dirs:
        Path(base_dir) / d
        os.makedirs(Path(base_dir) / d, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], save_path: str, is_best: bool = False) -> None:
    """
    Save training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
