"""Training module with divergence regularization and comprehensive logging."""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import NeuralFlowField
from .utils import save_checkpoint


logger = logging.getLogger(__name__)


class FlowFieldLoss(nn.Module):
    """Combined loss with velocity MSE and divergence regularization."""
    
    def __init__(self, divergence_weight: float = 0.01):
        """
        Initialize loss function.
        
        Args:
            divergence_weight: Weight for divergence regularization term
        """
        super().__init__()
        self.divergence_weight = divergence_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_velocity: torch.Tensor, target_velocity: torch.Tensor,
                divergence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            pred_velocity: Predicted velocities (batch_size, 2)
            target_velocity: Target velocities (batch_size, 2)
            divergence: Optional divergence values (batch_size,)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Velocity MSE loss
        velocity_loss = self.mse_loss(pred_velocity, target_velocity)
        
        # Divergence regularization (penalize non-zero divergence for incompressible flow)
        if divergence is not None and self.divergence_weight > 0:
            div_loss = torch.mean(divergence ** 2)
            total_loss = velocity_loss + self.divergence_weight * div_loss
        else:
            div_loss = torch.tensor(0.0, device=velocity_loss.device)
            total_loss = velocity_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'velocity': velocity_loss.item(),
            'divergence': div_loss.item()
        }
        
        return total_loss, loss_dict


def compute_divergence_loss(model: NeuralFlowField, x: torch.Tensor, y: torch.Tensor,
                            t: torch.Tensor, sample_ratio: float = 0.1) -> torch.Tensor:
    """
    Compute divergence regularization loss.
    
    Args:
        model: Neural flow field model
        x: X coordinates (batch_size,)
        y: Y coordinates (batch_size,)
        t: Time values (batch_size,)
        sample_ratio: Fraction of batch to use for divergence computation
        
    Returns:
        Mean squared divergence
    """
    # Sample subset for efficiency (divergence computation is expensive)
    batch_size = len(x)
    sample_size = max(1, int(batch_size * sample_ratio))
    
    if sample_size < batch_size:
        indices = torch.randperm(batch_size, device=x.device)[:sample_size]
        x = x[indices]
        y = y[indices]
        t = t[indices]
    
    # Enable gradients for divergence computation (needed even during validation)
    with torch.enable_grad():
        # Detach and re-attach to ensure fresh computation
        x_div = x.detach().requires_grad_(True)
        y_div = y.detach().requires_grad_(True)
        t_div = t.detach()
        
        # Compute divergence
        divergence = model.compute_divergence(x_div, y_div, t_div)
    
    return divergence


def train_epoch(model: NeuralFlowField, dataloader: DataLoader,
               criterion: FlowFieldLoss, optimizer: optim.Optimizer,
               device: torch.device, compute_divergence: bool = True,
               grad_clip: Optional[float] = 1.0) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural flow field model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        compute_divergence: Whether to compute divergence regularization
        grad_clip: Gradient clipping max norm (None to disable)
        
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_losses = {'total': 0.0, 'velocity': 0.0, 'divergence': 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move data to device
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        t = batch['t'].to(device)
        target_vx = batch['vx'].to(device)
        target_vy = batch['vy'].to(device)
        target_velocity = torch.stack([target_vx, target_vy], dim=-1)
        
        # Forward pass
        pred_velocity = model(x, y, t)
        
        # Compute divergence if enabled
        if compute_divergence:
            divergence = compute_divergence_loss(model, x, y, t, sample_ratio=0.1)
        else:
            divergence = None
        
        # Compute loss
        loss, loss_dict = criterion(pred_velocity, target_velocity, divergence)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss_dict['total']:.6f}"})
    
    # Average losses
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    return avg_losses


def validate(model: NeuralFlowField, dataloader: DataLoader,
            criterion: FlowFieldLoss, device: torch.device,
            compute_divergence: bool = False) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Neural flow field model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        compute_divergence: Whether to compute divergence regularization
        
    Returns:
        Dictionary of average losses and metrics
    """
    model.eval()
    
    total_losses = {'total': 0.0, 'velocity': 0.0, 'divergence': 0.0}
    all_pred = []
    all_target = []
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validating"):
        # Move data to device
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        t = batch['t'].to(device)
        target_vx = batch['vx'].to(device)
        target_vy = batch['vy'].to(device)
        target_velocity = torch.stack([target_vx, target_vy], dim=-1)
        
        # Forward pass (no gradients for velocity prediction)
        with torch.no_grad():
            pred_velocity = model(x, y, t)
        
        # Compute divergence if enabled (needs gradients)
        if compute_divergence:
            divergence = compute_divergence_loss(model, x, y, t, sample_ratio=0.1)
        else:
            divergence = None
        
        # Compute loss (no gradients)
        with torch.no_grad():
            loss, loss_dict = criterion(pred_velocity, target_velocity, divergence)
            
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Store predictions for metrics
        all_pred.append(pred_velocity.cpu().numpy())
        all_target.append(target_velocity.cpu().numpy())
    
    # Average losses
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    # Compute additional metrics
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    
    mae = np.mean(np.abs(all_pred - all_target))
    rmse = np.sqrt(np.mean((all_pred - all_target) ** 2))
    
    avg_losses['mae'] = mae
    avg_losses['rmse'] = rmse
    
    return avg_losses


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def train_model(model: NeuralFlowField, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict[str, Any], device: torch.device,
                checkpoint_dir: str, log_dir: str,
                resume_checkpoint: Optional[str] = None) -> NeuralFlowField:
    """
    Complete training loop with checkpointing and logging.
    
    Args:
        model: Neural flow field model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        resume_checkpoint: Optional path to checkpoint to resume from
        
    Returns:
        Trained model
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(log_dir)
    
    # Training configuration
    train_config = config.get('training', {})
    num_epochs = train_config.get('num_epochs', 200)
    learning_rate = train_config.get('learning_rate', 1e-4)
    weight_decay = train_config.get('weight_decay', 1e-4)
    divergence_weight = train_config.get('loss', {}).get('divergence_weight', 0.01)
    grad_clip = train_config.get('gradient_clip', 1.0)
    checkpoint_freq = train_config.get('checkpoint_frequency', 10)
    
    # Early stopping
    early_stopping_config = train_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get('patience', 20),
        min_delta=early_stopping_config.get('min_delta', 1e-4)
    )
    
    # Loss function
    criterion = FlowFieldLoss(divergence_weight=divergence_weight)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler_config = train_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', num_epochs)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 50),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device,
            compute_divergence=(divergence_weight > 0),
            grad_clip=grad_clip
        )
        
        # Validate
        val_losses = validate(
            model, val_loader, criterion, device,
            compute_divergence=(divergence_weight > 0)
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
            f"Train Loss: {train_losses['total']:.6f}, "
            f"Val Loss: {val_losses['total']:.6f}, "
            f"Val MAE: {val_losses['mae']:.6f}"
        )
        
        # TensorBoard logging
        writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train_velocity', train_losses['velocity'], epoch)
        writer.add_scalar('Loss/train_divergence', train_losses['divergence'], epoch)
        writer.add_scalar('Loss/val_total', val_losses['total'], epoch)
        writer.add_scalar('Loss/val_velocity', val_losses['velocity'], epoch)
        writer.add_scalar('Metrics/val_mae', val_losses['mae'], epoch)
        writer.add_scalar('Metrics/val_rmse', val_losses['rmse'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
        
        if (epoch + 1) % checkpoint_freq == 0 or is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_losses['total'],
                    'val_loss': val_losses['total'],
                    'best_val_loss': best_val_loss,
                    'config': config
                },
                checkpoint_path,
                is_best=is_best
            )
            
            if is_best:
                logger.info(f"New best model saved with val loss: {best_val_loss:.6f}")
        
        # Early stopping
        if early_stopping(val_losses['total']):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    logger.info("Training completed")
    
    return model
