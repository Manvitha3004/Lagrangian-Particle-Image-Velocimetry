"""Neural Flow Field model with sinusoidal positional encoding."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for coordinates."""
    
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        """
        Initialize positional encoder.
        
        Args:
            num_frequencies: Number of frequency bands (L in the paper)
            include_input: Whether to include original input in encoding
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Create frequency bands: [2^0, 2^1, ..., 2^(L-1)]
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded tensor of shape (batch_size, encoded_dim)
            where encoded_dim = input_dim * (1 + 2 * num_frequencies) if include_input
                              = input_dim * 2 * num_frequencies otherwise
        """
        # x shape: (batch_size, input_dim)
        batch_size, input_dim = x.shape
        
        # Expand dimensions for broadcasting: (batch_size, input_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute scaled inputs: (batch_size, input_dim, num_frequencies)
        scaled = x_expanded * self.freq_bands
        
        # Compute sin and cos: (batch_size, input_dim, num_frequencies)
        sin_enc = torch.sin(2 * np.pi * scaled)
        cos_enc = torch.cos(2 * np.pi * scaled)
        
        # Interleave sin and cos: (batch_size, input_dim, 2 * num_frequencies)
        encoding = torch.stack([sin_enc, cos_enc], dim=-1)
        encoding = encoding.reshape(batch_size, input_dim, -1)
        
        # Flatten to (batch_size, input_dim * 2 * num_frequencies)
        encoding = encoding.reshape(batch_size, -1)
        
        # Optionally concatenate original input
        if self.include_input:
            encoding = torch.cat([x, encoding], dim=-1)
        
        return encoding
    
    def get_output_dim(self, input_dim: int) -> int:
        """Get output dimension for given input dimension."""
        encoded_dim = input_dim * 2 * self.num_frequencies
        if self.include_input:
            encoded_dim += input_dim
        return encoded_dim


class FlowFieldMLP(nn.Module):
    """Multi-layer perceptron for flow field prediction."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 256, 256, 128],
                 output_dim: int = 2,
                 activation: str = 'relu',
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (2 for vx, vy)
            activation: Activation function ('relu', 'tanh', 'gelu')
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class NeuralFlowField(nn.Module):
    """Complete neural flow field model combining positional encoding and MLP."""
    
    def __init__(self,
                 num_frequencies: int = 10,
                 hidden_dims: List[int] = [256, 256, 256, 128],
                 activation: str = 'relu',
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        """
        Initialize neural flow field.
        
        Args:
            num_frequencies: Number of frequency bands for positional encoding
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        # Positional encoder for (x, y, t)
        self.pos_encoder = PositionalEncoder(num_frequencies=num_frequencies, include_input=True)
        
        # Input dimension: 3 (x, y, t) -> encoded dimension
        input_dim = 3  # x, y, t
        encoded_dim = self.pos_encoder.get_output_dim(input_dim)
        
        # MLP for velocity prediction
        self.mlp = FlowFieldMLP(
            input_dim=encoded_dim,
            hidden_dims=hidden_dims,
            output_dim=2,  # vx, vy
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field at given positions and time.
        
        Args:
            x: X coordinates, shape (batch_size,) or (batch_size, 1)
            y: Y coordinates, shape (batch_size,) or (batch_size, 1)
            t: Time values, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Velocity tensor of shape (batch_size, 2) where [:, 0] is vx and [:, 1] is vy
        """
        # Ensure inputs are 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Concatenate coordinates: (batch_size, 3)
        coords = torch.cat([x, y, t], dim=-1)
        
        # Encode positions
        encoded = self.pos_encoder(coords)
        
        # Predict velocity
        velocity = self.mlp(encoded)
        
        return velocity
    
    def predict_velocity(self, positions: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Predict velocities for batch of positions and times.
        
        Args:
            positions: Tensor of shape (batch_size, 2) containing (x, y) coordinates
            times: Tensor of shape (batch_size,) or (batch_size, 1) containing time values
            
        Returns:
            Velocities of shape (batch_size, 2)
        """
        x = positions[:, 0]
        y = positions[:, 1]
        return self.forward(x, y, times)
    
    def compute_divergence(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of velocity field using automatic differentiation.
        
        Divergence = ∂vx/∂x + ∂vy/∂y
        
        Args:
            x: X coordinates, shape (batch_size,) or (batch_size, 1)
            y: Y coordinates, shape (batch_size,) or (batch_size, 1)
            t: Time values, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Divergence values, shape (batch_size,)
        """
        # Ensure inputs require gradients
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach() if isinstance(t, torch.Tensor) else torch.tensor(t)
        
        # Predict velocity
        velocity = self.forward(x, y, t)
        vx = velocity[:, 0]
        vy = velocity[:, 1]
        
        # Compute gradients
        # ∂vx/∂x
        dvx_dx = torch.autograd.grad(
            outputs=vx,
            inputs=x,
            grad_outputs=torch.ones_like(vx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ∂vy/∂y
        dvy_dy = torch.autograd.grad(
            outputs=vy,
            inputs=y,
            grad_outputs=torch.ones_like(vy),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Sum to get divergence
        divergence = dvx_dx.squeeze() + dvy_dy.squeeze()
        
        return divergence
    
    def compute_vorticity(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute vorticity (curl) of velocity field.
        
        Vorticity = ∂vy/∂x - ∂vx/∂y
        
        Args:
            x: X coordinates, shape (batch_size,) or (batch_size, 1)
            y: Y coordinates, shape (batch_size,) or (batch_size, 1)
            t: Time values, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Vorticity values, shape (batch_size,)
        """
        # Ensure inputs require gradients
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach() if isinstance(t, torch.Tensor) else torch.tensor(t)
        
        # Predict velocity
        velocity = self.forward(x, y, t)
        vx = velocity[:, 0]
        vy = velocity[:, 1]
        
        # Compute gradients
        # ∂vy/∂x
        dvy_dx = torch.autograd.grad(
            outputs=vy,
            inputs=x,
            grad_outputs=torch.ones_like(vy),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # ∂vx/∂y
        dvx_dy = torch.autograd.grad(
            outputs=vx,
            inputs=y,
            grad_outputs=torch.ones_like(vx),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute vorticity
        vorticity = dvy_dx.squeeze() - dvx_dy.squeeze()
        
        return vorticity


def create_model(config: dict, device: torch.device) -> NeuralFlowField:
    """
    Create neural flow field model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    num_frequencies = model_config.get('positional_encoding', {}).get('num_frequencies', 10)
    mlp_config = model_config.get('mlp', {})
    hidden_dims = mlp_config.get('hidden_dims', [256, 256, 256, 128])
    activation = mlp_config.get('activation', 'relu')
    use_layer_norm = mlp_config.get('use_layer_norm', True)
    dropout = mlp_config.get('dropout', 0.0)
    
    model = NeuralFlowField(
        num_frequencies=num_frequencies,
        hidden_dims=hidden_dims,
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout=dropout
    )
    
    return model.to(device)
