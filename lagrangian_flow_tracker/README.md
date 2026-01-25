# Lagrangian Particle Tracking with Neural Flow Fields

A production-ready implementation of neural network-based Lagrangian particle tracking for fluid flow analysis. This system learns 2D+time fluid flow fields from particle trajectory data or optical flow files and performs accurate forward/backward particle tracking using neural implicit representations.

## Features

- **Neural Flow Field Model**: Time-conditional MLP with sinusoidal positional encoding
- **Multiple Data Sources**: Support for videos, optical flow (.flo) files, and trajectory data
- **Physics-Based Training**: Divergence regularization for incompressible flow
- **Advanced Integration**: Euler and RK4 methods for particle tracking
- **Comprehensive Visualization**: Velocity fields, streamlines, trajectories, and animations
- **Production Ready**: Error handling, logging, checkpointing, and GPU acceleration

## Project Structure

```
lagrangian_flow_tracker/
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Video/flow processing, particle tracking
│   ├── model.py               # Neural network architecture
│   ├── training.py            # Training loop with physics constraints
│   ├── tracking.py            # Particle integration (Euler, RK4)
│   ├── visualization.py       # Plotting and animation tools
│   └── utils.py               # Helper functions
├── configs/
│   └── config.yaml            # Configuration file
├── data/
│   ├── raw_videos/            # Input videos
│   ├── processed/             # Processed trajectories
│   └── results/               # Output results
├── checkpoints/               # Model checkpoints
├── logs/                      # Training logs
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── track_particles.py         # Particle tracking script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Setup

1. **Clone or navigate to the project directory**:
```bash
cd lagrangian_flow_tracker
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Prepare Your Data

The system supports three data input modes:

#### Option A: Optical Flow Files (.flo)
If you have pre-computed optical flow files (Middlebury format):

```bash
# Your flow files should be in a directory like:
# ../Dataset/backstep/backstep_Re1000_00001_flow.flo
# ../Dataset/backstep/backstep_Re1000_00002_flow.flo
# ...
```

Update `configs/config.yaml`:
```yaml
data:
  flow_dir: "../Dataset/backstep"
  max_flow_files: 50
  num_particles: 2000
  num_timesteps: 20
```

#### Option B: Video Files
If you have particle tracking videos:

```bash
# Place videos in:
# data/raw_videos/video1.mp4
# data/raw_videos/video2.avi
```

Update `configs/config.yaml`:
```yaml
data:
  video_dir: "data/raw_videos"
  particle_detection:
    min_area: 5
    max_area: 100
    threshold: 50
```

#### Option C: Existing Trajectory Data
If you already have trajectory data as CSV with columns: `particle_id`, `frame`, `x`, `y`, `vx`, `vy`:

```bash
# Place in:
# data/processed/trajectories.csv
```

### 2. Configure Training

Edit `configs/config.yaml` to adjust:
- Model architecture (hidden layers, positional encoding frequencies)
- Training hyperparameters (learning rate, batch size, epochs)
- Physics constraints (divergence regularization weight)
- Device settings (GPU/CPU)

### 3. Train the Model

```bash
python train.py --config configs/config.yaml
```

**Options**:
```bash
# Resume from checkpoint
python train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_50.pth

# Specify data directory
python train.py --config configs/config.yaml --data_dir ../Dataset/cylinder

# Force CPU training
python train.py --config configs/config.yaml --no_cuda
```

**Expected output**:
- `checkpoints/best_model.pth` - Best performing model
- `checkpoints/checkpoint_epoch_*.pth` - Periodic checkpoints
- `data/processed/trajectories.csv` - Extracted trajectories
- `data/processed/normalizer_stats.json` - Normalization parameters
- `logs/train.log` - Training log
- `logs/tensorboard/` - TensorBoard logs

**Monitor training**:
```bash
tensorboard --logdir logs/tensorboard
```

### 4. Evaluate the Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

**Options**:
```bash
# Specify test data
python evaluate.py --checkpoint checkpoints/best_model.pth --test_data data/processed/test_trajectories.csv

# Custom output directory
python evaluate.py --checkpoint checkpoints/best_model.pth --output_dir results/experiment1 --visualize
```

**Expected output**:
- `data/results/metrics/velocity_metrics.json` - Quantitative metrics
- `data/results/velocity_fields/` - Velocity field visualizations
- `data/results/trajectories/` - Trajectory comparisons
- Error analysis plots

### 5. Track Particles

#### Forward Tracking
```bash
python track_particles.py --checkpoint checkpoints/best_model.pth --mode forward --steps 100 --animate
```

#### Backward Tracking
```bash
python track_particles.py --checkpoint checkpoints/best_model.pth --mode backward --steps 100
```

#### Source Finding
```bash
python track_particles.py --checkpoint checkpoints/best_model.pth --mode source --steps 50
```

**Options**:
```bash
# Custom starting positions from CSV
python track_particles.py --checkpoint checkpoints/best_model.pth --start_positions initial_positions.csv --mode forward

# More particles
python track_particles.py --checkpoint checkpoints/best_model.pth --num_particles 500 --mode forward

# Different integration method
python track_particles.py --checkpoint checkpoints/best_model.pth --method euler --dt 0.005

# Generate animation
python track_particles.py --checkpoint checkpoints/best_model.pth --mode forward --animate
```

**Expected output**:
- `data/results/tracking/trajectories_forward.csv` - Trajectory data
- `data/results/tracking/trajectories_forward.npy` - NumPy array
- `data/results/tracking/trajectories_forward.png` - Visualization
- `data/results/tracking/tracking_animation_forward.gif` - Animation (if --animate)

## Configuration Guide

### Model Architecture

```yaml
model:
  positional_encoding:
    num_frequencies: 10  # Higher = more detail, but slower
  mlp:
    hidden_dims: [256, 256, 256, 128]  # Larger = more capacity
    activation: "relu"  # relu, tanh, or gelu
    dropout: 0.0  # Regularization (0.1-0.3 if overfitting)
```

### Training Parameters

```yaml
training:
  batch_size: 4096  # Adjust based on GPU memory
  num_epochs: 200
  learning_rate: 0.0001  # Decrease if loss oscillates
  
  loss:
    divergence_weight: 0.01  # Physics constraint (0.001-0.1)
  
  early_stopping:
    patience: 20  # Stop if no improvement
```

### Tracking Settings

```yaml
tracking:
  integration_method: "rk4"  # rk4 more accurate, euler faster
  dt: 0.01  # Smaller = more accurate but slower
```

## Data Format Specifications

### Input Video Format
- Supported: MP4, AVI, MOV
- Requirements: Bright particles on dark background
- Recommended: 640x480 or higher, 30+ fps

### Optical Flow Format (.flo)
Middlebury format:
- Header: Magic number (202021.25), width, height (32-bit)
- Data: Float32 array of shape (height, width, 2)
- [:,:,0] = horizontal flow (u), [:,:,1] = vertical flow (v)

### Trajectory CSV Format
Required columns:
- `particle_id` (int): Unique particle identifier
- `frame` (int): Frame/time index
- `x` (float): X position
- `y` (float): Y position
- `vx` (float): X velocity
- `vy` (float): Y velocity

## Performance Tips

1. **GPU Acceleration**: Use CUDA for 10-50x speedup
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Size**: Increase until GPU memory is ~80% full
   - 8GB GPU: batch_size = 2048-4096
   - 16GB GPU: batch_size = 8192-16384

3. **Data Loading**: Use appropriate `num_workers` in DataLoader
   - Windows: 0 (due to multiprocessing limitations)
   - Linux/Mac: 2-4

4. **Memory Management**: For large datasets, process in chunks
   ```yaml
   data:
     max_flow_files: 50  # Limit number of files
   ```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 2048  # Was 4096
```

**2. Training loss not decreasing**
- Check learning rate (try 1e-5 to 1e-3)
- Verify data normalization
- Reduce divergence_weight if too high

**3. Poor tracking accuracy**
- Increase model capacity (more hidden layers/neurons)
- Train longer (more epochs)
- Reduce time step dt
- Use RK4 instead of Euler

**4. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**5. Video processing fails**
```bash
# Install additional codecs
pip install opencv-python-headless
```

## Advanced Usage

### Custom Loss Functions

Edit `src/training.py` to add custom physics constraints:

```python
def custom_loss(model, x, y, t):
    # Example: Vorticity conservation
    vorticity = model.compute_vorticity(x, y, t)
    return torch.mean(vorticity ** 2)
```

### Batch Processing Multiple Datasets

```bash
# Process multiple flow directories
for dir in ../Dataset/*/; do
    python train.py --config configs/config.yaml --data_dir "$dir" --output_dir "results/$(basename $dir)"
done
```

### Export Results

```python
# Load and export predictions
import torch
import numpy as np
from src.model import create_model
from src.utils import load_config, load_checkpoint

config = load_config('configs/config.yaml')
model = create_model(config, torch.device('cpu'))
load_checkpoint('checkpoints/best_model.pth', model)

# Query velocity at any point
x, y, t = 100.0, 200.0, 5.0
velocity = model(torch.tensor([x]), torch.tensor([y]), torch.tensor([t]))
print(f"Velocity at ({x}, {y}, {t}): {velocity.numpy()}")
```

## Validation Checklist

Test your installation:

- [ ] Load sample flow data: `python -c "from src.data_processing import read_flo_file; print('OK')"`
- [ ] Create model: `python -c "from src.model import NeuralFlowField; print('OK')"`
- [ ] Train for 1 epoch: `python train.py --config configs/config.yaml` (Ctrl+C after 1 epoch)
- [ ] Predict velocity: Test with evaluate.py
- [ ] Track particles: `python track_particles.py --checkpoint checkpoints/best_model.pth --steps 10`

## Citation

If you use this code in your research, please cite:

```
@software{lagrangian_flow_tracker,
  title={Lagrangian Particle Tracking with Neural Flow Fields},
  author={ICIP-2026 Research Team},
  year={2026},
  url={https://github.com/yourusername/lagrangian-flow-tracker}
}
```

## License

MIT License - See LICENSE file for details

## Support

For questions and issues:
1. Check the troubleshooting section above
2. Review configuration in `configs/config.yaml`
3. Check logs in `logs/train.log`
4. Verify data format specifications

## Acknowledgments

- Optical flow format: [Middlebury Vision](http://vision.middlebury.edu/flow/)
- Neural implicit representations: NeRF, SIREN architectures
- Physics-informed neural networks (PINNs) literature

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Production Ready
