# Quick Start Guide

## Get Running in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
cd lagrangian_flow_tracker
pip install -r requirements.txt
```

### Step 2: Test Installation (1 minute)

```bash
python test_installation.py
```

Expected output: All tests should pass ✓

### Step 3: Run Demo (2 minutes)

```bash
python demo.py
```

This creates a synthetic vortex flow and demonstrates the full pipeline.

---

## Working with Your Dataset

Your dataset is already available at `../Dataset/backstep/` with .flo files.

### Quick Training (10-30 minutes on GPU)

```bash
# Train on backstep flow data
python train.py --config configs/config.yaml

# Monitor progress
tensorboard --logdir logs/tensorboard
```

### Quick Evaluation (1 minute)

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

### Quick Tracking (30 seconds)

```bash
# Forward tracking
python track_particles.py --checkpoint checkpoints/best_model.pth --mode forward --steps 50

# Backward tracking (source finding)
python track_particles.py --checkpoint checkpoints/best_model.pth --mode source --steps 30

# With animation
python track_particles.py --checkpoint checkpoints/best_model.pth --mode forward --steps 100 --animate
```

---

## Common Commands

### Training Commands

```bash
# Basic training
python train.py --config configs/config.yaml

# Resume from checkpoint
python train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_50.pth

# Different dataset
python train.py --config configs/config.yaml --data_dir ../Dataset/cylinder

# CPU only
python train.py --config configs/config.yaml --no_cuda
```

### Evaluation Commands

```bash
# Full evaluation with plots
python evaluate.py --checkpoint checkpoints/best_model.pth --visualize

# Custom test data
python evaluate.py --checkpoint checkpoints/best_model.pth --test_data data/processed/test.csv

# Save to specific directory
python evaluate.py --checkpoint checkpoints/best_model.pth --output_dir results/exp1 --visualize
```

### Tracking Commands

```bash
# Forward tracking 100 random particles
python track_particles.py --checkpoint checkpoints/best_model.pth --num_particles 100 --mode forward --steps 100

# From specific positions
python track_particles.py --checkpoint checkpoints/best_model.pth --start_positions positions.csv --mode forward

# Backward tracking
python track_particles.py --checkpoint checkpoints/best_model.pth --mode backward --steps 50

# Find sources
python track_particles.py --checkpoint checkpoints/best_model.pth --mode source --steps 40

# Create animation
python track_particles.py --checkpoint checkpoints/best_model.pth --mode forward --animate --steps 100

# Use Euler method instead of RK4
python track_particles.py --checkpoint checkpoints/best_model.pth --method euler --dt 0.001
```

---

## Configuration Quick Reference

Edit `configs/config.yaml` for:

### Model Settings
```yaml
model:
  positional_encoding:
    num_frequencies: 10  # 6-15 typical
  mlp:
    hidden_dims: [256, 256, 256, 128]  # Larger = more capacity
```

### Training Settings
```yaml
training:
  batch_size: 4096  # Reduce if GPU memory issues
  num_epochs: 200
  learning_rate: 0.0001  # 1e-5 to 1e-3 range
  loss:
    divergence_weight: 0.01  # 0.001-0.1 for incompressible flows
```

### Data Settings
```yaml
data:
  flow_dir: "../Dataset/backstep"
  max_flow_files: 50  # null for all files
  num_particles: 2000  # More = better coverage
  num_timesteps: 20  # Longer trajectories
```

---

## Expected Outputs

### After Training
- `checkpoints/best_model.pth` - Best model
- `checkpoints/checkpoint_epoch_*.pth` - Periodic saves
- `data/processed/trajectories.csv` - Training data
- `data/processed/normalizer_stats.json` - Normalization params
- `logs/train.log` - Training log
- `logs/tensorboard/` - TensorBoard logs

### After Evaluation
- `data/results/metrics/velocity_metrics.json` - Performance metrics
- `data/results/velocity_fields/*.png` - Flow visualizations
- `data/results/trajectories/*.png` - Error analysis

### After Tracking
- `data/results/tracking/trajectories_*.csv` - Trajectory data
- `data/results/tracking/trajectories_*.npy` - NumPy array
- `data/results/tracking/trajectories_*.png` - Plot
- `data/results/tracking/tracking_animation_*.gif` - Animation

---

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 2048  # was 4096
```

### Training Loss Not Decreasing
- Check learning rate (try 1e-4 to 1e-5)
- Reduce divergence_weight
- Train longer

### Poor Tracking Accuracy
- Increase model size
- Train more epochs
- Use RK4 instead of Euler
- Reduce dt

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

---

## Tips

1. **Start Small**: Train on 10-20 flow files first, then scale up
2. **Monitor**: Use TensorBoard to watch training progress
3. **GPU**: Training on GPU is 10-50x faster than CPU
4. **Save Often**: Checkpoints are saved every 10 epochs by default
5. **Visualize**: Always use `--visualize` flag when evaluating

---

## Next Steps

1. ✅ Run `test_installation.py`
2. ✅ Run `demo.py` to see it work
3. ✅ Train on your data with `train.py`
4. ✅ Evaluate with `evaluate.py --visualize`
5. ✅ Track particles with `track_particles.py`

For detailed documentation, see [README.md](README.md)
