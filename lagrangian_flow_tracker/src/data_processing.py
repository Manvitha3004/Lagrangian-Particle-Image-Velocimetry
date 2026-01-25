"""Data processing for particle tracking from videos and flow files."""

import os
import struct
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


def read_flo_file(flo_path: str) -> np.ndarray:
    """
    Read .flo optical flow file (Middlebury format).
    
    Args:
        flo_path: Path to .flo file
        
    Returns:
        Flow field array of shape (height, width, 2) where [:,:,0] is u and [:,:,1] is v
        
    Raises:
        ValueError: If file format is invalid
    """
    with open(flo_path, 'rb') as f:
        # Read magic number
        magic = struct.unpack('f', f.read(4))[0]
        if magic != 202021.25:  # Magic number for .flo files
            raise ValueError(f"Invalid .flo file magic number: {magic}")
        
        # Read dimensions
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        
        # Read flow data
        data = np.fromfile(f, np.float32)
        
    # Reshape to (height, width, 2)
    flow = data.reshape((height, width, 2))
    
    return flow


def write_flo_file(flow: np.ndarray, flo_path: str) -> None:
    """
    Write flow field to .flo file.
    
    Args:
        flow: Flow field array of shape (height, width, 2)
        flo_path: Output path for .flo file
    """
    height, width = flow.shape[:2]
    
    with open(flo_path, 'wb') as f:
        # Write magic number
        f.write(struct.pack('f', 202021.25))
        
        # Write dimensions
        f.write(struct.pack('i', width))
        f.write(struct.pack('i', height))
        
        # Write flow data
        flow.astype(np.float32).tofile(f)


def extract_frames(video_path: str, output_dir: Optional[str] = None, 
                   frame_skip: int = 1) -> np.ndarray:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Optional directory to save frames as images
        frame_skip: Extract every Nth frame (1 = all frames)
        
    Returns:
        Array of frames with shape (num_frames, height, width, 3)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video can't be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Extracting frames from {video_path} (total: {total_frames})")
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frames.append(frame)
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    frame_path = os.path.join(output_dir, f"frame_{len(frames):06d}.png")
                    cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    frames_array = np.array(frames)
    logger.info(f"Extracted {len(frames)} frames (shape: {frames_array.shape})")
    
    return frames_array


def detect_particles(frame: np.ndarray, min_area: int = 5, max_area: int = 100,
                    threshold: int = 50) -> List[Tuple[float, float]]:
    """
    Detect particles in a frame using blob detection.
    
    Args:
        frame: Input frame (grayscale or BGR)
        min_area: Minimum particle area in pixels
        max_area: Maximum particle area in pixels
        threshold: Intensity threshold for particle detection
        
    Returns:
        List of (x, y) particle positions
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Threshold image to get bright particles
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    particles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Compute centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                particles.append((cx, cy))
    
    return particles


def track_particles(frames: np.ndarray, max_distance: float = 50.0,
                   min_track_length: int = 5,
                   detection_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Track particles across frames using nearest-neighbor matching.
    
    Args:
        frames: Array of frames with shape (num_frames, height, width, channels)
        max_distance: Maximum distance for particle matching between frames
        min_track_length: Minimum number of frames a track must span
        detection_params: Parameters for particle detection
        
    Returns:
        DataFrame with columns: particle_id, frame, x, y
    """
    if detection_params is None:
        detection_params = {'min_area': 5, 'max_area': 100, 'threshold': 50}
    
    num_frames = len(frames)
    logger.info(f"Tracking particles across {num_frames} frames")
    
    # Detect particles in first frame
    particles_prev = detect_particles(frames[0], **detection_params)
    
    # Initialize tracks
    tracks = {i: [(0, x, y)] for i, (x, y) in enumerate(particles_prev)}
    next_track_id = len(particles_prev)
    
    # Track through subsequent frames
    for frame_idx in tqdm(range(1, num_frames), desc="Tracking particles"):
        particles_curr = detect_particles(frames[frame_idx], **detection_params)
        
        if len(particles_prev) == 0 or len(particles_curr) == 0:
            # No matches possible
            for x, y in particles_curr:
                tracks[next_track_id] = [(frame_idx, x, y)]
                next_track_id += 1
            particles_prev = particles_curr
            continue
        
        # Compute distance matrix
        prev_array = np.array(particles_prev)
        curr_array = np.array(particles_curr)
        
        distances = np.sqrt(
            (prev_array[:, np.newaxis, 0] - curr_array[np.newaxis, :, 0]) ** 2 +
            (prev_array[:, np.newaxis, 1] - curr_array[np.newaxis, :, 1]) ** 2
        )
        
        # Match particles (greedy nearest-neighbor)
        matched_prev = set()
        matched_curr = set()
        
        # Find matches in order of increasing distance
        dist_flat = distances.flatten()
        indices = np.argsort(dist_flat)
        
        for idx in indices:
            prev_idx = idx // len(particles_curr)
            curr_idx = idx % len(particles_curr)
            
            if prev_idx in matched_prev or curr_idx in matched_curr:
                continue
            
            if distances[prev_idx, curr_idx] <= max_distance:
                # Find track ID for previous particle
                for track_id, track_data in tracks.items():
                    if track_data[-1][0] == frame_idx - 1:
                        if (track_data[-1][1], track_data[-1][2]) == particles_prev[prev_idx]:
                            # Extend this track
                            x, y = particles_curr[curr_idx]
                            tracks[track_id].append((frame_idx, x, y))
                            matched_prev.add(prev_idx)
                            matched_curr.add(curr_idx)
                            break
        
        # Create new tracks for unmatched particles
        for curr_idx, (x, y) in enumerate(particles_curr):
            if curr_idx not in matched_curr:
                tracks[next_track_id] = [(frame_idx, x, y)]
                next_track_id += 1
        
        particles_prev = particles_curr
    
    # Convert to DataFrame
    rows = []
    for track_id, track_data in tracks.items():
        if len(track_data) >= min_track_length:
            for frame, x, y in track_data:
                rows.append({'particle_id': track_id, 'frame': frame, 'x': x, 'y': y})
    
    df = pd.DataFrame(rows)
    
    if len(df) > 0:
        df = df.sort_values(['particle_id', 'frame']).reset_index(drop=True)
        logger.info(f"Tracked {df['particle_id'].nunique()} particles, {len(df)} total observations")
    else:
        logger.warning("No particles tracked!")
    
    return df


def compute_velocities(trajectories: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    """
    Compute velocities from particle trajectories.
    
    Args:
        trajectories: DataFrame with columns: particle_id, frame, x, y
        dt: Time step between frames
        
    Returns:
        DataFrame with added columns: vx, vy
    """
    df = trajectories.copy()
    df['vx'] = 0.0
    df['vy'] = 0.0
    
    for particle_id in df['particle_id'].unique():
        mask = df['particle_id'] == particle_id
        particle_data = df[mask].sort_values('frame')
        
        if len(particle_data) < 2:
            continue
        
        # Compute finite differences
        x = particle_data['x'].values
        y = particle_data['y'].values
        
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        
        # Assign velocities (use forward difference, repeat last value)
        indices = particle_data.index[:-1]
        df.loc[indices, 'vx'] = vx
        df.loc[indices, 'vy'] = vy
        
        # For last point, use same velocity as second-to-last
        if len(vx) > 0:
            df.loc[particle_data.index[-1], 'vx'] = vx[-1]
            df.loc[particle_data.index[-1], 'vy'] = vy[-1]
    
    logger.info(f"Computed velocities: mean |v| = {np.sqrt(df['vx']**2 + df['vy']**2).mean():.3f}")
    
    return df


def load_flow_sequence(flow_dir: str, pattern: str = "*.flo",
                       max_files: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load sequence of .flo flow files.
    
    Args:
        flow_dir: Directory containing .flo files
        pattern: Glob pattern for flow files
        max_files: Maximum number of files to load (None = all)
        
    Returns:
        Tuple of (flow_sequence, file_paths) where flow_sequence has shape
        (num_frames, height, width, 2)
    """
    flow_dir = Path(flow_dir)
    flow_files = sorted(flow_dir.glob(pattern))
    
    if max_files:
        flow_files = flow_files[:max_files]
    
    if len(flow_files) == 0:
        raise FileNotFoundError(f"No .flo files found in {flow_dir}")
    
    logger.info(f"Loading {len(flow_files)} flow files from {flow_dir}")
    
    flows = []
    file_paths = []
    
    for flo_path in tqdm(flow_files, desc="Loading flow files"):
        flow = read_flo_file(str(flo_path))
        flows.append(flow)
        file_paths.append(str(flo_path))
    
    flow_sequence = np.array(flows)
    logger.info(f"Loaded flow sequence with shape: {flow_sequence.shape}")
    
    return flow_sequence, file_paths


def sample_particles_from_flow(flow_sequence: np.ndarray, num_particles: int = 1000,
                                num_timesteps: int = 10) -> pd.DataFrame:
    """
    Sample particle trajectories from flow field sequence.
    
    Args:
        flow_sequence: Flow fields with shape (num_frames, height, width, 2)
        num_particles: Number of particles to sample
        num_timesteps: Number of timesteps to track each particle
        
    Returns:
        DataFrame with columns: particle_id, frame, x, y, vx, vy
    """
    num_frames, height, width, _ = flow_sequence.shape
    
    rows = []
    
    for particle_id in tqdm(range(num_particles), desc="Sampling particles"):
        # Random starting position
        start_frame = np.random.randint(0, max(1, num_frames - num_timesteps))
        x = np.random.uniform(0, width - 1)
        y = np.random.uniform(0, height - 1)
        
        # Track particle through flow field
        for t in range(num_timesteps):
            frame_idx = start_frame + t
            if frame_idx >= num_frames:
                break
            
            # Bilinear interpolation of flow at current position
            x_int = int(np.floor(x))
            y_int = int(np.floor(y))
            
            # Check bounds
            if x_int < 0 or x_int >= width - 1 or y_int < 0 or y_int >= height - 1:
                break
            
            # Interpolation weights
            wx = x - x_int
            wy = y - y_int
            
            # Bilinear interpolation
            flow_00 = flow_sequence[frame_idx, y_int, x_int]
            flow_10 = flow_sequence[frame_idx, y_int, x_int + 1]
            flow_01 = flow_sequence[frame_idx, y_int + 1, x_int]
            flow_11 = flow_sequence[frame_idx, y_int + 1, x_int + 1]
            
            vx = (1 - wx) * (1 - wy) * flow_00[0] + wx * (1 - wy) * flow_10[0] + \
                 (1 - wx) * wy * flow_01[0] + wx * wy * flow_11[0]
            vy = (1 - wx) * (1 - wy) * flow_00[1] + wx * (1 - wy) * flow_10[1] + \
                 (1 - wx) * wy * flow_01[1] + wx * wy * flow_11[1]
            
            rows.append({
                'particle_id': particle_id,
                'frame': frame_idx,
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy
            })
            
            # Advect particle
            x += vx
            y += vy
    
    df = pd.DataFrame(rows)
    logger.info(f"Sampled {num_particles} particles, {len(df)} total observations")
    
    return df


class FlowDataset(Dataset):
    """PyTorch dataset for flow field training."""
    
    def __init__(self, trajectories: pd.DataFrame, normalizer: Optional[Any] = None):
        """
        Initialize dataset.
        
        Args:
            trajectories: DataFrame with columns: particle_id, frame, x, y, vx, vy
            normalizer: Optional DataNormalizer for data normalization
        """
        self.data = trajectories.copy()
        self.normalizer = normalizer
        
        # Extract arrays
        self.x = self.data['x'].values.astype(np.float32)
        self.y = self.data['y'].values.astype(np.float32)
        self.t = self.data['frame'].values.astype(np.float32)
        self.vx = self.data['vx'].values.astype(np.float32)
        self.vy = self.data['vy'].values.astype(np.float32)
        
        # Normalize if normalizer provided
        if self.normalizer:
            self.x, self.y = self.normalizer.normalize_position(self.x, self.y)
            self.t = self.normalizer.normalize_time(self.t)
            self.vx, self.vy = self.normalizer.normalize_velocity(self.vx, self.vy)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': torch.tensor(self.x[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.float32),
            't': torch.tensor(self.t[idx], dtype=torch.float32),
            'vx': torch.tensor(self.vx[idx], dtype=torch.float32),
            'vy': torch.tensor(self.vy[idx], dtype=torch.float32)
        }


def create_datasets(trajectories: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15, normalizer: Optional[Any] = None,
                   random_seed: int = 42) -> Tuple[FlowDataset, FlowDataset, FlowDataset]:
    """
    Split trajectories into train/val/test datasets.
    
    Args:
        trajectories: DataFrame with particle trajectories
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        normalizer: Optional DataNormalizer
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Split by particle IDs to keep trajectories intact
    particle_ids = trajectories['particle_id'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(particle_ids)
    
    n_train = int(len(particle_ids) * train_ratio)
    n_val = int(len(particle_ids) * val_ratio)
    
    train_ids = particle_ids[:n_train]
    val_ids = particle_ids[n_train:n_train + n_val]
    test_ids = particle_ids[n_train + n_val:]
    
    train_data = trajectories[trajectories['particle_id'].isin(train_ids)]
    val_data = trajectories[trajectories['particle_id'].isin(val_ids)]
    test_data = trajectories[trajectories['particle_id'].isin(test_ids)]
    
    logger.info(f"Dataset split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    train_dataset = FlowDataset(train_data, normalizer)
    val_dataset = FlowDataset(val_data, normalizer)
    test_dataset = FlowDataset(test_data, normalizer)
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset: FlowDataset, val_dataset: FlowDataset,
                      test_dataset: FlowDataset, batch_size: int = 4096,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
