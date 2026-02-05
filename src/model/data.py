"""
Data loader module
For loading preprocessed surgical video features and label data
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class SurgicalPhaseDataset(Dataset):
    """
    Surgical phase prediction dataset

    Loads preprocessed .npz files, each containing:
        - video_id: Video number
        - tokens: DINOv2 features (N, 768)
        - frame_ids: Frame numbers (N,)
        - phase_id: Phase labels (N,)
        - future_schedule: Future phase schedule (N, 7, 2)
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 normalize_features: bool = False,
                 normalize_schedule: bool = True,
                 cache_data: bool = False):
        """
        Initialize dataset

        Args:
            data_dir: Data root directory (containing train/val/test subdirectories)
            split: Dataset split ('train', 'val', 'test')
            transform: Optional data augmentation
            normalize_schedule: Whether to normalize future schedule by video length
            cache_data: Whether to cache all data to memory
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.transform = transform
        self.normalize_features = normalize_features
        self.normalize_schedule = normalize_schedule
        self.cache_data = cache_data

        # Get all .npz files
        self.file_list = sorted(list(self.split_dir.glob('*.npz')))



        # Build file_idx
        self.index_map = []
        self.video_info = []
        self.video_lengths = []

        print(f"\nload {split} dataset...")
        self._build_index()

        self.cached_data = {} if cache_data else None
        if cache_data:
            print(f"cache all data to memory...")
            self._cache_all_data()

    def _build_index(self):
        total_frames = 0

        for file_idx, filepath in enumerate(self.file_list):
            try:
                data = np.load(filepath)
                n_frames = len(data['tokens'])
                video_id = int(data['video_id'])
                self.video_lengths.append(n_frames)

                # video info
                self.video_info.append({
                    'file_idx': file_idx,
                    'filepath': filepath,
                    'video_id': video_id,
                    'n_frames': n_frames,
                    'start_idx': total_frames,
                    'end_idx': total_frames + n_frames
                })

                self.index_map.append((file_idx, -1))

                total_frames += n_frames

            except Exception as e:
                print(f"Warning: load {filepath.name} failed: {e}")

        self.total_frames = total_frames
        print(f"  ✓ load {len(self.file_list)} videos, {self.total_frames} frames")

    def _cache_all_data(self):
        """cache all data to memory"""
        for file_idx, filepath in enumerate(self.file_list):
            self.cached_data[file_idx] = np.load(filepath)

    def __len__(self) -> int:
        """Return: video count"""
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample

        Args:
            idx: Sample index

        Returns:
            dict with keys:
                - features: (T, 768) feature sequence
                - phase_id: Current phase label (0-6)
                - future_schedule: (7, 2) future phase schedule
                - frame_id: Frame number sequence
                - video_id: Video number
        """
        file_idx, _ = self.index_map[idx]

        # Load data
        if self.cached_data is not None:
            data = self.cached_data[file_idx]
        else:
            filepath = self.file_list[file_idx]
            data = np.load(filepath)

        video_id = int(data['video_id'])
        video_len = max(self.video_lengths[file_idx], 1)

        # Entire video sequence (no end frame truncation)
        features = data['tokens'].astype(np.float32)  # (T, 768)
        phase_id = data['phase_id'].astype(np.int64)  # (T,)
        future_schedule = data['future_schedule'].astype(np.float32)  # (T, 7, 2)
        frame_ids = data['frame_ids'].astype(np.int64)

        # Feature normalization: per-frame L2
        if self.normalize_features:
            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            features = features / norms

        if self.normalize_schedule:
            future_schedule = future_schedule / float(video_len)

        if self.transform is not None:
            features = self.transform(features)

        sample = {
            'features': torch.from_numpy(features),  # (T, 768)
            'phase_id': torch.from_numpy(phase_id),               # (T,)
            'future_schedule': torch.from_numpy(future_schedule), # (T, 7, 2)
            'frame_id': torch.from_numpy(frame_ids),
            'video_id': torch.tensor(video_id, dtype=torch.long),
            'seq_len': torch.tensor(features.shape[0], dtype=torch.long)
        }

        return sample

    def get_video_data(self, video_id: int) -> Dict[str, np.ndarray]:
        """
        Get entire video data

        Args:
            video_id: Video number (1-80)

        Returns:
            dict with all frames from the video
        """
        for info in self.video_info:
            if info['video_id'] == video_id:
                filepath = info['filepath']
                return dict(np.load(filepath))

        raise ValueError(f"Video {video_id} not found")

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'n_videos': len(self.file_list),
            'n_frames': self.total_frames,
            'video_info': self.video_info,
            'split': self.split
        }

        # Calculate phase distribution
        phase_counts = np.zeros(7, dtype=int)
        for file_idx in range(len(self.file_list)):
            if self.cached_data is not None:
                data = self.cached_data[file_idx]
            else:
                data = np.load(self.file_list[file_idx])

            phases = data['phase_id']
            for phase in range(7):
                phase_counts[phase] += np.sum(phases == phase)

        stats['phase_distribution'] = phase_counts
        stats['phase_percentages'] = phase_counts / phase_counts.sum() * 100

        return stats


def create_dataloaders(data_dir: str,
                      batch_size: int = 1,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      normalize_features: bool = False,
                      normalize_schedule: bool = True,
                      cache_data: bool = False,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation and test data loaders

    Args:
        data_dir: Data root directory
        batch_size: Batch size (fixed to 1, one video per batch)
        num_workers: Data loading thread count
        pin_memory: Whether to pin memory (recommended for GPU training)
        normalize_features: Whether to normalize features
        normalize_schedule: Whether to normalize future_schedule by video length
        cache_data: Whether to cache data to memory
        seed: Random seed for worker initialization

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Fixed batch_size=1: one video per batch
    if batch_size != 1:
        print("[WARN] Sequence training has fixed batch_size=1, ignoring input batch_size")
        batch_size = 1

    # Create datasets
    train_dataset = SurgicalPhaseDataset(
        data_dir=data_dir,
        split='train',
        normalize_features=normalize_features,
        normalize_schedule=normalize_schedule,
        cache_data=cache_data
    )

    val_dataset = SurgicalPhaseDataset(
        data_dir=data_dir,
        split='val',
        normalize_features=normalize_features,
        normalize_schedule=normalize_schedule,
        cache_data=cache_data
    )

    test_dataset = SurgicalPhaseDataset(
        data_dir=data_dir,
        split='test',
        normalize_features=normalize_features,
        normalize_schedule=normalize_schedule,
        cache_data=cache_data
    )

    # Worker random seed for reproducibility
    def _worker_init_fn(worker_id: int):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential by video
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn
    )

    # Print statistics
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)

    for name, dataset in [('Training set', train_dataset),
                          ('Validation set', val_dataset),
                          ('Test set', test_dataset)]:
        stats = dataset.get_statistics()
        print(f"\n{name}:")
        print(f"  Video count: {stats['n_videos']}")
        print(f"  Frame count: {stats['n_frames']}")
        print(f"  Batch count: {len(train_loader) if name == 'Training set' else len(val_loader) if name == 'Validation set' else len(test_loader)}")
        print(f"  Phase distribution:")
        for phase in range(7):
            print(f"    Phase {phase}: {stats['phase_distribution'][phase]:6d} ({stats['phase_percentages'][phase]:5.2f}%)")

    print("="*70 + "\n")

    return train_loader, val_loader, test_loader



