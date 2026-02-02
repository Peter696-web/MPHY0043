"""
数据加载器模块
用于加载预处理后的手术视频特征和标签数据
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
    手术阶段预测数据集
    
    加载预处理后的 .npz 文件，每个文件包含:
        - video_id: 视频编号
        - tokens: DINOv2 特征 (N, 768)
        - frame_ids: 帧编号 (N,)
        - phase_id: 阶段标签 (N,)
        - future_schedule: 未来阶段时间表 (N, 7, 2)
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 normalize_features: bool = False,
                 normalize_schedule: bool = True,
                 cache_data: bool = False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录 (包含 train/val/test 子目录)
            split: 数据集分割 ('train', 'val', 'test')
            transform: 可选的数据增强
            normalize_schedule: 是否按视频长度归一化未来时间表
            cache_data: 是否缓存所有数据到内存
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
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            dict with keys:
                - features: (T, 768) 特征序列
                - phase_id: 当前阶段标签 (0-6)
                - future_schedule: (7, 2) 未来阶段时间表
                - frame_id: 帧编号序列
                - video_id: 视频编号
        """
        file_idx, _ = self.index_map[idx]
        
        # 加载数据
        if self.cached_data is not None:
            data = self.cached_data[file_idx]
        else:
            filepath = self.file_list[file_idx]
            data = np.load(filepath)
        
        video_id = int(data['video_id'])
        video_len = max(self.video_lengths[file_idx], 1)

        # 整个视频序列 (不截取末帧)
        features = data['tokens'].astype(np.float32)  # (T, 768)
        phase_id = data['phase_id'].astype(np.int64)  # (T,)
        future_schedule = data['future_schedule'].astype(np.float32)  # (T, 7, 2)
        frame_ids = data['frame_ids'].astype(np.int64)

        # 特征归一化：逐帧 L2
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
        获取整个视频的数据
        
        Args:
            video_id: 视频编号 (1-80)
            
        Returns:
            dict with all frames from the video
        """
        for info in self.video_info:
            if info['video_id'] == video_id:
                filepath = info['filepath']
                return dict(np.load(filepath))
        
        raise ValueError(f"视频 {video_id} 未找到")
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'n_videos': len(self.file_list),
            'n_frames': self.total_frames,
            'video_info': self.video_info,
            'split': self.split
        }
        
        # 计算阶段分布
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


def _worker_init_fn(worker_id: int, base_seed: int = 42):
    """Worker init at module scope to be picklable."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(data_dir: str,
                      batch_size: int = 1,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      normalize_features: bool = False,
                      normalize_schedule: bool = True,
                      cache_data: bool = False,
                      seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据根目录
        batch_size: 批次大小（固定为1，每批次一个视频）
        num_workers: 数据加载线程数
        pin_memory: 是否锁页内存 (GPU训练时推荐)
        normalize_features: 是否归一化特征
        normalize_schedule: 是否按视频长度对 future_schedule 做归一化
        cache_data: 是否缓存数据到内存
        seed: 随机种子，用于 worker 初始化
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 固定 batch_size=1：每批次一个视频
    if batch_size != 1:
        print("[WARN] 序列训练已固定 batch_size=1，忽略传入的 batch_size")
        batch_size = 1

    # 创建数据集
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
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 按视频顺序
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(None if num_workers == 0 else lambda wid: _worker_init_fn(wid, seed))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(None if num_workers == 0 else lambda wid: _worker_init_fn(wid, seed))
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(None if num_workers == 0 else lambda wid: _worker_init_fn(wid, seed))
    )
    
    # 打印统计信息
    print("\n" + "="*70)
    print("数据集统计信息")
    print("="*70)
    
    for name, dataset in [('训练集', train_dataset), 
                          ('验证集', val_dataset), 
                          ('测试集', test_dataset)]:
        stats = dataset.get_statistics()
        print(f"\n{name}:")
        print(f"  视频数: {stats['n_videos']}")
        print(f"  帧数: {stats['n_frames']}")
        print(f"  批次数: {len(train_loader) if name == '训练集' else len(val_loader) if name == '验证集' else len(test_loader)}")
        print(f"  阶段分布:")
        for phase in range(7):
            print(f"    Phase {phase}: {stats['phase_distribution'][phase]:6d} ({stats['phase_percentages'][phase]:5.2f}%)")
    
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader



