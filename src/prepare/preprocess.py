"""
数据预处理脚本
将特征文件和标签文件合并，并按照 8:1:1 的比例分割为训练集、验证集和测试集
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import shutil


def load_feature_file(feature_path: str) -> Dict[str, np.ndarray]:
    """
    加载特征文件 (.npz)
    
    Args:
        feature_path: 特征文件路径
        
    Returns:
        包含 'tokens' 和 'frame_ids' 的字典
    """
    data = np.load(feature_path)
    return {
        'tokens': data['tokens'],
        'frame_ids': data['frame_ids']
    }


def load_label_file(label_path: str) -> Dict[str, np.ndarray]:
    """
    加载标签文件 (.npy)
    
    Args:
        label_path: 标签文件路径
        
    Returns:
        包含 'phase_id' 和 'future_schedule' 的字典
    """
    data = np.load(label_path, allow_pickle=True).item()
    return data


def merge_feature_and_label(video_num: int, 
                            features_dir: str, 
                            labels_dir: str) -> Dict[str, np.ndarray]:
    """
    合并单个视频的特征和标签
    
    Args:
        video_num: 视频编号 (1-80)
        features_dir: 特征文件夹路径
        labels_dir: 标签文件夹路径
        
    Returns:
        合并后的数据字典
    """
    # 构建文件路径
    feature_file = os.path.join(features_dir, f'video{video_num:02d}.npz')
    label_file = os.path.join(labels_dir, f'video{video_num:02d}_labels.npy')
    
    # 检查文件是否存在
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"特征文件不存在: {feature_file}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"标签文件不存在: {label_file}")
    
    # 加载数据
    features = load_feature_file(feature_file)
    labels = load_label_file(label_file)
    
    # 合并数据
    merged_data = {
        'video_id': video_num,
        'tokens': features['tokens'],
        'frame_ids': features['frame_ids'],
        'phase_id': labels['phase_id'],
        'future_schedule': labels['future_schedule']
    }
    
    # 验证数据长度一致性
    n_frames = len(features['tokens'])
    if len(features['frame_ids']) != n_frames:
        print(f"警告: video{video_num:02d} frame_ids 长度不匹配")
    if len(labels['phase_id']) != n_frames:
        print(f"警告: video{video_num:02d} phase_id 长度不匹配")
    
    return merged_data


def split_dataset(total_videos: int, 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, 
                 test_ratio: float = 0.1) -> Tuple[list, list, list]:
    """
    按照指定比例分割数据集
    
    Args:
        total_videos: 总视频数量
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        (train_videos, val_videos, test_videos) 三个列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于 1.0"
    
    video_ids = list(range(1, total_videos + 1))
    
    n_train = int(total_videos * train_ratio)
    n_val = int(total_videos * val_ratio)
    
    train_videos = video_ids[:n_train]
    val_videos = video_ids[n_train:n_train + n_val]
    test_videos = video_ids[n_train + n_val:]
    
    return train_videos, val_videos, test_videos


def process_and_save_dataset(features_dir: str,
                            labels_dir: str,
                            output_dir: str,
                            total_videos: int = 80,
                            train_ratio: float = 0.8,
                            val_ratio: float = 0.1,
                            test_ratio: float = 0.1):
    """
    处理所有视频并保存到指定的输出目录
    
    Args:
        features_dir: 特征文件夹路径
        labels_dir: 标签文件夹路径
        output_dir: 输出目录
        total_videos: 总视频数量
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 分割数据集
    train_videos, val_videos, test_videos = split_dataset(
        total_videos, train_ratio, val_ratio, test_ratio
    )
    
    print(f"\n数据集分割:")
    print(f"训练集: {len(train_videos)} 个视频 (video{train_videos[0]:02d} - video{train_videos[-1]:02d})")
    print(f"验证集: {len(val_videos)} 个视频 (video{val_videos[0]:02d} - video{val_videos[-1]:02d})")
    print(f"测试集: {len(test_videos)} 个视频 (video{test_videos[0]:02d} - video{test_videos[-1]:02d})")
    print()
    
    # 处理每个分割
    splits = {
        'train': (train_videos, train_dir),
        'val': (val_videos, val_dir),
        'test': (test_videos, test_dir)
    }
    
    for split_name, (video_list, save_dir) in splits.items():
        print(f"处理 {split_name} 集...")
        for video_num in video_list:
            try:
                # 合并特征和标签
                merged_data = merge_feature_and_label(
                    video_num, features_dir, labels_dir
                )
                
                # 保存为 .npz 文件（压缩格式，节省空间）
                output_file = os.path.join(save_dir, f'video{video_num:02d}.npz')
                np.savez_compressed(output_file, **merged_data)
                
                print(f"  ✓ 保存 video{video_num:02d}.npz "
                      f"(frames: {len(merged_data['tokens'])}, "
                      f"feature_dim: {merged_data['tokens'].shape[1]})")
                
            except FileNotFoundError as e:
                print(f"  ✗ 跳过 video{video_num:02d}: {e}")
            except Exception as e:
                print(f"  ✗ 处理 video{video_num:02d} 时出错: {e}")
    
    print("\n处理完成！")
    print_dataset_summary(output_dir)


def print_dataset_summary(output_dir: str):
    """
    打印数据集摘要信息
    
    Args:
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("数据集摘要")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
            print(f"\n{split.upper()} 集:")
            print(f"  文件数量: {len(files)}")
            print(f"  文件路径: {split_dir}")
            if files:
                print(f"  视频范围: {files[0]} - {files[-1]}")
                
                # 读取第一个文件查看数据形状
                sample_file = os.path.join(split_dir, files[0])
                data = np.load(sample_file)
                print(f"  数据形状示例 ({files[0]}):")
                for key in data.keys():
                    print(f"    - {key}: {data[key].shape}")


def verify_data_integrity(output_dir: str):
    """
    验证生成的数据完整性
    
    Args:
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("数据完整性验证")
    print("="*60)
    
    all_good = True
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            print(f"✗ {split} 目录不存在!")
            all_good = False
            continue
            
        files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        print(f"\n检查 {split.upper()} 集 ({len(files)} 个文件)...")
        
        for filename in files:
            filepath = os.path.join(split_dir, filename)
            try:
                data = np.load(filepath)
                
                # 检查必需的键
                required_keys = ['video_id', 'tokens', 'frame_ids', 'phase_id', 'future_schedule']
                for key in required_keys:
                    if key not in data:
                        print(f"  ✗ {filename}: 缺少键 '{key}'")
                        all_good = False
                
                # 检查数据长度一致性
                n_frames = len(data['tokens'])
                if len(data['frame_ids']) != n_frames or len(data['phase_id']) != n_frames:
                    print(f"  ✗ {filename}: 数据长度不一致")
                    all_good = False
                    
            except Exception as e:
                print(f"  ✗ {filename}: 读取失败 - {e}")
                all_good = False
    
    if all_good:
        print("\n✓ 所有数据验证通过!")
    else:
        print("\n✗ 发现数据问题，请检查!")


def main():
    """
    主函数
    """
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / 'data' / 'features' / 'dinov2-base-cls'
    labels_dir = project_root / 'data' / 'labels' / 'aligned_labels'
    output_dir = project_root / 'data' / 'processed'
    
    print("="*60)
    print("数据预处理脚本")
    print("="*60)
    print(f"特征目录: {features_dir}")
    print(f"标签目录: {labels_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 检查输入目录是否存在
    if not features_dir.exists():
        raise FileNotFoundError(f"特征目录不存在: {features_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
    
    # 处理数据集
    process_and_save_dataset(
        features_dir=str(features_dir),
        labels_dir=str(labels_dir),
        output_dir=str(output_dir),
        total_videos=80,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # 验证数据完整性
    verify_data_integrity(str(output_dir))


if __name__ == '__main__':
    main()
