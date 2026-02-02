"""
Data preprocessing script
Merge feature and label files and store into processed/train|val|test according to a predefined split.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import shutil


def load_feature_file(feature_path: str) -> Dict[str, np.ndarray]:
    """
    Load a feature file (.npz).
    
    Args:
        feature_path: path to the feature file
        
    Returns:
        A dictionary containing 'tokens' and 'frame_ids'.
    """
    data = np.load(feature_path)
    return {
        'tokens': data['tokens'],
        'frame_ids': data['frame_ids']
    }


def load_label_file(label_path: str) -> Dict[str, np.ndarray]:
    """
    Load a label file (.npy).
    
    Args:
        label_path: path to the label file
        
    Returns:
        A dictionary containing 'phase_id' and 'future_schedule'.
    """
    data = np.load(label_path, allow_pickle=True).item()
    return data


def merge_feature_and_label(video_name: str, 
                            features_dir: str, 
                            labels_dir: str) -> Dict[str, np.ndarray]:
    """
    Merge features and labels for a single video.
    
    Args:
        video_name: video folder name (e.g., video01)
        features_dir: directory containing feature files
        labels_dir: directory containing label files
        
    Returns:
        A merged data dictionary for the video.
    """
    # Build file paths
    feature_file = os.path.join(features_dir, f'{video_name}.npz')
    label_file = os.path.join(labels_dir, f'{video_name}_labels.npy')
    
    # Check files
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    # Load data
    features = load_feature_file(feature_file)
    labels = load_label_file(label_file)
    
    # Merge
    video_num = int(video_name.replace('video', ''))
    merged_data = {
        'video_id': video_num,
        'tokens': features['tokens'],
        'frame_ids': features['frame_ids'],
        'phase_id': labels['phase_id'],
        'future_schedule': labels['future_schedule']
    }
    
    # Basic consistency checks
    n_frames = len(features['tokens'])
    if len(features['frame_ids']) != n_frames:
        print(f"Warning: {video_name} frame_ids length mismatch")
    if len(labels['phase_id']) != n_frames:
        print(f"Warning: {video_name} phase_id length mismatch")
    
    return merged_data


def process_and_save_dataset(features_root: str,
                            labels_dir: str,
                            output_dir: str,
                            manifest_path: str):
    """
    Process all videos and save them to the output directory.
    
    Args:
        features_root: feature root directory (with train/val/test subfolders)
        labels_dir: label directory
        output_dir: processed output directory (processed/train|val|test)
        manifest_path: path to split_manifest.json
    """
    # Reset output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    splits = {
        'train': (manifest.get('train', []), train_dir),
        'val': (manifest.get('val', []), val_dir),
        'test': (manifest.get('test', []), test_dir)
    }
    
    for split_name, (video_list, save_dir) in splits.items():
        print(f"Processing {split_name} split...")
        for video_name in video_list:
            try:
                # Merge features and labels
                merged_data = merge_feature_and_label(
                    video_name, os.path.join(features_root, split_name), labels_dir
                )
                
                # Save as compressed .npz
                output_file = os.path.join(save_dir, f'{video_name}.npz')
                np.savez_compressed(output_file, **merged_data)
                
                print(f"  ✓ saved {video_name}.npz "
                      f"(frames: {len(merged_data['tokens'])}, "
                      f"feature_dim: {merged_data['tokens'].shape[1]})")
                
            except FileNotFoundError as e:
                print(f"  ✗ skip {video_name}: {e}")
            except Exception as e:
                print(f"  ✗ error processing {video_name}: {e}")
    
    print("\nProcessing done!")
    print_dataset_summary(output_dir)


def print_dataset_summary(output_dir: str):
    """
    Print dataset summary.
    
    Args:
        output_dir: processed directory
    """
    print("\n" + "="*60)
    print("Dataset summary")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
            print(f"\n{split.upper()} split:")
            print(f"  file count: {len(files)}")
            print(f"  path: {split_dir}")
            if files:
                print(f"  video range: {files[0]} - {files[-1]}")
                
                # Inspect first file
                sample_file = os.path.join(split_dir, files[0])
                data = np.load(sample_file)
                print(f"  shape example ({files[0]}):")
                for key in data.keys():
                    print(f"    - {key}: {data[key].shape}")


def verify_data_integrity(output_dir: str):
    """
    Verify data integrity in processed splits.
    
    Args:
        output_dir: processed directory
    """
    print("\n" + "="*60)
    print("Data integrity check")
    print("="*60)
    
    all_good = True
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            print(f"✗ {split} directory not found!")
            all_good = False
            continue
            
        files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        print(f"\nChecking {split.upper()} split ({len(files)} files)...")
        
        for filename in files:
            filepath = os.path.join(split_dir, filename)
            try:
                data = np.load(filepath)
                
                # Required keys
                required_keys = ['video_id', 'tokens', 'frame_ids', 'phase_id', 'future_schedule']
                for key in required_keys:
                    if key not in data:
                        print(f"  ✗ {filename}: missing key '{key}'")
                        all_good = False
                
                # Length consistency
                n_frames = len(data['tokens'])
                if len(data['frame_ids']) != n_frames or len(data['phase_id']) != n_frames:
                    print(f"  ✗ {filename}: inconsistent lengths")
                    all_good = False
                    
            except Exception as e:
                print(f"  ✗ {filename}: failed to read - {e}")
                all_good = False
    
    if all_good:
        print("\n✓ All checks passed!")
    else:
        print("\n✗ Data issues found, please inspect!")


def main():
    """
    Entry point.
    """
    # Paths
    project_root = Path(__file__).parent.parent.parent
    features_dir = project_root / 'data' / 'features' / 'dinov2-base-cls'
    labels_dir = project_root / 'data' / 'labels' / 'aligned_labels'
    output_dir = project_root / 'data' / 'processed'
    manifest_path = project_root / 'data' / 'split_manifest.json'
    
    print("="*60)
    print("Data preprocessing")
    print("="*60)
    print(f"特征目录: {features_dir}")
    print(f"Label dir: {labels_dir}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Check inputs
    if not features_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    
    # Process dataset
    process_and_save_dataset(
        features_root=str(features_dir),
        labels_dir=str(labels_dir),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path)
    )
    
    # Verify integrity
    verify_data_integrity(str(output_dir))


if __name__ == '__main__':
    main()
