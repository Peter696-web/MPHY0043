"""
Generate Trajectory Predictions (Task A -> Task B)
Uses a trained Task A model (Baseline) to predict future schedules for all videos.
These predictions are saved as .npy files to be used as input for Task B.
"""

import sys
import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.model.mstcn import MSTCNSurgicalPredictor
from src.model.data import create_dataloaders

def generate_trajectories(checkpoint_path, data_dir, output_dir, device='cuda'):
    """
    Generate and save regression outputs for all dataset splits
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # 2. Rebuild Model
    print("Building model...")
    model = MSTCNSurgicalPredictor(
        feature_dim=768,
        hidden_dim=config.get('mstcn_channels', 64),
        num_stages=config.get('mstcn_stages', 4),
        num_layers=config.get('mstcn_layers', 10),
        dropout=0.0,  # No dropout during inference
        num_phases=7,
        use_external_time_input=False 
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded. Best MAE: {checkpoint.get('best_mae', 'N/A')}")

    # 3. Data Loaders
    print("Loading datasets...")
    # use config['data_dir'] if available, else argument
    load_dir = data_dir if data_dir else config.get('data_dir', 'data/processed')
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=load_dir,
        batch_size=1,      # Must be 1 for sequence processing
        num_workers=0,     # Set to 0 to avoid multiprocessing pickle error on macOS
        normalize_features=True,  # Match training
        normalize_schedule=True   # Typically True, but for generation we output what the model predicts
    )

    # 4. Inference Loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [('train', train_loader), ('val', val_loader), ('test', test_loader)]
    
    for split_name, loader in splits:
        print(f"\nProcessing {split_name} split ({len(loader)} videos)...")
        save_subdir = output_dir / split_name
        save_subdir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                features = batch['features'].to(device)
                video_id_tensor = batch['video_id']
                video_id = int(video_id_tensor.item())
                
                # Forward Pass
                outputs = model(features)
                
                # Get Regression Output: future_schedule
                # Shape: (B, T, Phases, 2)
                pred_schedule = outputs['future_schedule']
                
                # Convert to numpy
                pred_numpy = pred_schedule.cpu().numpy()[0] # (T, 7, 2)
                
                # Save
                save_name = f"video{video_id:02d}_schedule.npy"
                np.save(save_subdir / save_name, pred_numpy)
                
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(loader)} videos")
                    
    print(f"\nAll predictions saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate trajectory predictions (Task A outputs)")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the Best MAE checkpoint (Task A)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to processed data (features)')
    parser.add_argument('--output_dir', type=str, default='data/estimated_times',
                        help='Where to save the generated .npy files')
    
    args = parser.parse_args()
    
    generate_trajectories(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
