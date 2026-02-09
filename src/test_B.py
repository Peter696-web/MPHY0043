"""
Evaluation Script for Task B (Pure Classification)
Loads a trained checkpoint and evaluates on the Test Set.
Saves predictions for visualization.
"""

import os
import json
import time
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.data import create_dataloaders
from model.metrics import MetricsCalculator
from model.mstcn_B import MSTCNSurgicalPredictor

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Override data dir if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir
        
    print(f"   Model Config loaded. External Time Injection: {config.get('use_external_time_input')}")

    # 2. Build Model
    model = MSTCNSurgicalPredictor(
        feature_dim=config['feature_dim'],
        hidden_dim=config.get('mstcn_channels', 64),
        num_stages=config.get('mstcn_stages', 4),
        num_layers=config.get('mstcn_layers', 10),
        dropout=config['dropout'],
        num_phases=config['num_phases'],
        use_external_time_input=config.get('use_external_time_input', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 3. Data Loader (Test Set)
    print("\nLoading Test Data...")
    _, _, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=1,
        num_workers=args.num_workers,
        normalize_features=config['normalize_features'],
        normalize_schedule=config['normalize_schedule'],
        use_external_time=config.get('use_external_time_input', True),
        seed=42 # Fixed seed for evaluation
    )
    
    # 4. Evaluation Loop
    metrics_calc = MetricsCalculator()
    all_predictions = []
    
    print(f"🚀 Starting Evaluation on {len(test_loader)} videos...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            features = batch['features'].to(device)
            targets = {'phase_id': batch['phase_id'].to(device)}
            
            predictions = model(features)
            
            # Update Metrics
            metrics_calc.update(predictions, batch)
            
            # Store predictions for saving
            all_predictions.append({
                'phase_logits': predictions['phase_logits'].cpu(),
                # We can store targets/frame_ids if needed, but visualize_B retrieves them from disk/dataloaders usually
                # But to rely less on order, let's keep it simple as expected by visualize_video_B.py
                # visualize_video_B.py expects a list of dicts with 'phase_logits'
            })
            
            if (i+1) % 10 == 0:
                print(f"   Processed {i+1}/{len(test_loader)} videos")
    
    # 5. Compute & Print Metrics
    metrics = metrics_calc.compute()
    print("\n" + "="*30)
    print("TEST SET RESULTS")
    print("="*30)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("="*30)
    
    # 6. Save Predictions
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'predictions': all_predictions}, save_path)
        print(f"\n✅ Predictions saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Task B Model on Test Set')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data directory')
    parser.add_argument('--save_path', type=str, default='results/models/task_B_input_injection/test_predictions.pt', help='Where to save predictions')
    parser.add_argument('--num_workers', type=int, default=0)
    
    args = parser.parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()
