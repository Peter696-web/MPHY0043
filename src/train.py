"""
Training Script for Surgical Phase Prediction
Multi-task learning: Phase classification + Future schedule prediction
"""

import os
import json
import time
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model.data import create_dataloaders
from model.loss import MultiTaskLoss
from model.metrics import MetricsCalculator
from model.mstcn import MSTCNSurgicalPredictor


class Trainer:
    """Training manager for surgical phase prediction"""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Seed for reproducibility
        self._set_seed(self.config.get('seed', 42))
        torch.cuda.empty_cache()
        
        # Create save directory
        self.save_dir = Path(config['save_dir'])
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize model, loss, optimizer
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        
        # Initialize data loaders
        self._build_dataloaders()
        
        # Training state
        self.start_epoch = 0
        self.best_score = float('-inf')  # Track best composite score (higher is better)
        self.best_f1 = 0.0  # Track for logging
        self.best_mae = float('inf')  # Track for logging
        self.train_history = []
        self.val_history = []
        self.best_checkpoint_path = None  # Track the path of the current best checkpoint
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _build_model(self):
        """Build model"""
        # Only use MS-TCN++ model
        model_type = self.config.get('model_type', 'mstcn')

        if model_type != 'mstcn':
            raise ValueError(f"Unknown model type: {model_type}. Only 'mstcn' is supported now.")

        self.model = MSTCNSurgicalPredictor(
            feature_dim=self.config['feature_dim'],
            hidden_dim=self.config.get('mstcn_channels', 64),
            num_stages=self.config.get('mstcn_stages', 4),
            num_layers=self.config.get('mstcn_layers', 10),
            dropout=self.config['dropout'],
            num_phases=self.config['num_phases']
        )
            
        self.model = self.model.to(self.device)
    
    def _build_loss(self):
        """Build loss function"""
        self.criterion = MultiTaskLoss(
            alpha=self.config['loss_alpha'],
            beta=self.config['loss_beta'],
            use_mstcn=True
        )
    
    def _build_optimizer(self):
        """Build optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # Minimize MAE
            factor=0.5,
            patience=self.config['lr_patience'],
            verbose=True
        )
    
    def _build_dataloaders(self):
        """Build data loaders"""
        print("\nLoading data...")
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            normalize_features=self.config['normalize_features'],
            normalize_schedule=self.config['normalize_schedule'],
            seed=self.config.get('seed', 42)
        )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        epoch_losses = []
        metrics_calc = MetricsCalculator()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            features = batch['features'].to(self.device)
            targets = {
                'phase_id': batch['phase_id'].to(self.device),
                'future_schedule': batch['future_schedule'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(features)
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Record losses
            epoch_losses.append({
                'total': losses['total'].item(),
                'classification': losses['classification'].item(),
                'regression': losses['regression'].item()
            })
            
            # Update metrics
            metrics_calc.update(predictions, batch)
            
            # Print progress
            if (batch_idx + 1) % self.config['print_freq'] == 0:
                avg_loss = np.mean([x['total'] for x in epoch_losses[-100:]])
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
        
        # Compute epoch metrics
        elapsed = time.time() - start_time
        avg_losses = {
            'total': np.mean([x['total'] for x in epoch_losses]),
            'classification': np.mean([x['classification'] for x in epoch_losses]),
            'regression': np.mean([x['regression'] for x in epoch_losses])
        }
        metrics = metrics_calc.compute()
        
        return avg_losses, metrics, elapsed
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        val_losses = []
        metrics_calc = MetricsCalculator()
        
        # For visualization: save predictions
        all_predictions = []
        all_targets = []
        all_video_ids = []
        all_frame_ids = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                features = batch['features'].to(self.device)
                targets = {
                    'phase_id': batch['phase_id'].to(self.device),
                    'future_schedule': batch['future_schedule'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(features)
                losses = self.criterion(predictions, targets)
                
                # Record losses
                val_losses.append({
                    'total': losses['total'].item(),
                    'classification': losses['classification'].item(),
                    'regression': losses['regression'].item()
                })
                
                # Update metrics
                metrics_calc.update(predictions, batch)
                
                # Save for visualization
                all_predictions.append({
                    'phase_logits': predictions['phase_logits'].cpu(),
                    'future_schedule': predictions['future_schedule'].cpu()
                })
                all_targets.append({
                    'phase_id': batch['phase_id'].cpu(),
                    'future_schedule': batch['future_schedule'].cpu()
                })
                all_video_ids.append(batch['video_id'].cpu())
                all_frame_ids.append(batch['frame_id'].cpu())
        
        # Compute metrics
        avg_losses = {
            'total': np.mean([x['total'] for x in val_losses]),
            'classification': np.mean([x['classification'] for x in val_losses]),
            'regression': np.mean([x['regression'] for x in val_losses])
        }
        metrics = metrics_calc.compute()
        
        # Prepare visualization data
        vis_data = {
            'predictions': all_predictions,
            'targets': all_targets,
            'video_ids': all_video_ids,
            'frame_ids': all_frame_ids
        }
        
        return avg_losses, metrics, vis_data
    
    def compute_validation_score(self, metrics):
        """
        Compute validation score for model selection
        
        Combine F1 and MAE into a composite score:
        - F1: higher is better (range 0-1)
        - MAE: lower is better (convert to range 0-1 by normalization)
        
        Composite Score = w1 * F1 + w2 * (1 - normalized_MAE)
        where normalized_MAE = MAE / 100 (assuming MAE typically < 100)
        
        Returns:
            float: Composite score (higher is better)
        """
        # Weights for F1 and MAE (can be tuned)
        w_f1 = 0.5
        w_mae = 0.5
        
        # Normalize MAE to 0-1 range (inverse: lower MAE = higher score)
        # Assume typical MAE is in range [0, 100]
        mae_normalized = min(metrics['mae'] / 100.0, 1.0)
        mae_score = 1.0 - mae_normalized  # Convert to "higher is better"
        
        # Composite score
        score = w_f1 * metrics['f1'] + w_mae * mae_score
        
        return score
    
    def save_checkpoint(self, epoch, is_best=False, composite_score=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'best_f1': self.best_f1,
            'best_mae': self.best_mae,
            'config': self.config
        }
        
        # Save best checkpoint (overwrites previous best)
        if is_best:
            try:
                # Remove previous best checkpoint if it exists
                if self.best_checkpoint_path and self.best_checkpoint_path.exists():
                    self.best_checkpoint_path.unlink()
                
                # Cleanup pattern match just in case
                for f in self.save_dir.glob('checkpoint_best_epoch*.pth'):
                     try:
                         if self.best_checkpoint_path and f != self.best_checkpoint_path:
                             f.unlink()
                     except: pass
                
                # Remove previous visualisations
                for f in self.save_dir.glob('val_predictions_*.pt'):
                    f.unlink()

            except Exception as e:
                print(f"Warning: could not delete old files: {e}")

            # New filename with epoch
            best_name = f'checkpoint_best_epoch{epoch}.pth'
            self.best_checkpoint_path = self.save_dir / best_name
            
            torch.save(checkpoint, self.best_checkpoint_path)
            print(f"  → Best model saved to {best_name} (MAE: {self.best_mae:.4f}, F1: {self.best_f1:.4f})")
    
    def save_visualization_data(self, vis_data, epoch):
        """Save data for visualization"""
        # Save correspond to the best epoch
        save_path = self.save_dir / f'val_predictions_best_epoch{epoch}.pt'
        torch.save(vis_data, save_path)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting training...")
        print("="*70)
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\nEpoch [{epoch+1}/{self.config['num_epochs']}]")
            print("-" * 70)
            
            # Train
            train_losses, train_metrics, train_time = self.train_epoch(epoch)
            
            # Validate
            val_losses, val_metrics, vis_data = self.validate()
            
            # Compute validation score - Now only MAE
            # val_score = self.compute_validation_score(val_metrics)
            
            # Update learning rate (still use MAE for scheduler)
            self.scheduler.step(val_metrics['mae'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary (time: {train_time:.1f}s):")
            print(f"  Train Loss: {train_losses['total']:.4f} "
                  f"(cls: {train_losses['classification']:.4f}, "
                  f"reg: {train_losses['regression']:.4f})")
            print(f"  Val   Loss: {val_losses['total']:.4f} "
                  f"(cls: {val_losses['classification']:.4f}, "
                  f"reg: {val_losses['regression']:.4f})")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, "
                  f"MAE: {train_metrics['mae']:.4f}")
            print(f"  Val   Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"MAE: {val_metrics['mae']:.4f}")
            print(f"  Current Best MAE: {self.best_mae:.4f}")
            
            # Save history
            self.train_history.append({
                'epoch': epoch + 1,
                'losses': train_losses,
                'metrics': train_metrics
            })
            self.val_history.append({
                'epoch': epoch + 1,
                'losses': val_losses,
                'metrics': val_metrics,
                # 'composite_score': val_score
            })
            
            # Check if this is the best model based on MAE (lower is better)
            # If MAE is same (very rare for floats, but checking approx equal), compare F1
            is_best = False
            val_mae = val_metrics['mae']
            val_f1 = val_metrics['f1']
            
            if val_mae < self.best_mae:
                is_best = True
            elif abs(val_mae - self.best_mae) < 1e-6:
                if val_f1 > self.best_f1:
                    is_best = True
                    print(f"  → Same MAE ({val_mae:.4f}), better F1: {val_f1:.4f} > {self.best_f1:.4f}")
            
            if is_best:
                self.best_score = val_mae 
                self.best_f1 = val_f1
                self.best_mae = val_mae
                self.save_checkpoint(epoch + 1, is_best=True, composite_score=val_mae)
                self.save_visualization_data(vis_data, epoch + 1)
            
            # We do NOT save the latest checkpoint every epoch to save space
            # self.save_checkpoint(epoch + 1, is_best=False)
            
            # Save training history
            history = {
                'train': self.train_history,
                'val': self.val_history
            }
            with open(self.save_dir / 'history.json', 'w') as f:
                # Convert numpy types to native Python types
                def convert(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                history_clean = json.loads(
                    json.dumps(history, default=convert)
                )
                json.dump(history_clean, f, indent=2)
        
        print("\n" + "="*70)
        print("Training completed!")
        print(f"Best validation composite score: {self.best_score:.4f}")
        print(f"  → Best F1: {self.best_f1:.4f}")
        print(f"  → Best MAE: {self.best_mae:.4f}")
        print(f"Model saved to: {self.save_dir}")
        print("="*70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train surgical phase prediction model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--exp_name', type=str, default='mstcn_exp',
                        help='Experiment name (auto creates save_dir)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (overrides exp_name)')
    
    # Model (simplified)
    parser.add_argument('--channels', type=int, default=64,
                        help='MS-TCN feature channels')
    parser.add_argument('--stages', type=int, default=4,
                        help='Number of refinement stages')
    parser.add_argument('--layers', type=int, default=10,
                        help='Dilated conv layers per stage')
    
    # Training (simplified)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    
    # Optional
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Auto-generate save_dir from exp_name if not specified
    if args.save_dir is None:
        args.save_dir = f'results/models/{args.exp_name}'
    
    # Convert to full config dict with defaults
    config = {
        # Data
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        
        # Model
        'model_type': 'mstcn',
        'feature_dim': 768,
        'mstcn_channels': args.channels,
        'mstcn_stages': args.stages,
        'mstcn_layers': args.layers,
        'dropout': 0.3,
        'num_phases': 7,
        
        # Loss
        'loss_alpha': 1.0,
        'loss_beta': 0.5,
        'loss_gamma': 0.15,
        
        # Training
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'grad_clip': 5.0,
        'lr_patience': 5,
        
        # Data loading
        'num_workers': args.workers,
        'normalize_features': True,
        'normalize_schedule': True,
        'seed': args.seed,
        
        # Logging
        'print_freq': 100,
        'save_vis_freq': 5
    }
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
