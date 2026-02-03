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
        self.best_score = -float('inf')
        self.train_history = []
        self.val_history = []
        
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
            mode='max',  # Maximize validation score
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
        
        Balanced combination of classification and regression:
        - F1 score (classification performance): range [0, 1]
        - RMSE score (regression performance): range [0, 1], lower RMSE is better
        
        Formula: val_score = 0.5 * F1 + 0.5 * (100 / (100 + RMSE))
        
        This ensures both tasks contribute equally to the final score.
        """
        # Convert RMSE to a score in [0, 1] range
        # When RMSE=0:   score=1.0
        # When RMSE=100: score=0.5
        # When RMSE=300: score=0.25
        rmse_score = 100.0 / (100.0 + metrics['rmse'])
        
        # Weighted combination (50% classification, 50% regression)
        score = 0.5 * metrics['f1'] + 0.5 * rmse_score
        
        return score
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'checkpoint_best.pth')
            print(f"  → Best model saved (score: {self.best_score:.4f})")
    
    def save_visualization_data(self, vis_data, epoch):
        """Save data for visualization"""
        if isinstance(epoch, int):
            save_path = self.save_dir / f'val_predictions_epoch{epoch:03d}.pt'
        else:
            save_path = self.save_dir / f'val_predictions_{epoch}.pt'
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
            
            # Compute validation score
            val_score = self.compute_validation_score(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_score)
            
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
            print(f"  Val Score: {val_score:.4f}")
            
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
                'score': val_score
            })
            
            # Save checkpoint and visualization data
            is_best_score = val_score > self.best_score
            is_best_f1 = val_metrics['f1'] > getattr(self, 'best_f1', -float('inf'))
            is_best_rmse = val_metrics['rmse'] < getattr(self, 'best_rmse', float('inf'))
            
            if is_best_score:
                self.best_score = val_score
                self.save_checkpoint(epoch + 1, is_best=True)
                self.save_visualization_data(vis_data, epoch + 1)
                print(f"  → Best Score model saved: {self.best_score:.4f}")
            
            if is_best_f1:
                self.best_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'f1': self.best_f1,
                    'config': self.config
                }, self.save_dir / 'checkpoint_best_f1.pth')
                self.save_visualization_data(vis_data, f"{epoch + 1}_f1")
                print(f"  → Best F1 model saved: {self.best_f1:.4f}")
            
            if is_best_rmse:
                self.best_rmse = val_metrics['rmse']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'rmse': self.best_rmse,
                    'config': self.config
                }, self.save_dir / 'checkpoint_best_rmse.pth')
                self.save_visualization_data(vis_data, f"{epoch + 1}_rmse")
                print(f"  → Best RMSE model saved: {self.best_rmse:.4f}")
            
            # Save latest checkpoint
            self.save_checkpoint(epoch + 1, is_best=False)
            
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
        print(f"Best validation score: {self.best_score:.4f}")
        print(f"Model saved to: {self.save_dir}")
        print("="*70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train surgical phase prediction model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/labels/aligned_labels',
                        help='Path to processed data directory')
    parser.add_argument('--save_dir', type=str, default='results/models/mstcn_exp',
                        help='Directory to save checkpoints')
    
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
