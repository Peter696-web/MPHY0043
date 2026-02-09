"""
Training Script for Task B (Pure Classification with Input Injection)
Only computes Classification and Smoothing Loss. No Regression Loss.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.loss import FocalLoss
from model.data import create_dataloaders
from model.metrics import MetricsCalculator
from model.mstcn_B import MSTCNSurgicalPredictor  # Import from modified class



class ClassificationLoss(nn.Module):
    """
    Task B Loss: Classification (Focal) + Temporal Smoothing
    No Regression.
    """
    def __init__(self,  gamma=0.15):
        super().__init__()
        self.gamma = gamma # Smoothing weight
        self.ce_loss = FocalLoss(gamma=2.0)
    
    def temporal_smoothing_loss(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]
        loss = torch.mean(diff ** 2)
        return loss
    
    def forward(self, predictions, targets):
        # 1. Classification (Deep Supervision)
        phase_targets = targets['phase_id']
        B, T = phase_targets.shape
        
        stage_outputs = predictions['stage_outputs']
        loss_cls_all = 0.0
        
        for stage_logits in stage_outputs:
            loss_cls_all += self.ce_loss(
                stage_logits.reshape(B * T, -1),
                phase_targets.reshape(B * T)
            )
        loss_cls = loss_cls_all / len(stage_outputs)
        
        # 2. Smoothing
        loss_smooth = 0.0
        for stage_logits in stage_outputs:
            loss_smooth += self.temporal_smoothing_loss(stage_logits)
        loss_smooth /= len(stage_outputs)
        
        # Total
        total = loss_cls + self.gamma * loss_smooth
        
        return {
            'total': total,
            'classification': loss_cls.detach(),
            'smoothing': loss_smooth.detach()
        }

class TrainerB:
    """Training manager for Task B"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._set_seed(self.config.get('seed', 42))
        torch.cuda.empty_cache()
        
        self.save_dir = Path(config['save_dir'])
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        self._build_model()
        self._build_loss()
        self._build_optimizer()
        self._build_dataloaders()
        
        self.start_epoch = 0
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.train_history = []
        self.val_history = []
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _build_model(self):
        # Uses Task B model 
        self.model = MSTCNSurgicalPredictor(
            feature_dim=self.config['feature_dim'],
            hidden_dim=self.config.get('mstcn_channels', 64),
            num_stages=self.config.get('mstcn_stages', 4),
            num_layers=self.config.get('mstcn_layers', 10),
            dropout=self.config['dropout'],
            num_phases=self.config['num_phases'],
            use_external_time_input=self.config.get('use_external_time_input', True)
        )
        self.model = self.model.to(self.device)
        
    def _build_loss(self):
        self.criterion = ClassificationLoss(gamma=self.config['loss_gamma'])
        
    def _build_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max', # Maximize F1/Acc
            factor=0.5,
            patience=self.config['lr_patience'],
            verbose=True
        )

    def _build_dataloaders(self):
        print("\nLoading data...")
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            normalize_features=self.config['normalize_features'],
            normalize_schedule=self.config['normalize_schedule'],
            use_external_time=self.config.get('use_external_time_input', True),
            seed=self.config.get('seed', 42)
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        metrics_calc = MetricsCalculator()
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            targets = {'phase_id': batch['phase_id'].to(self.device)}
            
            predictions = self.model(features)
            loss_dict = self.criterion(predictions, targets)
            
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            epoch_losses.append(loss_dict['total'].item())
            
            metrics_calc.update(predictions, batch)
            
            if (batch_idx + 1) % self.config['print_freq'] == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {np.mean(epoch_losses[-100:]):.4f}")
                
        metrics = metrics_calc.compute()
        elapsed = time.time() - start_time
        return np.mean(epoch_losses), metrics, elapsed

    def validate(self):
        self.model.eval()
        val_losses = []
        metrics_calc = MetricsCalculator()
        all_predictions = [] 
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                targets = {'phase_id': batch['phase_id'].to(self.device)}
                
                predictions = self.model(features)
                loss_dict = self.criterion(predictions, targets)
                val_losses.append(loss_dict['total'].item())
                
                metrics_calc.update(predictions, batch)
                
                all_predictions.append({
                    'phase_logits': predictions['phase_logits'].cpu()
                })
        
        metrics = metrics_calc.compute()
        vis_data = {'predictions': all_predictions} # simplified vis data
        return np.mean(val_losses), metrics, vis_data

    def save_checkpoint(self, epoch, metric_type=None, value=0.0):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config
        }
        if metric_type == 'f1':
            filename = 'checkpoint_best_f1.pth'
            torch.save(checkpoint, self.save_dir / filename)
            print(f"  ★ New Best F1 Model saved to {filename} (F1: {value:.4f})")
            
    def save_visualization_data(self, vis_data):
        torch.save(vis_data, self.save_dir / 'val_predictions_best_f1.pt')

    def train(self):
        print("Starting Task B Training (Classification Only)...")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\nEpoch [{epoch+1}/{self.config['num_epochs']}]")
            
            train_loss, train_metrics, train_time = self.train_epoch(epoch)
            val_loss, val_metrics, vis_data = self.validate()
            
            self.scheduler.step(val_metrics['f1'])
            
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save Best F1
            if val_metrics['f1'] > self.best_f1:
                print(f"  ↗ Improved F1: {val_metrics['f1']:.4f} > {self.best_f1:.4f}")
                self.best_f1 = val_metrics['f1']
                self.save_checkpoint(epoch + 1, 'f1', self.best_f1)
                self.save_visualization_data(vis_data)
                
            self.train_history.append({'epoch': epoch+1, 'loss': train_loss, 'metrics': train_metrics})
            self.val_history.append({'epoch': epoch+1, 'loss': val_loss, 'metrics': val_metrics})
            
            with open(self.save_dir / 'history.json', 'w') as f:
                 def convert(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    return obj
                 json.dump({'train': self.train_history, 'val': self.val_history}, f, default=convert, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train Task B (Pure Classification)')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--exp_name', type=str, default='task_B')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    # Task B specific defaults
    parser.add_argument('--use_external_time_input', type=int, default=1, help='Enable input injection')
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = f'results/models/{args.exp_name}'
        
    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'model_type': 'mstcn',
        'feature_dim': 768,
        'mstcn_channels': 64,
        'mstcn_stages': 4,
        'mstcn_layers': 10,
        'dropout': 0.3,
        'num_phases': 7,
        'use_external_time_input': bool(args.use_external_time_input),
        'loss_gamma': 0.15,
        'num_epochs': args.epochs,
        'batch_size': 1,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'grad_clip': 5.0,
        'lr_patience': 5,
        'num_workers': 0,
        'normalize_features': True,
        'normalize_schedule': True,
        'seed': 42,
        'print_freq': 100
    }
    
    trainer = TrainerB(config)
    trainer.train()

if __name__ == '__main__':
    main()
