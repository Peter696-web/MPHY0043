"""
Evaluation Metrics Module
Simple classification and regression metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    """Simplified metrics calculator"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated data"""
        self.phase_preds = []
        self.phase_targets = []
        self.schedule_preds = []
        self.schedule_targets = []
    
    def update(self, predictions, targets):
        """Update metrics (accumulate one batch)"""
        # Classification (frame-level)
        phase_logits = predictions['phase_logits'].detach().cpu()   # (B, T, num_phases)
        B, T, _ = phase_logits.shape
        phase_pred = torch.argmax(phase_logits, dim=2).reshape(-1).numpy()
        phase_true = targets['phase_id'].detach().cpu().reshape(-1).numpy()
        
        self.phase_preds.extend(phase_pred.tolist())
        self.phase_targets.extend(phase_true.tolist())
        
        # Regression (frame-level, flatten time so shapes align across videos)
        pred_sched = predictions['future_schedule'].detach().cpu().numpy()   # (B, T, 7, 2)
        true_sched = targets['future_schedule'].detach().cpu().numpy()

        self.schedule_preds.append(pred_sched.reshape(-1, pred_sched.shape[-2], pred_sched.shape[-1]))
        self.schedule_targets.append(true_sched.reshape(-1, true_sched.shape[-2], true_sched.shape[-1]))
    
    def compute(self):
        """Compute all metrics"""
        metrics = {}
        
        # Classification metrics
        phase_pred = np.array(self.phase_preds)
        phase_true = np.array(self.phase_targets)
        
        metrics['accuracy'] = accuracy_score(phase_true, phase_pred)
        metrics['precision'] = precision_score(phase_true, phase_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(phase_true, phase_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(phase_true, phase_pred, average='macro', zero_division=0)
        
        # Regression metrics
        pred_sched = np.concatenate(self.schedule_preds, axis=0)
        true_sched = np.concatenate(self.schedule_targets, axis=0)
        
        # Only compute on valid positions
        valid_mask = (true_sched >= 0)
        valid_pred = pred_sched[valid_mask]
        valid_true = true_sched[valid_mask]
        
        # MAE and RMSE
        metrics['mae'] = np.abs(valid_pred - valid_true).mean()
        metrics['rmse'] = np.sqrt(((valid_pred - valid_true) ** 2).mean())
        
        return metrics
    
    def get_summary(self, metrics=None):
        """Get formatted metrics summary"""
        if metrics is None:
            metrics = self.compute()
        
        summary = []
        summary.append("="*70)
        summary.append("Evaluation Metrics")
        summary.append("="*70)
        summary.append("\n[Classification Task]")
        summary.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        summary.append(f"  Precision: {metrics['precision']:.4f}")
        summary.append(f"  Recall:    {metrics['recall']:.4f}")
        summary.append(f"  F1 Score:  {metrics['f1']:.4f}")
        summary.append("\n[Regression Task]")
        summary.append(f"  MAE:  {metrics['mae']:.4f}")
        summary.append(f"  RMSE: {metrics['rmse']:.4f}")
        summary.append("="*70)
        
        return "\n".join(summary)
