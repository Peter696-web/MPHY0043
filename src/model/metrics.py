"""
Evaluation Metrics Module
Simplified: Only Accuracy, F1, MSE
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class MetricsCalculator:
    """Simplified metrics calculator - only essential metrics"""
    
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
        
        # Regression (frame-level)
        pred_sched = predictions['future_schedule'].detach().cpu().numpy()   # (B, T, 7, 2)
        true_sched = targets['future_schedule'].detach().cpu().numpy()

        self.schedule_preds.append(pred_sched.reshape(-1, pred_sched.shape[-2], pred_sched.shape[-1]))
        self.schedule_targets.append(true_sched.reshape(-1, true_sched.shape[-2], true_sched.shape[-1]))
    
    def compute(self):
        """Compute essential metrics only: Accuracy, F1, MSE"""
        metrics = {}
        
        # Classification metrics
        phase_pred = np.array(self.phase_preds)
        phase_true = np.array(self.phase_targets)
        
        metrics['accuracy'] = float(accuracy_score(phase_true, phase_pred))
        metrics['f1'] = float(f1_score(phase_true, phase_pred, average='macro', zero_division=0))
        
        # Regression metrics - MSE only
        pred_sched = np.concatenate(self.schedule_preds, axis=0)  # (N, 7, 2)
        true_sched = np.concatenate(self.schedule_targets, axis=0)
        
        # Valid mask
        valid_mask = (true_sched >= 0)
        
        if valid_mask.sum() > 0:
            mse = np.mean((pred_sched[valid_mask] - true_sched[valid_mask]) ** 2)
        else:
            mse = 0.0
        
        metrics['mse'] = float(mse)
        
        return metrics
