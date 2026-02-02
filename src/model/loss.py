"""
Loss Function Module
Simple multi-task loss: Classification (CrossEntropy) + Regression (MSE)
"""

import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Simplified multi-task loss function
    
    Loss =  CrossEntropy(phase) +  MSE(future_schedule)
    
    Task 1: Current phase classification
    Task 2: Future phase schedule prediction (start_offset, duration)
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Args:
            alpha: Weight for classification task
            beta: Weight for regression task
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Compute multi-task loss
        
        Args:
            predictions: dict with 'phase_logits', 'future_schedule'
            targets: dict with 'phase_id', 'future_schedule'
        
        Returns:
            dict with 'total', 'classification', 'regression'
        """
        # reshape to (B*T, ...)
        phase_logits = predictions['phase_logits']   # (B, T, num_phases)
        phase_targets = targets['phase_id']          # (B, T)
        B, T, _ = phase_logits.shape

        loss_cls = self.ce_loss(
            phase_logits.reshape(B * T, -1),
            phase_targets.reshape(B * T)
        )
        
        # Regression loss (only compute on valid positions)
        pred_sched = predictions['future_schedule']           # (B, T, num_phases, 2)
        true_sched = targets['future_schedule']               # (B, T, num_phases, 2)
        valid_mask = (true_sched >= 0).float()
        
        mse = ((pred_sched - true_sched.clamp(min=0)) ** 2) * valid_mask
        loss_reg = mse.sum() / (valid_mask.sum() + 1e-8)
        
        # Total loss with task weights
        total = self.alpha * loss_cls + self.beta * loss_reg
        
        return {
            'total': total,
            'classification': loss_cls.detach(),
            'regression': loss_reg.detach()
        }
