"""
Loss Function Module
Multi-task loss: Classification (CrossEntropy) + Regression (MSE) + Smoot        # Total loss with task weights
        total = self.alpha * loss_cls + self.beta * loss_reg
        if self.use_mstcn:
            total += self.gamma * loss_smooth
        
        return {
            'total': total,
            'classification': loss_cls,
            'regression': loss_reg,
            'smoothing': torch.tensor(loss_smooth) if isinstance(loss_smooth, float) else loss_smooth
        })
Smoothing loss for MS-TCN: penalize rapid phase transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with temporal smoothing
    
    Loss = alpha * CrossEntropy(phase) + beta * MSE(future_schedule) + gamma * TMSE(phase)
    
    Task 1: Current phase classification
    Task 2: Future phase schedule prediction (start_offset, duration)
    Task 3: Temporal smoothing (MS-TCN standard)
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.15, use_mstcn=False):
        """
        Args:
            alpha: Weight for classification task
            beta: Weight for regression task
            gamma: Weight for temporal smoothing loss (MS-TCN only)
            use_mstcn: Whether using MS-TCN (enables multi-stage loss)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_mstcn = use_mstcn
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def temporal_smoothing_loss(self, logits):
        """
        Temporal smoothing loss (TMSE): penalize rapid transitions
        Computes MSE between adjacent frame predictions
        
        Args:
            logits: (B, T, num_classes) frame-wise logits
        Returns:
            loss: scalar
        """
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)
        
        # MSE between adjacent frames
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]  # (B, T-1, C)
        loss = torch.mean(diff ** 2)
        
        return loss
    
    def forward(self, predictions, targets):
        """
        Compute multi-task loss
        
        Args:
            predictions: dict with:
                - 'phase_logits': (B, T, num_phases) final predictions
                - 'future_schedule': (B, T, num_phases, 2)
                - 'stage_outputs': list of (B, T, num_phases) for MS-TCN
            targets: dict with 'phase_id', 'future_schedule'
        
        Returns:
            dict with 'total', 'classification', 'regression', 'smoothing'
        """
        # 1. Classification loss
        phase_logits = predictions['phase_logits']   # (B, T, num_phases)
        phase_targets = targets['phase_id']          # (B, T)
        B, T, _ = phase_logits.shape

        loss_cls = self.ce_loss(
            phase_logits.reshape(B * T, -1),
            phase_targets.reshape(B * T)
        )
        
        # 2. Regression loss (only compute on valid positions)
        pred_sched = predictions['future_schedule']           # (B, T, num_phases, 2)
        true_sched = targets['future_schedule']               # (B, T, num_phases, 2)
        valid_mask = (true_sched >= 0).float()
        
        mse = ((pred_sched - true_sched.clamp(min=0)) ** 2) * valid_mask
        loss_reg = mse.sum() / (valid_mask.sum() + 1e-8)
        
        # 3. Temporal smoothing loss (MS-TCN)
        loss_smooth = 0.0
        if self.use_mstcn and 'stage_outputs' in predictions:
            # Apply smoothing loss to all stages
            for stage_logits in predictions['stage_outputs']:
                loss_smooth += self.temporal_smoothing_loss(stage_logits)
            loss_smooth /= len(predictions['stage_outputs'])
        
        # Total loss with task weights
        total = self.alpha * loss_cls + self.beta * loss_reg
        if self.use_mstcn:
            total += self.gamma * loss_smooth
        
        return {
            'total': total,
            'classification': loss_cls.detach(),
            'regression': loss_reg.detach()
        }
