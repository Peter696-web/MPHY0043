"""
MS-TCN++ Model for Surgical Phase Recognition
Multi-Stage Temporal Convolutional Network with refinement stages
Based on: MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation (CVPR 2019)
and MS-TCN++: Improving Multi-Stage Temporal Convolutional Network (arXiv 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class DilatedResidualLayer(nn.Module):
    """
    Single dilated residual convolutional layer
    Uses dilated convolution to expand receptive field, residual connection for stable training
    """
    
    def __init__(self, d_in, d_out, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv_dilated = nn.Conv1d(
            d_in, d_out, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(d_out, d_out, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out  # Residual connection


class SingleStageModel(nn.Module):
    """
    MS-TCN single stage model
    Composed of multiple stacked dilated convolutional layers with doubling dilation rates
    """
    
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout=0.3):
        """
        Args:
            num_layers: Number of convolutional layers (controls receptive field size)
            num_f_maps: Number of feature maps (channels)
            dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout ratio
        """
        super().__init__()
        
        # Input projection
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        
        # Dilated convolutional layers (dilation rates: 1, 2, 4, 8, ...)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                num_f_maps, num_f_maps,
                kernel_size=3,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C_in, T) Input features
        Returns:
            out: (B, num_classes, T) Classification logits
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MSTCN(nn.Module):
    """
    MS-TCN++: Multi-Stage Temporal Convolutional Network
    
    Stage 1: Predict from original features
    Subsequent stages: Refine predictions from previous stage's softmax output (coarse-to-fine)
    """
    
    def __init__(self,
                 num_stages=4,
                 num_layers=10,
                 num_f_maps=64,
                 feature_dim=768,
                 num_classes=7,
                 dropout=0.3):
        """
        Args:
            num_stages: Number of refinement stages (more stages = finer but more computation)
            num_layers: Number of convolutional layers per stage
            num_f_maps: Number of feature maps
            feature_dim: Input feature dimension
            num_classes: Number of phase classes
            dropout: Dropout ratio
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # Stage 1: Predict from original features
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, feature_dim, num_classes, dropout
        )
        
        # Stage 2-N: Refine from previous stage's softmax output
        self.stages = nn.ModuleList([
            SingleStageModel(
                num_layers, num_f_maps, num_classes, num_classes, dropout
            )
            for _ in range(num_stages - 1)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, T, feature_dim) Input feature sequence
        Returns:
            outputs: list of (B, T, num_classes), each element corresponds to one stage's prediction
        """
        # Convert to conv format: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Stage 1
        out = self.stage1(x)
        outputs = [out]
        
        # Stage 2-N (refinement stages)
        for stage in self.stages:
            # Use previous stage's softmax output as input
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        
        # Convert back: (B, num_classes, T) -> (B, T, num_classes)
        outputs = [o.permute(0, 2, 1) for o in outputs]
        
        return outputs


class MSTCNSurgicalPredictor(nn.Module):
    """
    MS-TCN++ for surgical phase prediction (multi-task: classification + regression)
    
    Classification task: Use MS-TCN backbone
    Regression task: Predict future_schedule from final stage features
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 hidden_dim: int = 64,
                 num_stages: int = 4,
                 num_layers: int = 10,
                 dropout: float = 0.3,
                 num_phases: int = 7,
                 use_external_time_input: bool = False):
        """
        Args:
            feature_dim: Input feature dimension (DINOv2: 768)
            hidden_dim: MS-TCN internal feature map count
            num_stages: Number of refinement stages
            num_layers: Number of convolutional layers per stage
            dropout: Dropout ratio
            num_phases: Number of phase classes
            use_external_time_input: If True, input features will include external time prediction
        """
        super().__init__()
        
        self.num_phases = num_phases
        self.num_stages = num_stages
        
        # Adjust input dimension if using external time input (Task B)
        stage1_input_dim = feature_dim + (num_phases * 2 if use_external_time_input else 0)
        
        # MS-TCN backbone (phase classification)
        self.mstcn = MSTCN(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=hidden_dim,
            feature_dim=stage1_input_dim,
            num_classes=num_phases,
            dropout=dropout
        )
        
        # Classification-only Task B:
        # The external time input (if enabled) is handled via stage1_input_dim above.
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass (Task B: Classification only)
        
        Args:
            x: (B, T, feature_dim) Input feature sequence
               If use_external_time_input=True, feature_dim includes the time features.
            
        Returns:
            dict with keys:
                - phase_logits: (B, T, num_phases) Final stage phase classification logits
                - stage_outputs: list of (B, T, num_phases) Predictions from each stage
        """
        # MS-TCN classification (multi-stage outputs)
        stage_outputs = self.mstcn(x)  # list of (B, T, num_phases)
        
        # Get final stage classification probabilities
        phase_logits = stage_outputs[-1]  # (B, T, num_phases)
        
        # No regression branch
        
        return {
            'phase_logits': phase_logits,           # Final stage predictions
            'stage_outputs': stage_outputs          # For multi-stage loss computation
        }


