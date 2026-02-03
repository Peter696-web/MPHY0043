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
    单个膨胀残差卷积层
    使用膨胀卷积扩大感受野，残差连接稳定训练
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
        return x + out  # 残差连接


class SingleStageModel(nn.Module):
    """
    MS-TCN 的单阶段模型
    由多层膨胀卷积堆叠而成，每层膨胀率翻倍
    """
    
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dropout=0.3):
        """
        Args:
            num_layers: 卷积层数（控制感受野大小）
            num_f_maps: 特征图数量（通道数）
            dim: 输入特征维度
            num_classes: 输出类别数
            dropout: Dropout 比例
        """
        super().__init__()
        
        # 输入投影
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        
        # 膨胀卷积层（膨胀率：1, 2, 4, 8, ...）
        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                num_f_maps, num_f_maps,
                kernel_size=3,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # 输出投影
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C_in, T) 输入特征
        Returns:
            out: (B, num_classes, T) 分类 logits
        """
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MSTCN(nn.Module):
    """
    MS-TCN++: 多阶段时序卷积网络
    
    第一阶段：从原始特征预测
    后续阶段：从前一阶段的 softmax 输出细化预测（coarse-to-fine）
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
            num_stages: 细化阶段数（越多越精细，但计算量大）
            num_layers: 每阶段的卷积层数
            num_f_maps: 特征图数量
            feature_dim: 输入特征维度
            num_classes: 阶段类别数
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # Stage 1: 从原始特征预测
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, feature_dim, num_classes, dropout
        )
        
        # Stage 2-N: 从前一阶段的 softmax 输出细化
        self.stages = nn.ModuleList([
            SingleStageModel(
                num_layers, num_f_maps, num_classes, num_classes, dropout
            )
            for _ in range(num_stages - 1)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, T, feature_dim) 输入特征序列
        Returns:
            outputs: list of (B, T, num_classes)，每个元素对应一个阶段的预测
        """
        # 转为卷积格式：(B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Stage 1
        out = self.stage1(x)
        outputs = [out]
        
        # Stage 2-N (refinement stages)
        for stage in self.stages:
            # 用前一阶段的 softmax 输出作为输入
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        
        # 转回：(B, num_classes, T) -> (B, T, num_classes)
        outputs = [o.permute(0, 2, 1) for o in outputs]
        
        return outputs


class MSTCNSurgicalPredictor(nn.Module):
    """
    MS-TCN++ 用于手术阶段预测（多任务：分类 + 回归）
    
    分类任务：使用 MS-TCN 主干
    回归任务：从最后阶段特征预测 future_schedule
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 hidden_dim: int = 64,
                 num_stages: int = 4,
                 num_layers: int = 10,
                 dropout: float = 0.3,
                 num_phases: int = 7):
        """
        Args:
            feature_dim: 输入特征维度（DINOv2: 768）
            hidden_dim: MS-TCN 内部特征图数量
            num_stages: 细化阶段数
            num_layers: 每阶段卷积层数
            dropout: Dropout 比例
            num_phases: 阶段类别数
        """
        super().__init__()
        
        self.num_phases = num_phases
        self.num_stages = num_stages
        
        # MS-TCN 主干（阶段分类）
        self.mstcn = MSTCN(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=hidden_dim,
            feature_dim=feature_dim,
            num_classes=num_phases,
            dropout=dropout
        )
        
        # 回归分支：从视觉特征 + 阶段概率 + 时间特征预测 future_schedule
        # 方案C+ (阶段感知 + 时间特征) - 恢复此版本，效果更好
        # 输入维度：feature_dim + num_phases + 1 (视觉特征 + 阶段概率 + 全局时间)
        self.regression_branch = nn.Sequential(
            nn.Conv1d(feature_dim + num_phases + 1, hidden_dim, 1),  # 768 + 7 + 1 = 776
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, num_phases * 2, 1)  # 7 phases × 2 values
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播（方案C+：阶段感知 + 时间特征）
        
        核心：回归分支使用阶段概率，帮助理解手术流程和阶段顺序
        
        Args:
            x: (B, T, feature_dim) 输入特征序列
            
        Returns:
            dict with keys:
                - phase_logits: (B, T, num_phases) 最终阶段分类 logits
                - future_schedule: (B, T, num_phases, 2) 未来时间表预测
                - stage_outputs: list of (B, T, num_phases) 各阶段的预测（用于多阶段监督）
        """
        B, T, C = x.shape
        
        # MS-TCN 分类（多阶段输出）
        stage_outputs = self.mstcn(x)  # list of (B, T, num_phases)
        
        # 获取最终阶段的分类概率
        phase_logits = stage_outputs[-1]  # (B, T, num_phases)
        phase_probs = F.softmax(phase_logits, dim=-1)  # (B, T, num_phases)
        
        # 方案C+核心：添加时间特征
        # 1. 全局时间位置: t/T (归一化的时间戳)
        time_pos = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) / T  # (1, T, 1)
        time_pos = time_pos.expand(B, -1, -1)  # (B, T, 1)
        
        # 合并: 视觉特征 + 阶段概率 + 时间位置
        combined_features = torch.cat([x, phase_probs, time_pos], dim=-1)  # (B, T, 768+7+1=776)
        
        # 回归分支（从合并特征预测）
        combined_1d = combined_features.permute(0, 2, 1)  # (B, 776, T)
        schedule_flat = self.regression_branch(combined_1d)  # (B, num_phases*2, T)
        schedule_flat = schedule_flat.permute(0, 2, 1)  # (B, T, num_phases*2)
        future_schedule = schedule_flat.reshape(B, T, self.num_phases, 2)
        future_schedule = torch.relu(future_schedule) + 1e-6
        
        return {
            'phase_logits': phase_logits,           # 最后阶段的预测
            'future_schedule': future_schedule,
            'stage_outputs': stage_outputs          # 用于计算多阶段损失
        }


# ============================================================================
# 测试代码
# ============================================================================

def test_mstcn():
    """测试 MS-TCN 模型"""
    print("="*70)
    print("测试 MS-TCN++ 模型")
    print("="*70)
    
    # 模拟输入
    batch_size = 2
    seq_len = 500  # 一个视频约 500-3000 帧
    feature_dim = 768
    num_phases = 7
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # 创建模型
    model = MSTCNSurgicalPredictor(
        feature_dim=feature_dim,
        hidden_dim=64,
        num_stages=4,
        num_layers=10,
        num_phases=num_phases
    )
    
    # 前向传播
    outputs = model(x)
    
    print(f"\n输入: {x.shape}")
    print(f"阶段分类 logits: {outputs['phase_logits'].shape}")
    print(f"未来时间表: {outputs['future_schedule'].shape}")
    print(f"所有阶段预测: {[o.shape for o in outputs['stage_outputs']]}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试感受野
    print(f"\n理论感受野: ~{2**(10+1) * 3} 帧（约 {2**(10+1)*3/25:.1f} 秒 @ 25fps）")
    
    print("="*70)


if __name__ == '__main__':
    test_mstcn()
