"""
模型架构模块
多任务学习模型：同时预测阶段分类和未来时间表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SurgicalPhasePredictor(nn.Module):
    """
    手术阶段预测模型 (基础版本 - MLP)
    
    输入: DINOv2 特征 (batch, 768)
    
    输出:
        - phase_logits: (batch, 7) - 当前阶段分类
        - future_schedule: (batch, 7, 2) - 所有阶段的时间表
            [:, phase_id, 0] = start_offset (距当前帧多少秒后开始)
            [:, phase_id, 1] = duration (该阶段持续多少秒)
    """
    
    def __init__(self, 
                 feature_dim: int = 768,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 num_phases: int = 7):
        """
        初始化模型
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            num_phases: 阶段数量
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        # 共享特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 任务头1: 当前阶段分类
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_phases)
        )
        
        # 任务头2: 未来阶段时间表预测
        self.schedule_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_phases * 2)  # 7个阶段 × 2个值 (start_offset, duration)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, feature_dim) - DINOv2 特征
            
        Returns:
            dict with keys:
                - phase_logits: (batch, 7) - 阶段分类 logits
                - future_schedule: (batch, 7, 2) - 时间表预测
        """
        # 特征编码
        features = self.encoder(x)  # (batch, hidden_dim)
        
        # 输出1: 阶段分类
        phase_logits = self.phase_classifier(features)  # (batch, 7)
        
        # 输出2: 时间表预测
        schedule_flat = self.schedule_predictor(features)  # (batch, 14)
        future_schedule = schedule_flat.view(-1, self.num_phases, 2)  # (batch, 7, 2)
        
        # 对时间值应用激活函数，确保非负
        # start_offset: 可以是0或正数，用 ReLU
        # duration: 必须是正数，用 ReLU + 小偏移
        future_schedule = torch.relu(future_schedule) + 1e-6
        
        return {
            'phase_logits': phase_logits,
            'future_schedule': future_schedule
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推理模式：返回预测结果
        
        Args:
            x: (batch, feature_dim)
            
        Returns:
            dict with predicted values
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            # 获取预测的阶段
            phase_probs = F.softmax(outputs['phase_logits'], dim=1)
            predicted_phase = torch.argmax(phase_probs, dim=1)
            
            # 获取当前阶段剩余时间
            batch_size = x.size(0)
            batch_indices = torch.arange(batch_size, device=x.device)
            current_remaining = outputs['future_schedule'][batch_indices, predicted_phase, 1]
            
            return {
                'predicted_phase': predicted_phase,
                'phase_probs': phase_probs,
                'current_remaining': current_remaining,
                'future_schedule': outputs['future_schedule']
            }


class LSTMSurgicalPredictor(nn.Module):
    """
    基于LSTM的手术阶段预测模型（序列建模）
    
    考虑时间上下文信息，适合视频序列预测
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_phases: int = 7,
                 bidirectional: bool = False):
        """
        初始化LSTM模型
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            num_phases: 阶段数量
            bidirectional: 是否使用双向LSTM
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_phases = num_phases
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 输出头
        lstm_output_dim = hidden_dim * self.num_directions
        
        # 阶段分类头
        self.phase_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_phases)
        )
        
        # 时间表预测头
        self.schedule_predictor = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_phases * 2)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, feature_dim) 或 (batch, feature_dim)
            
        Returns:
            dict with predictions
        """
        # 如果是2D输入，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * num_directions)
        
        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim * num_directions)
        
        # 分类和回归
        phase_logits = self.phase_classifier(last_hidden)
        schedule_flat = self.schedule_predictor(last_hidden)
        future_schedule = schedule_flat.view(-1, self.num_phases, 2)
        future_schedule = torch.relu(future_schedule) + 1e-6
        
        return {
            'phase_logits': phase_logits,
            'future_schedule': future_schedule
        }


class TransformerSurgicalPredictor(nn.Module):
    """
    基于Transformer的手术阶段预测模型
    
    使用自注意力机制捕获长距离依赖
    """
    
    def __init__(self,
                 feature_dim: int = 768,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.3,
                 num_phases: int = 7):
        """
        初始化Transformer模型
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
            num_phases: 阶段数量
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        # 输入投影
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出头
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_phases)
        )
        
        self.schedule_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_phases * 2)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, feature_dim) 或 (batch, feature_dim)
            
        Returns:
            dict with predictions
        """
        # 如果是2D输入，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Transformer编码
        transformer_out = self.transformer(x)  # (batch, seq_len, hidden_dim)
        
        # 取最后一个时间步（或全局平均池化）
        # last_hidden = transformer_out[:, -1, :]  # 方式1
        last_hidden = transformer_out.mean(dim=1)  # 方式2: 平均池化
        
        # 分类和回归
        phase_logits = self.phase_classifier(last_hidden)
        schedule_flat = self.schedule_predictor(last_hidden)
        future_schedule = schedule_flat.view(-1, self.num_phases, 2)
        future_schedule = torch.relu(future_schedule) + 1e-6
        
        return {
            'phase_logits': phase_logits,
            'future_schedule': future_schedule
        }


def create_model(model_type: str = 'mlp', **kwargs) -> nn.Module:
    """
    模型工厂函数
    
    Args:
        model_type: 'mlp', 'lstm', 'transformer'
        **kwargs: 模型参数
        
    Returns:
        model: 创建的模型
    """
    if model_type == 'mlp':
        return SurgicalPhasePredictor(**kwargs)
    elif model_type == 'lstm':
        return LSTMSurgicalPredictor(**kwargs)
    elif model_type == 'transformer':
        return TransformerSurgicalPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# 使用示例
# ============================================================================

def test_models():
    """测试不同模型架构"""
    batch_size = 16
    feature_dim = 768
    
    print("="*70)
    print("测试模型架构")
    print("="*70)
    
    # 测试MLP模型
    print("\n1. MLP模型:")
    mlp_model = SurgicalPhasePredictor(feature_dim=feature_dim, hidden_dim=256)
    x = torch.randn(batch_size, feature_dim)
    outputs = mlp_model(x)
    print(f"  输入: {x.shape}")
    print(f"  phase_logits: {outputs['phase_logits'].shape}")
    print(f"  future_schedule: {outputs['future_schedule'].shape}")
    print(f"  参数量: {sum(p.numel() for p in mlp_model.parameters()):,}")
    
    # 测试LSTM模型
    print("\n2. LSTM模型:")
    lstm_model = LSTMSurgicalPredictor(feature_dim=feature_dim, hidden_dim=256)
    outputs = lstm_model(x)
    print(f"  输入: {x.shape}")
    print(f"  phase_logits: {outputs['phase_logits'].shape}")
    print(f"  future_schedule: {outputs['future_schedule'].shape}")
    print(f"  参数量: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    # 测试Transformer模型
    print("\n3. Transformer模型:")
    transformer_model = TransformerSurgicalPredictor(feature_dim=feature_dim, hidden_dim=256)
    outputs = transformer_model(x)
    print(f"  输入: {x.shape}")
    print(f"  phase_logits: {outputs['phase_logits'].shape}")
    print(f"  future_schedule: {outputs['future_schedule'].shape}")
    print(f"  参数量: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # 测试预测函数
    print("\n4. 测试预测:")
    predictions = mlp_model.predict(x)
    print(f"  predicted_phase: {predictions['predicted_phase'].shape}")
    print(f"  phase_probs: {predictions['phase_probs'].shape}")
    print(f"  current_remaining: {predictions['current_remaining'].shape}")
    print(f"  示例预测: phase={predictions['predicted_phase'][0].item()}, "
          f"remaining={predictions['current_remaining'][0].item():.2f}")
    
    print("="*70)


if __name__ == '__main__':
    test_models()
