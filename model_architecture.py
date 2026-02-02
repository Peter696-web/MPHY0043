"""
多任务学习模型：同时预测当前阶段剩余时间 + 所有未来阶段时间表

任务1: 预测当前阶段的剩余时间
任务2: 预测所有未来阶段的起始和结束时间

注意：这两个任务共享 future_schedule 作为标签！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurgicalPhasePredictor(nn.Module):
    """
    手术阶段预测模型
    
    输入: DINOv2 特征 (batch, 768)
    
    输出:
        - phase_logits: (batch, 7) - 当前阶段分类
        - future_schedule: (batch, 7, 2) - 所有阶段的时间表
            [:, phase_id, 0] = start_offset (距当前帧多少秒后开始)
            [:, phase_id, 1] = duration (该阶段持续多少秒)
    
    任务1 (当前阶段剩余时间):
        从 future_schedule[batch_idx, predicted_phase_id, 1] 中提取
    
    任务2 (所有未来阶段时间表):
        直接使用 future_schedule 计算每个阶段的 start_time 和 end_time
    """
    
    def __init__(self, feature_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        
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
            nn.Linear(128, 7)
        )
        
        # 任务头2: 未来阶段时间表预测
        self.schedule_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7 * 2)  # 7个阶段 × 2个值 (start_offset, duration)
        )
    
    def forward(self, x):
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
        future_schedule = schedule_flat.view(-1, 7, 2)     # (batch, 7, 2)
        
        # 对时间值应用激活函数，确保非负
        # start_offset: 可以是0或正数，用 ReLU
        # duration: 必须是正数，用 ReLU + 小偏移
        future_schedule = torch.relu(future_schedule) + 1e-6
        
        return {
            'phase_logits': phase_logits,
            'future_schedule': future_schedule
        }
    
    def predict_current_phase_remaining(self, x):
        """
        任务1: 预测当前阶段的剩余时间
        
        Args:
            x: (batch, feature_dim)
            
        Returns:
            current_remaining: (batch,) - 当前阶段剩余时间
        """
        outputs = self.forward(x)
        
        # 获取预测的阶段ID
        phase_probs = F.softmax(outputs['phase_logits'], dim=1)
        predicted_phase = torch.argmax(phase_probs, dim=1)  # (batch,)
        
        # 从 future_schedule 中提取当前阶段的 duration
        batch_size = x.size(0)
        batch_indices = torch.arange(batch_size, device=x.device)
        current_remaining = outputs['future_schedule'][batch_indices, predicted_phase, 1]
        
        return current_remaining
    
    def predict_future_phases_timeline(self, x, return_absolute_time=False, current_frame=None):
        """
        任务2: 预测所有未来阶段的起止时间
        
        Args:
            x: (batch, feature_dim)
            return_absolute_time: 是否返回绝对时间（需要提供 current_frame）
            current_frame: (batch,) - 当前帧索引，用于计算绝对时间
            
        Returns:
            timeline: dict with keys for each phase:
                - start_time: (batch,) or (batch, 7)
                - end_time: (batch,) or (batch, 7)
                - duration: (batch,) or (batch, 7)
        """
        outputs = self.forward(x)
        schedule = outputs['future_schedule']  # (batch, 7, 2)
        
        # 提取 start_offset 和 duration
        start_offsets = schedule[:, :, 0]  # (batch, 7)
        durations = schedule[:, :, 1]      # (batch, 7)
        
        if return_absolute_time and current_frame is not None:
            # 计算绝对时间
            current_frame = current_frame.unsqueeze(1)  # (batch, 1)
            start_times = current_frame + start_offsets  # (batch, 7)
            end_times = start_times + durations          # (batch, 7)
        else:
            # 返回相对时间
            start_times = start_offsets
            end_times = start_offsets + durations
        
        return {
            'start_times': start_times,    # (batch, 7)
            'end_times': end_times,        # (batch, 7)
            'durations': durations,        # (batch, 7)
            'start_offsets': start_offsets # (batch, 7)
        }


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    Loss = α * CrossEntropy(phase_id) + β * MSE(future_schedule)
    """
    
    def __init__(self, alpha=1.0, beta=1.0, use_log_transform=True):
        super().__init__()
        self.alpha = alpha  # 分类任务权重
        self.beta = beta    # 回归任务权重
        self.use_log_transform = use_log_transform
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        计算多任务损失
        
        Args:
            predictions: dict with 'phase_logits' and 'future_schedule'
            targets: dict with 'phase_id' and 'future_schedule'
            
        Returns:
            dict with loss components
        """
        # 任务1: 阶段分类损失
        loss_classification = self.ce_loss(
            predictions['phase_logits'],
            targets['phase_id']
        )
        
        # 任务2: 时间表预测损失
        pred_schedule = predictions['future_schedule']  # (batch, 7, 2)
        true_schedule = targets['future_schedule']      # (batch, 7, 2)
        
        # 创建mask：忽略已完成的阶段（标记为-1）
        valid_mask = (true_schedule >= 0).float()  # (batch, 7, 2)
        
        # 对时间值应用log变换（可选），避免大数值主导
        if self.use_log_transform:
            pred_schedule_transformed = torch.log1p(pred_schedule)
            true_schedule_transformed = torch.log1p(torch.clamp(true_schedule, min=0))
        else:
            pred_schedule_transformed = pred_schedule
            true_schedule_transformed = torch.clamp(true_schedule, min=0)
        
        # 计算MSE损失（只在有效位置）
        mse_per_element = self.mse_loss(
            pred_schedule_transformed,
            true_schedule_transformed
        )  # (batch, 7, 2)
        
        # 应用mask并求平均
        masked_mse = mse_per_element * valid_mask
        num_valid = valid_mask.sum() + 1e-8
        loss_schedule = masked_mse.sum() / num_valid
        
        # 总损失
        total_loss = self.alpha * loss_classification + self.beta * loss_schedule
        
        return {
            'total': total_loss,
            'classification': loss_classification,
            'schedule': loss_schedule,
            'num_valid_predictions': num_valid
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """示例：如何使用模型"""
    
    # 1. 创建模型
    model = SurgicalPhasePredictor(feature_dim=768, hidden_dim=256)
    criterion = MultiTaskLoss(alpha=1.0, beta=1.0, use_log_transform=True)
    
    # 2. 模拟输入数据
    batch_size = 16
    features = torch.randn(batch_size, 768)  # DINOv2 特征
    
    # 模拟标签
    targets = {
        'phase_id': torch.randint(0, 7, (batch_size,)),
        'future_schedule': torch.randn(batch_size, 7, 2).abs()  # 确保非负
    }
    
    # 3. 前向传播
    predictions = model(features)
    
    print("="*70)
    print("模型输出:")
    print(f"  - phase_logits: {predictions['phase_logits'].shape}")
    print(f"  - future_schedule: {predictions['future_schedule'].shape}")
    
    # 4. 任务1: 预测当前阶段剩余时间
    current_remaining = model.predict_current_phase_remaining(features)
    print(f"\n任务1 - 当前阶段剩余时间:")
    print(f"  - 预测值: {current_remaining[:5].detach().numpy()}")
    
    # 5. 任务2: 预测未来阶段时间表
    timeline = model.predict_future_phases_timeline(features)
    print(f"\n任务2 - 未来阶段时间表:")
    print(f"  - start_offsets: {timeline['start_offsets'].shape}")
    print(f"  - end_times: {timeline['end_times'].shape}")
    print(f"  - durations: {timeline['durations'].shape}")
    print(f"\n  示例（第1个样本的前3个阶段）:")
    for i in range(3):
        print(f"    Phase {i}: start={timeline['start_offsets'][0, i]:.1f}, "
              f"end={timeline['end_times'][0, i]:.1f}, "
              f"duration={timeline['durations'][0, i]:.1f}")
    
    # 6. 计算损失
    losses = criterion(predictions, targets)
    print(f"\n损失:")
    print(f"  - total: {losses['total']:.4f}")
    print(f"  - classification: {losses['classification']:.4f}")
    print(f"  - schedule: {losses['schedule']:.4f}")
    
    print("="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("多任务学习模型架构")
    print("="*70 + "\n")
    
    example_usage()
    
    print("\n模型说明:")
    print("  1. 输入: DINOv2 特征 (768维)")
    print("  2. 输出1: 当前阶段分类 (7类)")
    print("  3. 输出2: 未来阶段时间表 (7×2矩阵)")
    print("\n任务1: 从输出2中提取 future_schedule[predicted_phase, 1]")
    print("任务2: 使用输出2计算所有阶段的 start_time 和 end_time")
    print("\n关键优势: 一个标签(future_schedule)支持两个任务！")
