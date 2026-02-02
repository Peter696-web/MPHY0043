"""
可视化训练历史和预测结果
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_path):
    """绘制训练历史：loss曲线"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_epochs = [item['epoch'] for item in history['train']]
    train_losses = [item['losses']['total'] for item in history['train']]

    val_epochs = [item['epoch'] for item in history['val']]
    val_losses = [item['losses']['total'] for item in history['val']]

    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_losses, 'b-', label='训练集', linewidth=2)
    plt.plot(val_epochs, val_losses, 'r-', label='验证集', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 放大后半部分
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs[5:], train_losses[5:], 'b-', label='训练集', linewidth=2)
    plt.plot(val_epochs[5:], val_losses[5:], 'r-', label='验证集', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('损失曲线（Epoch 6-14）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_predictions(predictions_path, max_videos=5):
    """绘制时间预测对比图"""
    pred_data = torch.load(predictions_path)

    n_videos = len(pred_data['predictions'])
    n_plot = min(max_videos, n_videos)

    fig, axes = plt.subplots(n_plot, 1, figsize=(15, 4*n_plot))
    if n_plot == 1:
        axes = [axes]

    phase_names = ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6']

    for i in range(n_plot):
        predictions = pred_data['predictions'][i]
        targets = pred_data['targets'][i]
        video_id = pred_data['video_ids'][i].item()

        # 获取预测和真实的时间表
        pred_sched = predictions['future_schedule'][0].numpy()  # (7, 2)
        true_sched = targets['future_schedule'][0].numpy()  # (7, 2)

        ax = axes[i]

        # 为每个阶段绘制预测vs真实
        x = np.arange(len(phase_names))
        width = 0.35

        # 预测值
        pred_starts = pred_sched[:, 0]
        pred_durations = pred_sched[:, 1]

        # 真实值
        true_starts = true_sched[:, 0]
        true_durations = true_sched[:, 1]

        # 绘制起始时间
        ax.bar(x - width/2, pred_starts, width, label='预测起始时间', alpha=0.7, color='blue')
        ax.bar(x - width/2, pred_durations, width, bottom=pred_starts,
               label='预测持续时间', alpha=0.7, color='lightblue')

        ax.bar(x + width/2, true_starts, width, label='真实起始时间', alpha=0.7, color='red')
        ax.bar(x + width/2, true_durations, width, bottom=true_starts,
               label='真实持续时间', alpha=0.7, color='pink')

        ax.set_xlabel('手术阶段')
        ax.set_ylabel('归一化时间')
        ax.set_title(f'视频 {video_id} - 手术阶段时间预测对比')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/time_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_errors(predictions_path):
    """绘制时间预测误差分布"""
    pred_data = torch.load(predictions_path)

    all_pred_starts = []
    all_true_starts = []
    all_pred_durations = []
    all_true_durations = []

    for i in range(len(pred_data['predictions'])):
        predictions = pred_data['predictions'][i]
        targets = pred_data['targets'][i]

        pred_sched = predictions['future_schedule'][0].numpy()
        true_sched = targets['future_schedule'][0].numpy()

        # 只收集有效预测（非负值）
        valid_mask = true_sched[:, 0] >= 0
        all_pred_starts.extend(pred_sched[valid_mask, 0])
        all_true_starts.extend(true_sched[valid_mask, 0])
        all_pred_durations.extend(pred_sched[valid_mask, 1])
        all_true_durations.extend(true_sched[valid_mask, 1])

    # 计算误差
    start_errors = np.array(all_pred_starts) - np.array(all_true_starts)
    duration_errors = np.array(all_pred_durations) - np.array(all_true_durations)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 起始时间误差
    axes[0].hist(start_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(start_errors), color='red', linestyle='--',
                   label=f'均值: {np.mean(start_errors):.4f}')
    axes[0].set_xlabel('预测误差')
    axes[0].set_ylabel('频次')
    axes[0].set_title('起始时间预测误差分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 持续时间误差
    axes[1].hist(duration_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(duration_errors), color='red', linestyle='--',
                   label=f'均值: {np.mean(duration_errors):.4f}')
    axes[1].set_xlabel('预测误差')
    axes[1].set_ylabel('频次')
    axes[1].set_title('持续时间预测误差分布')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/time_errors.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 创建figures目录
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    # 1. 绘制训练历史
    print("绘制训练损失曲线...")
    plot_training_history('results/models/lstm_b1_norm/history.json')

    # 2. 绘制时间预测对比
    print("绘制时间预测对比...")
    plot_time_predictions('results/models/lstm_b1_norm/val_predictions_epoch001.pt')

    # 3. 绘制时间预测误差
    print("绘制时间预测误差分布...")
    plot_time_errors('results/models/lstm_b1_norm/val_predictions_epoch001.pt')

    print("可视化完成！图片保存至 results/figures/")