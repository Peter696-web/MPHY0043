"""
可视化简化标签结构
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 阶段名称和颜色
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction"
]

PHASE_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#FFA07A',  # 橙色
    '#98D8C8',  # 绿色
    '#F7DC6F',  # 黄色
    '#BB8FCE'   # 紫色
]


def visualize_future_schedule(video_id='video01', sample_frames=None):
    """
    可视化未来阶段时间表
    """
    # 加载数据
    labels = np.load(f'data/preprocessed/aligned_labels/{video_id}_labels.npy', 
                     allow_pickle=True).item()
    
    with open(f'data/preprocessed/aligned_labels/{video_id}.json', 'r') as f:
        metadata = json.load(f)
    
    phase_ids = labels['phase_id']
    schedule = labels['future_schedule']
    num_frames = len(phase_ids)
    
    # 选择采样帧（如果未指定，均匀采样5帧）
    if sample_frames is None:
        sample_frames = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    
    # 创建图表
    fig, axes = plt.subplots(len(sample_frames), 1, figsize=(14, 3*len(sample_frames)))
    if len(sample_frames) == 1:
        axes = [axes]
    
    for idx, frame_idx in enumerate(sample_frames):
        ax = axes[idx]
        
        current_phase = phase_ids[frame_idx]
        current_schedule = schedule[frame_idx]
        
        # 绘制每个阶段的时间条
        y_pos = 0
        labels_text = []
        
        for phase_id in range(7):
            start_offset, duration = current_schedule[phase_id]
            
            if start_offset == -1:
                # 已完成：灰色短条
                ax.barh(y_pos, 50, left=0, height=0.8, 
                       color='lightgray', alpha=0.5, edgecolor='black', linewidth=0.5)
                label = f"{PHASE_NAMES[phase_id][:20]:20s} | 已完成"
            elif start_offset == 0:
                # 进行中：高亮颜色
                ax.barh(y_pos, duration, left=frame_idx, height=0.8,
                       color=PHASE_COLORS[phase_id], alpha=0.9, edgecolor='black', linewidth=1)
                label = f"{PHASE_NAMES[phase_id][:20]:20s} | 剩余 {duration:.0f}秒"
            else:
                # 未开始：淡色
                ax.barh(y_pos, duration, left=frame_idx + start_offset, height=0.8,
                       color=PHASE_COLORS[phase_id], alpha=0.4, edgecolor='black', linewidth=0.5)
                label = f"{PHASE_NAMES[phase_id][:20]:20s} | {start_offset:.0f}秒后开始，持续{duration:.0f}秒"
            
            labels_text.append(label)
            y_pos += 1
        
        # 标记当前时刻
        ax.axvline(x=frame_idx, color='red', linestyle='--', linewidth=2, label='当前帧')
        
        # 设置坐标轴
        ax.set_yticks(range(7))
        ax.set_yticklabels(labels_text, fontsize=9)
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_title(f'Frame {frame_idx} - 当前阶段: {PHASE_NAMES[current_phase]}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(0, num_frames)
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path('visuals')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{video_id}_future_schedule.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化已保存: {output_path}")
    
    plt.show()


def compare_label_complexity():
    """
    对比标签复杂度
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 旧版标签（11种）
    old_labels = [
        'current_phase_id',
        'current_phase_remaining',
        'current_phase_progress',
        'phase_remaining_times',
        'phase_start_relative',
        'total_remaining',
        'surgery_progress',
        'elapsed_time',
        'current_phase_remaining_norm',
        'total_remaining_norm',
        '(未命名)'
    ]
    
    old_dims = [1, 1, 1, 7, 7, 1, 1, 1, 1, 1, 1]
    
    # 新版标签（2种）
    new_labels = ['phase_id', 'future_schedule']
    new_dims = [1, 14]  # 7×2=14
    
    # 绘制旧版
    bars1 = ax1.barh(range(len(old_labels)), old_dims, color='coral', alpha=0.7)
    ax1.set_yticks(range(len(old_labels)))
    ax1.set_yticklabels(old_labels, fontsize=9)
    ax1.set_xlabel('维度', fontsize=11)
    ax1.set_title('旧版标签（11种）\n总维度: 24', fontsize=12, fontweight='bold', color='darkred')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加维度数值
    for i, (bar, dim) in enumerate(zip(bars1, old_dims)):
        ax1.text(dim + 0.3, i, str(dim), va='center', fontsize=9)
    
    # 绘制新版
    bars2 = ax2.barh(range(len(new_labels)), new_dims, color='lightblue', alpha=0.7)
    ax2.set_yticks(range(len(new_labels)))
    ax2.set_yticklabels(new_labels, fontsize=11)
    ax2.set_xlabel('维度', fontsize=11)
    ax2.set_title('新版标签（2种）\n总维度: 15', fontsize=12, fontweight='bold', color='darkgreen')
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加维度数值
    for i, (bar, dim) in enumerate(zip(bars2, new_dims)):
        ax2.text(dim + 0.3, i, f'{dim} (7×2)' if dim == 14 else str(dim), va='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path('visuals')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'label_complexity_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    print("="*70)
    print("标签可视化脚本")
    print("="*70)
    
    # 1. 可视化未来阶段时间表
    print("\n1. 生成未来阶段时间表可视化...")
    visualize_future_schedule('video01', sample_frames=[0, 500, 1000, 1500, 1732])
    
    # 2. 对比标签复杂度
    print("\n2. 生成标签复杂度对比图...")
    compare_label_complexity()
    
    print("\n✓ 所有可视化完成！")
    print("  查看 visuals/ 目录")
