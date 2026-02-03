"""
Visualize Remaining Time Labels
显示每个阶段的剩余时间预测标签（呈现直角三角形效果）
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Phase names and colors
PHASE_NAMES = [
    'Preparation',
    'CalotTriangleDissection', 
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderPackaging',
    'CleaningCoagulation',
    'GallbladderRetraction'
]

PHASE_COLORS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Cyan
    '#45B7D1',  # Blue
    '#FFA07A',  # Orange
    '#98D8C8',  # Green
    '#F7DC6F',  # Yellow
    '#BB8FCE'   # Purple
]


def visualize_remaining_time_triangles(video_id: str, 
                                       label_dir: str = 'data/new_preprocessed/aligned_labels',
                                       save_dir: str = 'data/new_preprocessed/visualizations'):
    """
    可视化剩余时间标签（直角三角形效果）
    
    纵轴：剩余时间（秒）
    横轴：当前时间（秒）
    每个阶段的剩余时间会从最大值线性下降到0，形成直角三角形
    """
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {video_id} data...")
    labels = np.load(label_dir / f'{video_id}_labels.npy', allow_pickle=True).item()
    
    with open(label_dir / f'{video_id}.json', 'r') as f:
        metadata = json.load(f)
    
    num_frames = len(labels['phase_id'])
    time_axis = np.arange(num_frames)
    
    # Extract future_schedule
    phase_id = labels['phase_id']  # (N,)
    future_schedule = labels['future_schedule']  # (N, 7, 2)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'{video_id} - Remaining Time Prediction Labels (Triangle Visualization)', 
                 fontsize=16, fontweight='bold')
    
    # ========== 1. Current Phase Remaining Time (Main Triangle View) ==========
    ax1 = axes[0]
    
    # Extract current phase remaining time
    current_remaining = np.zeros(num_frames)
    for i in range(num_frames):
        current_phase = phase_id[i]
        # future_schedule[i, current_phase, 1] is the remaining duration
        current_remaining[i] = future_schedule[i, current_phase, 1]
    
    # Plot the triangle
    ax1.plot(time_axis, current_remaining, linewidth=2, color='#2E86AB', 
             label='Current Phase Remaining Time', alpha=0.8)
    ax1.fill_between(time_axis, 0, current_remaining, alpha=0.3, color='#2E86AB')
    
    # Add phase background colors
    for seg in metadata['segments']:
        phase_idx = seg['phase_id']
        start = seg['start_frame']
        end = seg['end_frame']
        ax1.axvspan(start, end + 1, alpha=0.15, color=PHASE_COLORS[phase_idx])
        
        # Add phase name
        mid = (start + end) / 2
        ax1.text(mid, ax1.get_ylim()[1] * 0.95, 
                PHASE_NAMES[phase_idx][:20], 
                ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Remaining Time (seconds)', fontweight='bold', fontsize=12)
    ax1.set_title('Current Phase Remaining Time (Triangle Effect)', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, num_frames)
    
    # ========== 2. All Phases Stacked View ==========
    ax2 = axes[1]
    
    # Plot each phase's remaining time as separate lines
    for phase_idx in range(7):
        # Extract remaining time for this phase
        phase_remaining = future_schedule[:, phase_idx, 1].copy()
        
        # Set to NaN where phase is not active (for cleaner visualization)
        phase_remaining[phase_remaining < 0] = np.nan
        
        ax2.plot(time_axis, phase_remaining, linewidth=2, 
                color=PHASE_COLORS[phase_idx], 
                label=f'P{phase_idx}: {PHASE_NAMES[phase_idx][:15]}',
                alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Remaining Time (seconds)', fontweight='bold', fontsize=12)
    ax2.set_title('All Phases Remaining Time (Multi-Triangle View)', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, num_frames)
    
    # ========== 3. Phase Sequence + Triangle Overlay ==========
    ax3 = axes[2]
    
    # Plot current remaining time again
    ax3.plot(time_axis, current_remaining, linewidth=3, color='#E63946', 
             label='Current Phase Remaining', alpha=0.9, zorder=3)
    
    # Add phase segments as colored bars at the bottom
    for seg in metadata['segments']:
        phase_idx = seg['phase_id']
        start = seg['start_frame']
        end = seg['end_frame']
        duration = seg['duration']
        
        # Draw phase bar
        ax3.barh(y=-ax3.get_ylim()[1] * 0.05, width=duration, left=start, 
                height=ax3.get_ylim()[1] * 0.1, 
                color=PHASE_COLORS[phase_idx], alpha=0.6, 
                edgecolor='black', linewidth=1)
        
        # Add phase label
        if duration > 30:  # Only show label if segment is wide enough
            ax3.text(start + duration/2, -ax3.get_ylim()[1] * 0.05,
                    f'P{phase_idx}', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
    
    ax3.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Remaining Time (seconds)', fontweight='bold', fontsize=12)
    ax3.set_title('Triangle + Phase Sequence', fontweight='bold', fontsize=13)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, num_frames)
    ax3.set_ylim(-50, ax3.get_ylim()[1])  # Add space for phase bars
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_dir / f'{video_id}_remaining_time_triangles.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"{video_id} Statistics")
    print(f"{'='*60}")
    print(f"Total frames: {num_frames} frames ({num_frames/60:.1f} minutes)")
    print(f"\nPhase distribution:")
    for seg in metadata['segments']:
        duration = seg['duration']
        percentage = duration / num_frames * 100
        print(f"  P{seg['phase_id']}: {seg['phase_name']:30s} | "
              f"{seg['start_frame']:4d}-{seg['end_frame']:4d} | "
              f"{duration:4d}s ({percentage:5.1f}%)")
    
    print(f"\nRemaining time range: {current_remaining[current_remaining >= 0].min():.0f} - "
          f"{current_remaining.max():.0f} seconds")


def visualize_heatmap(video_id: str,
                      label_dir: str = 'data/new_preprocessed/aligned_labels',
                      save_dir: str = 'data/new_preprocessed/visualizations'):
    """
    热力图：展示所有阶段的剩余时间
    """
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nCreating heatmap for {video_id}...")
    labels = np.load(label_dir / f'{video_id}_labels.npy', allow_pickle=True).item()
    
    num_frames = len(labels['phase_id'])
    future_schedule = labels['future_schedule']  # (N, 7, 2)
    
    # Extract remaining times for all phases
    remaining_times = future_schedule[:, :, 1].T  # (7, N)
    
    # Replace -1 with NaN for better visualization
    remaining_times_display = remaining_times.copy()
    remaining_times_display[remaining_times_display < 0] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Plot heatmap
    im = ax.imshow(remaining_times_display, aspect='auto', cmap='RdYlGn_r', 
                   interpolation='nearest', origin='lower')
    
    # Set ticks and labels
    ax.set_yticks(range(7))
    ax.set_yticklabels([f'P{i}: {PHASE_NAMES[i][:20]}' for i in range(7)], fontsize=10)
    ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_title(f'{video_id} - Phase Remaining Time Heatmap', fontweight='bold', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Remaining Time (seconds)', rotation=270, labelpad=20, fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(0, num_frames, 100))
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / f'{video_id}_remaining_time_heatmap.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    
    plt.show()


def compare_multiple_videos(video_ids: list,
                           label_dir: str = 'data/new_preprocessed/aligned_labels',
                           save_dir: str = 'data/new_preprocessed/visualizations'):
    """
    比较多个视频的剩余时间曲线
    """
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(len(video_ids), 1, figsize=(16, 4*len(video_ids)))
    if len(video_ids) == 1:
        axes = [axes]
    
    fig.suptitle('Multi-Video Remaining Time Comparison (Triangle Effect)', 
                 fontsize=16, fontweight='bold')
    
    for idx, video_id in enumerate(video_ids):
        ax = axes[idx]
        
        # Load data
        labels = np.load(label_dir / f'{video_id}_labels.npy', allow_pickle=True).item()
        
        with open(label_dir / f'{video_id}.json', 'r') as f:
            metadata = json.load(f)
        
        num_frames = len(labels['phase_id'])
        time_axis = np.arange(num_frames)
        
        # Extract current remaining time
        phase_id = labels['phase_id']
        future_schedule = labels['future_schedule']
        current_remaining = np.array([future_schedule[i, phase_id[i], 1] 
                                     for i in range(num_frames)])
        
        # Plot triangle
        ax.plot(time_axis, current_remaining, linewidth=2.5, color='#E63946', alpha=0.8)
        ax.fill_between(time_axis, 0, current_remaining, alpha=0.25, color='#E63946')
        
        # Add phase backgrounds
        for seg in metadata['segments']:
            ax.axvspan(seg['start_frame'], seg['end_frame'] + 1, 
                      alpha=0.15, color=PHASE_COLORS[seg['phase_id']])
        
        ax.set_ylabel(f'{video_id}\n({num_frames}s)', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, num_frames)
        
        if idx == len(video_ids) - 1:
            ax.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_path = save_dir / 'videos_comparison_triangles.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved comparison: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # Visualize single video with triangle effect
    print("=" * 60)
    print("Visualizing video01 with triangle effect...")
    print("=" * 60)
    visualize_remaining_time_triangles('video01')
    
    # Create heatmap
    visualize_heatmap('video01')
    
    # Compare multiple videos (optional)
    print("\n" + "=" * 60)
    print("Comparing multiple videos...")
    print("=" * 60)
    compare_multiple_videos(['video01', 'video02', 'video03'])
    
    print("\n✓ All visualizations complete!")
