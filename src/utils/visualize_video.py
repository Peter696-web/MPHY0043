"""
Enhanced Visualization for Surgical Phase Prediction
Visualize predictions vs ground truth with triangle effect (remaining time)
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for displaying plots
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json


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
    '#4ECDC4',  # Teal  
    '#45B7D1',  # Blue
    '#96CEB4',  # Green
    '#FFEAA7',  # Yellow
    '#DFE6E9',  # Gray
    '#A29BFE'   # Purple
]


def load_prediction_data(pred_path):
    """Load prediction .pt file"""
    return torch.load(pred_path, map_location='cpu')


def load_label_json(label_path):
    """Load ground truth JSON"""
    with open(label_path, 'r') as f:
        return json.load(f)


def extract_video_predictions(pred_data, video_id):
    """
    Extract predictions for specific video
    Handles video-level batching (B=1, T=video_length)
    """
    for i, vid_batch in enumerate(pred_data['video_ids']):
        # Check if this batch contains our video
        if vid_batch.item() == video_id or (vid_batch.dim() > 0 and video_id in vid_batch.tolist()):
            # Get batch index containing this video
            if vid_batch.dim() == 1 and len(vid_batch) == 1:
                b = 0
            else:
                b = (vid_batch == video_id).nonzero(as_tuple=True)[0][0].item()
            
            # Extract data for this video
            frame_ids = pred_data['frame_ids'][i][b].numpy()
            
            # Phase predictions  
            phase_logits = pred_data['predictions'][i]['phase_logits'][b]
            pred_phases = torch.argmax(phase_logits, dim=-1).numpy()
            true_phases = pred_data['targets'][i]['phase_id'][b].numpy()
            
            # Schedule predictions
            pred_schedule = pred_data['predictions'][i]['future_schedule'][b].numpy()
            true_schedule = pred_data['targets'][i]['future_schedule'][b].numpy()
            
            return {
                'frame_ids': frame_ids,
                'pred_phases': pred_phases,
                'true_phases': true_phases,
                'pred_schedule': pred_schedule,
                'true_schedule': true_schedule
            }
    
    return None


def plot_triangle_visualization(video_data, video_len, label_data, save_path, show=True):
    """
    Create triangle visualization - Current Phase Remaining Time
    Shows the remaining time for the current phase at each timestep
    """
    frame_ids = video_data['frame_ids']
    true_phases = video_data['true_phases']
    pred_phases = video_data['pred_phases']
    pred_schedule = video_data['pred_schedule']
    true_schedule = video_data['true_schedule']
    
    # Denormalize schedules
    pred_schedule_denorm = pred_schedule * video_len
    true_schedule_denorm = true_schedule * video_len
    
    # Extract current phase remaining time
    pred_remaining = []
    true_remaining = []
    
    for t in range(len(frame_ids)):
        current_phase = true_phases[t]
        current_pred_phases = pred_phases[t]  
        pred_remaining.append(pred_schedule_denorm[t, current_pred_phases, 1])
        true_remaining.append(max(0, true_schedule_denorm[t, current_phase, 1]))
    
    pred_remaining = np.array(pred_remaining)
    true_remaining = np.array(true_remaining)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    fig.suptitle(f'video{video_id:02d} - Current Phase Remaining Time Prediction', 
                 fontsize=16, fontweight='bold')
    
    # Draw phase backgrounds with P0, P1, P2, etc.
    ax.set_title('Current Phase Remaining Time (Triangle Effect)', fontsize=12, fontweight='bold')
    
    # Draw phase backgrounds with P0, P1, P2, etc.
    
    segments = label_data['segments']
    for seg in segments:
        ax.axvspan(seg['start_frame'], seg['end_frame'], 
                   color=PHASE_COLORS[seg['phase_id']], alpha=0.25)
        # Add P0, P1, P2 labels
        mid_point = (seg['start_frame'] + seg['end_frame']) / 2
        if seg['end_frame'] - seg['start_frame'] > 50:  # Only label wide segments
            ax.text(mid_point, ax.get_ylim()[1] * 0.98, f"P{seg['phase_id']}", 
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot ground truth remaining time (red solid line)
    ax.plot(frame_ids, true_remaining, 'r-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    
    # Plot prediction remaining time (blue dashed line)
    ax.plot(frame_ids, pred_remaining, 'b--', linewidth=2.5, label='Prediction', alpha=0.9)
    
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Remaining Time (seconds)', fontsize=11)
    
    # Create detailed legend with phase descriptions
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Line legend items
    line_elements = [
        Line2D([0], [0], color='r', linewidth=2.5, label='Ground Truth', alpha=0.9),
        Line2D([0], [0], color='b', linewidth=2.5, linestyle='--', label='Prediction', alpha=0.9)
    ]
    
    # Phase color legend items
    phase_elements = [
        Patch(facecolor=PHASE_COLORS[i], alpha=0.5, label=f'P{i}: {PHASE_NAMES[i]}') 
        for i in range(7)
    ]
    
    # Combine legends
    all_elements = line_elements + phase_elements
    ax.legend(handles=all_elements, loc='upper right', fontsize=9, 
              ncol=2, framealpha=0.95, edgecolor='black')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, video_len])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved triangle visualization: {save_path}")
    if show:
        plt.show()  # Display the plot
    plt.close()

def plot_phase_comparison(video_data, video_len, label_data, save_path, show=True):
    """
    Plot phase classification: prediction vs ground truth
    """
    frame_ids = video_data['frame_ids']
    pred_phases = video_data['pred_phases']
    true_phases = video_data['true_phases']
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    fig.suptitle(f'video{video_id:02d} - Phase Classification Over Time', 
                 fontsize=14, fontweight='bold')
    
    # Ground truth
    ax = axes[0]
    ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
    segments = label_data['segments']
    for seg in segments:
        ax.axvspan(seg['start_frame'], seg['end_frame'], 
                   color=PHASE_COLORS[seg['phase_id']], alpha=0.7)
        mid_point = (seg['start_frame'] + seg['end_frame']) / 2
        # Use P0, P1, P2 instead of full names
        if seg['end_frame'] - seg['start_frame'] > 30:
            ax.text(mid_point, 0, f"P{seg['phase_id']}", 
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.set_ylim([-0.5, 0.5])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Prediction  
    ax = axes[1]
    ax.set_title('Prediction', fontsize=12, fontweight='bold')
    prev_phase = pred_phases[0]
    start_frame = frame_ids[0]
    
    for i in range(1, len(frame_ids)):
        if pred_phases[i] != prev_phase or i == len(frame_ids) - 1:
            end_frame = frame_ids[i] if i == len(frame_ids) - 1 else frame_ids[i-1]
            ax.axvspan(start_frame, end_frame, color=PHASE_COLORS[prev_phase], alpha=0.7)
            mid_point = (start_frame + end_frame) / 2
            # Use P0, P1, P2 instead of full names
            if end_frame - start_frame > 30:
                ax.text(mid_point, 0, f"P{prev_phase}", 
                        ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            start_frame = frame_ids[i]
            prev_phase = pred_phases[i]
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylim([-0.5, 0.5])
    ax.set_yticks([])
    ax.set_xlim([0, video_len])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend with full names
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PHASE_COLORS[i], label=f'P{i}: {PHASE_NAMES[i]}') 
                       for i in range(7)]
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
               fontsize=9, frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved phase classification: {save_path}")
    if show:
        plt.show()  # Display the plot
    plt.close()


def calculate_metrics(video_data):
    """Calculate video-level metrics"""
    pred_phases = video_data['pred_phases']
    true_phases = video_data['true_phases']
    pred_schedule = video_data['pred_schedule']
    true_schedule = video_data['true_schedule']
    
    # Phase accuracy
    phase_acc = (pred_phases == true_phases).mean()
    
    # Schedule MSE
    valid_mask = (true_schedule >= 0)
    if valid_mask.sum() > 0:
        schedule_mse = ((pred_schedule[valid_mask] - true_schedule[valid_mask]) ** 2).mean()
    else:
        schedule_mse = 0.0
    
    return {
        'phase_accuracy': phase_acc,
        'schedule_mse': schedule_mse
    }


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions for a video')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to prediction .pt file')
    parser.add_argument('--video_id', type=int, required=True,
                        help='Video ID (e.g., 61 for video61)')
    parser.add_argument('--label_dir', type=str, default='data/labels/aligned_labels',
                        help='Directory with label JSON files')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (default: same as pred_path)')
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='Do not show plots on screen (only save to file)')
    args = parser.parse_args()
    
    # Set show flag (default: True, unless --no-show is specified)
    args.show = not args.no_show
    
    global video_id
    video_id = args.video_id
    
    pred_path = Path(args.pred_path)
    label_dir = Path(args.label_dir)
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = pred_path.parent / 'figures'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"\n📂 Loading predictions: {pred_path}")
    pred_data = load_prediction_data(pred_path)
    
    print(f"📂 Loading labels from: {label_dir}")
    label_path = label_dir / f"video{video_id:02d}.json"
    if not label_path.exists():
        print(f"❌ Error: Label not found: {label_path}")
        return
    label_data = load_label_json(label_path)
    video_len = label_data['num_frames']
    
    # Extract video predictions
    print(f"📊 Extracting video {video_id:02d}...")
    video_data = extract_video_predictions(pred_data, video_id)
    
    if video_data is None:
        print(f"❌ Error: Video {video_id} not in predictions!")
        print(f"Available videos: {[v.item() for v in pred_data['video_ids']]}")
        return
    
    print(f"   ✓ Found {len(video_data['frame_ids'])} frames")
    print(f"   ✓ Frame range: {video_data['frame_ids'].min()} - {video_data['frame_ids'].max()}")
    print(f"   ✓ Video length: {video_len} frames")
    
    # Debug: Check prediction data range
    print(f"\n🔍 Debug Info:")
    print(f"   pred_schedule range: [{video_data['pred_schedule'].min():.4f}, {video_data['pred_schedule'].max():.4f}]")
    print(f"   true_schedule range: [{video_data['true_schedule'].min():.4f}, {video_data['true_schedule'].max():.4f}]")
    
    # Denormalize and check
    pred_denorm = video_data['pred_schedule'] * video_len
    true_denorm = video_data['true_schedule'] * video_len
    print(f"   pred_schedule (denorm) range: [{pred_denorm.min():.1f}, {pred_denorm.max():.1f}] frames")
    print(f"   true_schedule (denorm) range: [{true_denorm[true_denorm >= 0].min():.1f}, {true_denorm.max():.1f}] frames")
    
    # Calculate metrics
    metrics = calculate_metrics(video_data)
    print(f"\n📈 Metrics for video {video_id:02d}:")
    print(f"   Phase Accuracy: {metrics['phase_accuracy']:.3f}")
    print(f"   Schedule MSE: {metrics['schedule_mse']:.4f}")
    
    # Generate visualizations
    print(f"\n📊 Generating plots...")
    
    # Triangle visualization (main)
    triangle_path = save_dir / f'video{video_id:02d}_remaining_time_triangles.png'
    plot_triangle_visualization(video_data, video_len, label_data, triangle_path, args.show)
    
    # Phase classification
    phase_path = save_dir / f'video{video_id:02d}_phase_classification.png'
    plot_phase_comparison(video_data, video_len, label_data, phase_path, args.show)
    
    print(f"\n✅ All plots saved to: {save_dir}")


if __name__ == '__main__':
    main()
