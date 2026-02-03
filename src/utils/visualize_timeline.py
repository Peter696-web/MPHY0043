"""
Visualize predicted vs ground truth timeline 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from pathlib import Path
import json


PHASE_NAMES = [
    'P0: Preparation',
    'P1: CalotTriangle',
    'P2: ClippingCutting',
    'P3: GallbladderDissection',
    'P4: GallbladderPackaging',
    'P5: CleaningCoagulation',
    'P6: GallbladderRetraction'
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
    return torch.load(pred_path, map_location='cpu')


def load_label_json(label_path):
    with open(label_path, 'r') as f:
        return json.load(f)


def extract_video_data(pred_data, video_id):
    for i, vid_batch in enumerate(pred_data['video_ids']):
        if vid_batch.item() == video_id or (vid_batch.dim() > 0 and video_id in vid_batch.tolist()):
            if vid_batch.dim() == 1 and len(vid_batch) == 1:
                b = 0
            else:
                b = (vid_batch == video_id).nonzero(as_tuple=True)[0][0].item()
            
            frame_ids = pred_data['frame_ids'][i][b].numpy()
            pred_schedule = pred_data['predictions'][i]['future_schedule'][b].numpy()
            true_schedule = pred_data['targets'][i]['future_schedule'][b].numpy()
            phase_logits = pred_data['predictions'][i]['phase_logits'][b].numpy()
            
            return {
                'frame_ids': frame_ids,
                'pred_schedule': pred_schedule,
                'true_schedule': true_schedule,
                'phase_logits': phase_logits
            }
    return None


def compute_phase_timeline_aggregated(schedule, frame_ids, video_len, phase_logits=None):
    """
    Args:
        schedule: (T, 7, 2) - [start_offset, remaining_time] for each phase
        frame_ids: (T,) - 
        video_len: 
        phase_logits: (T, 7) - 
    
    Returns:
        phase_ranges: list of (start, end) for each phase
    """

    schedule_denorm = schedule * video_len
    T = len(frame_ids)
    
    phase_ranges = []
    
    for phase_id in range(7):
        predicted_starts = []
        predicted_ends = []
        
        for t, current_frame in enumerate(frame_ids):
            start_offset = schedule_denorm[t, phase_id, 0]
            remaining = schedule_denorm[t, phase_id, 1]

            if start_offset <= 0:  
                pred_start = current_frame + start_offset
                pred_end = current_frame + remaining
            else:  
                pred_start = current_frame + start_offset
                pred_end = pred_start + remaining

            if pred_end > pred_start and pred_start >= 0:
                predicted_starts.append(pred_start)
                predicted_ends.append(pred_end)

        if len(predicted_starts) > 0:
            phase_start = np.median(predicted_starts)
            phase_end = np.median(predicted_ends)

            phase_start = max(0, min(phase_start, video_len))
            phase_end = max(phase_start, min(phase_end, video_len))
        else:
            if phase_logits is not None:
                phase_probs = np.exp(phase_logits) / np.exp(phase_logits).sum(axis=1, keepdims=True)
                phase_pred = phase_probs.argmax(axis=1)

                phase_frames = np.where(phase_pred == phase_id)[0]
                if len(phase_frames) > 0:
                    phase_start = frame_ids[phase_frames[0]]
                    phase_end = frame_ids[phase_frames[-1]] + 1
                else:
                    phase_start = 0
                    phase_end = 0
            else:
                phase_start = 0
                phase_end = 0
        
        phase_ranges.append((phase_start, phase_end))
    
    return phase_ranges


def compute_phase_timeline_from_classification(phase_logits, frame_ids):
    """
    Returns:
        phase_ranges: list of (start, end) for each phase
    """
    phase_probs = np.exp(phase_logits) / np.exp(phase_logits).sum(axis=1, keepdims=True)
    phase_pred = phase_probs.argmax(axis=1)
    
    phase_ranges = []
    for phase_id in range(7):
        phase_frames = np.where(phase_pred == phase_id)[0]
        if len(phase_frames) > 0:
            phase_start = frame_ids[phase_frames[0]]
            phase_end = frame_ids[phase_frames[-1]] + 1
        else:
            phase_start = 0
            phase_end = 0
        phase_ranges.append((phase_start, phase_end))
    
    return phase_ranges


def plot_gantt_chart(video_data, video_len, label_data, save_path, video_id, 
                     use_classification=False):
    pred_schedule = video_data['pred_schedule']
    frame_ids = video_data['frame_ids']
    phase_logits = video_data.get('phase_logits')

    if use_classification and phase_logits is not None:
        pred_ranges = compute_phase_timeline_from_classification(phase_logits, frame_ids)
        method_name = "Classification"
    else:
        pred_ranges = compute_phase_timeline_aggregated(
            pred_schedule, frame_ids, video_len, phase_logits
        )
        method_name = "Regression (Aggregated)"
    
    # load true ranges from label data
    true_ranges = []
    for phase_id in range(7):
        phase_segs = [seg for seg in label_data['segments'] if seg['phase_id'] == phase_id]
        if phase_segs:
            seg = phase_segs[0]
            true_ranges.append((seg['start_frame'], seg['end_frame']))
        else:
            true_ranges.append((0, 0))
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    title_suffix = f" [{method_name}]" if use_classification else ""
    fig.suptitle(f'Video {video_id:02d} - Surgical Phase Timeline {title_suffix}', 
                 fontsize=16, fontweight='bold')
    
    # === subplot-1 Ground Truth Time ===
    ax1.set_title('Ground Truth Timeline', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Phase', fontsize=12)
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(PHASE_NAMES, fontsize=10)
    ax1.set_ylim(-0.5, 6.5)
    ax1.grid(True, axis='x', alpha=0.3)
    
    for phase_id, (start, end) in enumerate(true_ranges):
        if end > start:
            duration = end - start
            ax1.barh(phase_id, duration, left=start, height=0.6,
                    color=PHASE_COLORS[phase_id], alpha=0.8, edgecolor='black', linewidth=1.5)
            if duration > 50:
                ax1.text(start + duration/2, phase_id, f'{int(duration)}f',
                        ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # === sublplot2: Predicted Time ===
    ax2.set_title(f'Predicted Timeline ({method_name})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (frames)', fontsize=12)
    ax2.set_ylabel('Phase', fontsize=12)
    ax2.set_yticks(range(7))
    ax2.set_yticklabels(PHASE_NAMES, fontsize=10)
    ax2.set_ylim(-0.5, 6.5)
    ax2.grid(True, axis='x', alpha=0.3)
    
    for phase_id, (start, end) in enumerate(pred_ranges):
        if end > start:
            duration = end - start
            ax2.barh(phase_id, duration, left=start, height=0.6,
                    color=PHASE_COLORS[phase_id], alpha=0.8, edgecolor='black', linewidth=1.5)
            if duration > 50:
                ax2.text(start + duration/2, phase_id, f'{int(duration)}f',
                        ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gantt chart saved to: {save_path}")
    
    # Error statistics
    print("\n" + "="*60)
    print(f"Phase Timeline Comparison - Video {video_id:02d}")
    print("="*60)
    
    total_error = 0
    valid_phases = 0
    
    for phase_id in range(7):
        true_start, true_end = true_ranges[phase_id]
        pred_start, pred_end = pred_ranges[phase_id]
        
        true_duration = true_end - true_start
        pred_duration = pred_end - pred_start
        
        if true_duration > 0:
            valid_phases += 1
            start_error = pred_start - true_start
            end_error = pred_end - true_end
            duration_error = pred_duration - true_duration
            duration_error_pct = (duration_error / true_duration) * 100
            
            total_error += abs(start_error) + abs(end_error)
            
            print(f"\n{PHASE_NAMES[phase_id]}:")
            print(f"  True:  {int(true_start):4d} → {int(true_end):4d}  (duration: {int(true_duration):4d}f)")
            print(f"  Pred:  {int(pred_start):4d} → {int(pred_end):4d}  (duration: {int(pred_duration):4d}f)")
            print(f"  Error: start={int(start_error):+4d}f, end={int(end_error):+4d}f, duration={int(duration_error):+4d}f ({duration_error_pct:+.1f}%)")
    
    if valid_phases > 0:
        mae = total_error / (valid_phases * 2)  # 2 for start and end
        print(f"\n{'='*60}")
        print(f"Overall MAE (start/end): {mae:.1f} frames ({mae/25:.2f} seconds @ 25fps)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize surgical phase timeline')
    parser.add_argument('--pred_path', type=str, required=True,
                       help='Path to prediction .pt file')
    parser.add_argument('--video_id', type=int, required=True,
                       help='Video ID to visualize')
    parser.add_argument('--label_dir', type=str, 
                       default='data/labels/aligned_labels',
                       help='Directory containing label JSON files')
    parser.add_argument('--save_dir', type=str,
                       default='results/figures/gantt_charts_v2',
                       help='Directory to save figures')
    parser.add_argument('--use_classification', action='store_true',
                       help='Use classification predictions instead of regression')
    
    args = parser.parse_args()
    

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    

    print(f"\n Loading prediction data from: {args.pred_path}")
    pred_data = load_prediction_data(args.pred_path)
    
    label_path = Path(args.label_dir) / f'video{args.video_id:02d}.json'
    print(f" Loading label data from: {label_path}")
    label_data = load_label_json(label_path)
    

    video_data = extract_video_data(pred_data, args.video_id)
    if video_data is None:
        print(f"Video {args.video_id} not found in prediction file!")
        return
    
    video_len = label_data['num_frames']
    print(f" Video {args.video_id:02d}: {video_len} frames")
    

    method_suffix = "_classification" if args.use_classification else "_regression"
    gantt_path = save_dir / f'gantt_video{args.video_id:02d}{method_suffix}.png'
    plot_gantt_chart(video_data, video_len, label_data, gantt_path, args.video_id,
                    use_classification=args.use_classification)
    
    print(f"\n Visualization completed!")


if __name__ == '__main__':
    main()
