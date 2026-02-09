"""
Visualization Script for Task A (Classification + Regression)
Visualizes phase classification and remaining time predictions.
"""

import matplotlib
matplotlib.use('Agg') 
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
import os
import csv
from sklearn.metrics import f1_score, mean_absolute_error

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
    return torch.load(pred_path, map_location='cpu')

def load_label_json(label_path):
    with open(label_path, 'r') as f:
        return json.load(f)

def get_segments(phases):
    if len(phases) == 0:
        return []
    segments = []
    start = 0
    current = phases[0]
    for i in range(1, len(phases)):
        if phases[i] != current:
            segments.append({'phase_id': current, 'start_frame': start, 'end_frame': i})
            start = i
            current = phases[i]
    segments.append({'phase_id': current, 'start_frame': start, 'end_frame': len(phases)})
    return segments

def calculate_metrics(y_true_phase, y_pred_phase, y_true_reg, y_pred_reg):
    accuracy = np.mean(y_true_phase == y_pred_phase)
    f1 = f1_score(y_true_phase, y_pred_phase, average='macro')
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    
    # Calculate Per-Phase MAE
    mae_per_phase = {}
    for i in range(len(PHASE_NAMES)):
        # Identify frames belonging to phase i in Ground Truth
        mask = (y_true_phase == i)
        if np.sum(mask) > 0:
            phase_mae = mean_absolute_error(y_true_reg[mask], y_pred_reg[mask])
            mae_per_phase[f'mae_phase_{i}'] = phase_mae
        else:
            mae_per_phase[f'mae_phase_{i}'] = np.nan
            
    return accuracy, f1, mae, mae_per_phase

def plot_combined_visualization(video_id, pred_dict, gt_dict, segments, save_path, mae_per_phase):
    frame_ids = np.arange(len(pred_dict['phase']))
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [1, 1, 0.6]})
    fig.suptitle(f'Video {video_id:02d} - Task A Results', fontsize=16, fontweight='bold')
    
    # Phase Classification & Predict The Time of All Up-Coming Phases 
    ax = axes[0]
    ax.set_title(f'Predict The Time of All Up-Coming Phases', fontsize=12, fontweight='bold')
    for seg in segments:
        ax.axvspan(seg['start_frame'], seg['end_frame'], color=PHASE_COLORS[seg['phase_id']], alpha=0.3)
        mid_point = (seg['start_frame'] + seg['end_frame']) / 2
        if seg['end_frame'] - seg['start_frame'] > 30:
            ax.text(mid_point, 0.5, f'P{seg["phase_id"]}', ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Draw predicted phases as Gantt chart bars at corresponding y positions
    pred_segments = get_segments(pred_dict['phase'])
    for seg in pred_segments:
        ax.broken_barh([(seg['start_frame'], seg['end_frame'] - seg['start_frame'])], 
                       (seg['phase_id'] - 0.4, 0.8), 
                       facecolors=PHASE_COLORS[seg['phase_id']], 
                       alpha=0.8, edgecolor='none')
    
    ax.step(frame_ids, pred_dict['phase'], where='mid', color='blue', linestyle='--', label='Prediction', linewidth=2, alpha=1.0)
    ax.step(frame_ids, gt_dict['phase'], where='mid', color='red', linestyle='-', label='Ground Truth', linewidth=2, alpha=1.0)
    ax.set_yticks(range(7))
    ax.set_ylabel('Phase ID')
    ax.set_xlim(0, len(frame_ids))
    ax.legend(loc='upper right')
    
    # Regression
    ax = axes[1]
    ax.set_title(f'Remaining Time Prediction (MAE: {pred_dict["metrics"]["mae"]:.3f} sec)', fontsize=12, fontweight='bold')
    
    # Plot lines first to establish scale
    ax.plot(frame_ids, gt_dict['schedule'], 'r-', linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(frame_ids, pred_dict['schedule'], 'b--', linewidth=2.5, label='Prediction', alpha=0.9)
    
    # Add phase backgrounds and labels
    ylim = ax.get_ylim()
    for seg in segments:
        ax.axvspan(seg['start_frame'], seg['end_frame'], color=PHASE_COLORS[seg['phase_id']], alpha=0.25)
        # Add labels for wide segments
        if seg['end_frame'] - seg['start_frame'] > 50:
            mid_point = (seg['start_frame'] + seg['end_frame']) / 2
            ax.text(mid_point, ylim[1] * 0.98, f"P{seg['phase_id']}", 
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Remaining Time (seconds)', fontsize=11)
    ax.set_xlim(0, len(frame_ids))
    ax.grid(True, alpha=0.3)
    
    # Enhanced Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    line_elements = [
        Line2D([0], [0], color='r', linewidth=2.5, label='Ground Truth', alpha=0.9),
        Line2D([0], [0], color='b', linewidth=2.5, linestyle='--', label='Prediction', alpha=0.9)
    ]
    phase_elements = [
        Patch(facecolor=PHASE_COLORS[i], alpha=0.5, label=f'P{i}: {PHASE_NAMES[i]}') 
        for i in range(len(PHASE_NAMES))
    ]
    ax.legend(handles=line_elements + phase_elements, loc='upper right', fontsize=9, 
              ncol=2, framealpha=0.95, edgecolor='black')
              
    # MAE per Phase (Bar Chart)
    ax = axes[2]
    ax.set_title('Mean Absolute Error (MAE) per Phase', fontsize=12, fontweight='bold')
    
    phases = list(range(len(PHASE_NAMES)))
    phase_maes = [mae_per_phase.get(f'mae_phase_{i}', np.nan) for i in phases]
    
    # Filter out NaNs for plotting, but keep positions
    plot_maes = [m if not np.isnan(m) else 0 for m in phase_maes]
    
    bars = ax.bar(phases, plot_maes, color=PHASE_COLORS, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(phases)
    ax.set_xticklabels([f'P{i}\n{PHASE_NAMES[i][:15]}..' for i in phases], fontsize=9, rotation=15)
    ax.set_ylabel('MAE (seconds)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for idx, (bar, val) in enumerate(zip(bars, phase_maes)):
        if not np.isnan(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}s',
                   ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0,
                   'N/A',
                   ha='center', va='bottom', color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def process_single_video(video_id, idx, pred_entry, args, metrics_list=None):
    split_dir = Path(args.data_dir) / args.split
    gt_file = split_dir / f'video{video_id:02d}.npz'
    
    if not gt_file.exists():
        print(f'❌ Error: GT file not found: {gt_file}')
        return

    gt_np = np.load(gt_file)
    phase_key = 'phase_labels' if 'phase_labels' in gt_np else 'phase_id'
    if phase_key not in gt_np:
        print(f'❌ Error: Missing phase key in {gt_file.name}')
        return
        
    true_phases = gt_np[phase_key]
    if 'future_schedule' not in gt_np:
         print(f'❌ Error: future_schedule not in {gt_file.name}')
         return
    true_schedule_full = gt_np['future_schedule'] 
    
    # Remove batch dimension if present
    if pred_entry['phase_logits'].dim() == 3:
        pred_entry['phase_logits'] = pred_entry['phase_logits'].squeeze(0)
    if 'future_schedule' in pred_entry and pred_entry['future_schedule'].dim() == 4:
        pred_entry['future_schedule'] = pred_entry['future_schedule'].squeeze(0)

    # Align lengths if mismatch exists
    T_true = len(true_phases)
    T_pred = pred_entry['phase_logits'].shape[0]
    T = min(T_true, T_pred)
    
    if T_true != T_pred:
        # print(f"⚠️ Warning: Length mismatch for video{video_id:02d}. GT: {T_true}, Pred: {T_pred}. Truncating to {T}.")
        true_phases = true_phases[:T]
        true_schedule_full = true_schedule_full[:T]
        pred_entry['phase_logits'] = pred_entry['phase_logits'][:T]
        if 'future_schedule' in pred_entry:
            pred_entry['future_schedule'] = pred_entry['future_schedule'][:T]

    if true_schedule_full.ndim == 3:
        true_remaining = np.array([true_schedule_full[t, int(true_phases[t]), 1] for t in range(T)])
    elif true_schedule_full.ndim == 2:
        true_remaining = true_schedule_full[:, 1]
    else:
        true_remaining = true_schedule_full

    pred_logits = pred_entry['phase_logits']
    pred_phases = torch.argmax(pred_logits, dim=-1).numpy()
    
    if 'future_schedule' in pred_entry:
        pred_sched_np = pred_entry['future_schedule'].numpy()
        pred_remaining = np.array([pred_sched_np[t, pred_phases[t], 1] for t in range(T)])
    else:
        pred_remaining = np.zeros(T)
    
    # Denormalize
    if true_remaining.max() > 1.0:
        true_remaining_frames = true_remaining
    else:
        true_remaining_frames = true_remaining * T
        
    pred_remaining_frames = pred_remaining * T
    
    true_remaining_frames = np.maximum(0, true_remaining_frames)
    pred_remaining_frames = np.maximum(0, pred_remaining_frames)

    # Convert frames to seconds
    # User confirmed 1 frame = 1 second (1 FPS)
    fps = 1.0
    true_remaining_sec = true_remaining_frames / fps
    pred_remaining_sec = pred_remaining_frames / fps
    
    acc, f1, mae, mae_per_phase = calculate_metrics(true_phases, pred_phases, true_remaining_sec, pred_remaining_sec)
    
    if metrics_list is not None:
        metric_entry = {'video_name': f'video{video_id:02d}', 'accuracy': acc, 'f1': f1, 'mae': mae}
        metric_entry.update(mae_per_phase)
        metrics_list.append(metric_entry)
        print(f'   Processed video{video_id:02d}: F1={f1:.4f}, MAE={mae:.4f}')
    else:
        print(f'\n📈 Metrics for video{video_id:02d}:')
        print(f'   Accuracy: {acc:.4f}\n   F1 Score: {f1:.4f}\n   MAE     : {mae:.4f}')
        print('   MAE per Phase:')
        for k, v in mae_per_phase.items():
            if not np.isnan(v):
                print(f'     {k}: {v:.4f}')

    label_path = Path(args.label_dir) / f'video{video_id:02d}.json'
    segments = []
    if label_path.exists():
        with open(label_path, 'r') as f: segments = json.load(f)['segments']
    
    # Filter and clamp segments to match the visualization length (T)
    T_vis = len(pred_phases)
    filtered_segments = []
    for seg in segments:
        if seg['start_frame'] >= T_vis:
            continue
        
        # Clamp end frame
        real_end = min(seg['end_frame'], T_vis)
        
        # Only keep segments with positive duration
        if real_end > seg['start_frame']:
            seg_copy = seg.copy()
            seg_copy['end_frame'] = real_end
            filtered_segments.append(seg_copy)
    segments = filtered_segments
    
    save_dir = Path(args.save_dir if args.save_dir else Path(args.pred_path).parent / 'figures') / args.split
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f'video{video_id:02d}_task_A_vis.png'
    
    plot_combined_visualization(video_id, {'phase': pred_phases, 'schedule': pred_remaining_sec, 'metrics': {'f1': f1, 'mae': mae}},
                                {'phase': true_phases, 'schedule': true_remaining_sec}, segments, plot_path, mae_per_phase)


def main():
    parser = argparse.ArgumentParser(description='Visualize Task A predictions')
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--video_id', type=int, default=None)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--label_dir', type=str, default='data/labels/aligned_labels')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--scores_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    
    args = parser.parse_args()
    
    pred_data = load_prediction_data(args.pred_path)
    predictions_list = pred_data['predictions'] if isinstance(pred_data, dict) and 'predictions' in pred_data else pred_data
    
    split_dir = Path(args.data_dir) / args.split
    if not split_dir.exists():
         print(f'❌ Error: Data directory {split_dir} does not exist.')
         return
    gt_files = sorted(list(split_dir.glob('*.npz')))
    
    if args.batch:
        print(f'🚀 Processing all videos in {args.split} set...')
        metrics_list = []
        for idx in range(min(len(predictions_list), len(gt_files))):
            gt_file = gt_files[idx]
            try:
                video_id = int(gt_file.stem.replace('video', ''))
                process_single_video(video_id, idx, predictions_list[idx], args, metrics_list)
            except ValueError:
                continue
                
        if args.scores_dir:
            score_path = Path(args.scores_dir)
        else:
            score_path = Path(args.pred_path).parent / 'scores'
        score_path.mkdir(parents=True, exist_ok=True)
        
        csv_path = score_path / f'task_A_metrics_{args.split}.csv'
        if metrics_list:
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['video_name', 'accuracy', 'f1', 'mae'] + [f'mae_phase_{i}' for i in range(len(PHASE_NAMES))]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_list)
            print(f'\n💾 Saved metrics to {csv_path}')
    else:
        if args.video_id is None:
            print('❌ Error: --video_id required if not using --batch')
            return
        target_file = f'video{args.video_id:02d}.npz'
        idx = -1
        for i, f in enumerate(gt_files):
            if f.name == target_file:
                idx = i
                break
        if idx == -1:
            print(f'❌ Error: Video {args.video_id} not found in {args.split} set.')
            return
        print(f'   Processing {target_file} (Index {idx})...')
        process_single_video(args.video_id, idx, predictions_list[idx], args)

if __name__ == '__main__':
    main()
