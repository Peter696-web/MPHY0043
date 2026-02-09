"""
Visualization Script for Task B (Pure Classification)
Visualizes phase classification predictions vs ground truth.
No regression output visualization.
"""

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
import sys
import argparse
import csv
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

# Define Phase names and colors
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

def calculate_class_metrics(pred_phases, true_phases):
    """
    Calculate F1 score for each class (phase).
    """
    labels = range(7)
    f1_scores = f1_score(true_phases, pred_phases, labels=labels, average=None, zero_division=0)
    return f1_scores

def plot_class_f1_scores(f1_scores, save_path):
    """
    Plot F1 scores for each phase as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(7), f1_scores, color=PHASE_COLORS, edgecolor='black', alpha=0.8)
    
    plt.title('F1 Score per Phase', fontsize=14, fontweight='bold')
    plt.xlabel('Phase', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks(range(7), [f'P{i}' for i in range(7)])
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PHASE_COLORS[i], label=f'P{i}: {PHASE_NAMES[i]}') 
                       for i in range(7)]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved class F1 plot: {save_path}")
    plt.close()

def plot_confusion_matrix_single(preds, trues, save_path):
    """
    Plot confusion matrix for a single model.
    """
    cm = confusion_matrix(trues, preds, labels=range(7))
    
    plt.figure(figsize=(8, 7))
    plt.title('Confusion Matrix: Task B Phase Classification', fontsize=14, fontweight='bold')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                xticklabels=[f'P{i}' for i in range(7)], 
                yticklabels=[f'P{i}' for i in range(7)])
                
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix: {save_path}")
    plt.close()

def load_label_json(label_path):
    """Load ground truth JSON"""
    with open(label_path, 'r') as f:
        return json.load(f)

def get_video_index(video_id, data_dir, split='val'):
    """
    Determine the index of the video in the validation set.
    Since train_B.py saves predictions sequentially based on sorted file names.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise ValueError(f"Data directory {split_dir} does not exist.")
    
    # Get sorted list of npz files
    files = sorted(list(split_dir.glob('*.npz')))
    
    # Find index
    target_file = f"video{video_id:02d}.npz"
    for idx, f in enumerate(files):
        if f.name == target_file:
            return idx, f
            
    return -1, None

def plot_phase_comparison(video_id, pred_phases, true_phases, frame_ids, label_data, save_path, show=True):
    """
    Plot phase classification: prediction vs ground truth
    """
    video_len = len(frame_ids)
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    fig.suptitle(f'video{video_id:02d} - Phase Classification Task B', 
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
    print(f"✅ Saved phase classification plot: {save_path}")
    if show:
        plt.show()  # Display the plot
    plt.close()

def process_single_video(video_id, idx, vid_pred, args, metrics_list=None):
    """
    Process a single video: calculate metrics and plot.
    If metrics_list is provided, append metrics to it instead of printing only.
    Returns pred_phases, true_phases for confusion matrix.
    """
    if 'phase_logits' not in vid_pred:
         print(f"Error: 'phase_logits' not found in prediction at index {idx}")
         return None, None
         
    phase_logits = vid_pred['phase_logits']
    if phase_logits.dim() == 3: # (B, T, C)
        phase_logits = phase_logits.squeeze(0)
    
    pred_phases = torch.argmax(phase_logits, dim=-1).numpy()
    
    # Load Ground Truth
    label_path = Path(args.label_dir) / f"video{video_id:02d}.json"
    if not label_path.exists():
        print(f"Error: Label file not found: {label_path}")
        return None, None
        
    label_data = load_label_json(label_path)
    video_len = label_data['num_frames']
    
    if len(pred_phases) != video_len:
        if metrics_list is None: 
            print(f"Warning at video {video_id}: Prediction length ({len(pred_phases)}) != Label length ({video_len})")
        min_len = min(len(pred_phases), video_len)
        pred_phases = pred_phases[:min_len]
        video_len = min_len
        frame_ids = np.arange(video_len)
    else:
        frame_ids = np.arange(video_len)

    # Output Dir
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(args.pred_path).parent / 'figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plotting
    save_path = save_dir / f"video{video_id:02d}_task_B_comparison.png"
    # For batch mode, force no-show
    show_plot = args.show if metrics_list is None else False
    plot_phase_comparison(video_id, pred_phases, None, frame_ids, label_data, save_path, show_plot)
    
    # Metrics
    true_phases = np.zeros(video_len, dtype=int)
    for seg in label_data['segments']:
        s, e, p = seg['start_frame'], seg['end_frame'], seg['phase_id']
        s = max(0, s)
        e = min(video_len, e)
        true_phases[s:e] = p
        
    acc = np.mean(pred_phases == true_phases)
    f1 = f1_score(true_phases, pred_phases, average='macro', zero_division=0)
    
    if metrics_list is not None:
        metrics_list.append({
            'video_name': f'video{video_id:02d}',
            'accuracy': acc,
            'f1': f1
        })
    else:
        print(f"\n📈 Accuracy for video {video_id}: {acc:.4f}")
        print(f"   F1 Score for video {video_id}: {f1:.4f}")
    
    return pred_phases, true_phases

def main():
    parser = argparse.ArgumentParser(description='Visualize Task B predictions')
    parser.add_argument('--pred_path', type=str, default=None,
                        help='Path to prediction .pt file')
    parser.add_argument('--video_id', type=int, default=None,
                        help='Video ID (e.g., 70), required unless --batch is set')
    parser.add_argument('--batch', action='store_true',
                        help='Process all videos in validation set')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data (to determine validation set order)')
    parser.add_argument('--label_dir', type=str, default='data/labels/aligned_labels',
                        help='Directory with label JSON files')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Data split to use (default: val)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--scores_dir', type=str, default=None,
                        help='Output directory for scores CSV (batch mode only)')
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='Do not show plots on screen')
    parser.add_argument('--confusion_matrix', action='store_true',
                        help='Generate confusion matrix (batch mode)')
    
    args = parser.parse_args()
    args.show = not args.no_show
    
    if args.batch:
        # Load predictions
        print(f"📂 Loading predictions from: {args.pred_path}")
        pred_data = load_prediction_data(args.pred_path)
        
        if 'predictions' not in pred_data:
            print("❌ Error: 'predictions' key not found in .pt file.")
            return

        predictions_list = pred_data['predictions']
        
        # Determine file list
        split_dir = Path(args.data_dir) / args.split
        if not split_dir.exists():
            print(f"❌ Error: Data directory {split_dir} does not exist.")
            return
        files = sorted(list(split_dir.glob('*.npz')))
        
        if len(files) != len(predictions_list):
            print(f"⚠️ Warning: Mismatch between files in {split_dir} ({len(files)}) and predictions ({len(predictions_list)})")
            
        print(f"🚀 Processing all {len(predictions_list)} videos...")
        metrics_list = []
        all_pred_phases = []
        all_true_phases = []
        
        for idx, file_path in enumerate(files):
            if idx >= len(predictions_list):
                break
                
            try:
                vid_id_str = file_path.stem.replace('video', '')
                vid_id = int(vid_id_str)
            except ValueError:
                continue
                
            print(f"   Processing {file_path.name} (ID: {vid_id})...")
            
            # Process and collect phases for confusion matrix / class metrics
            # Note: process_single_video saves individual comparison plots 
            pred_p, true_p = process_single_video(vid_id, idx, predictions_list[idx], args, metrics_list)
            
            if pred_p is not None and true_p is not None:
                all_pred_phases.append(pred_p)
                all_true_phases.append(true_p)
                
        # Save per-video metrics
        if args.scores_dir:
            scores_dir = Path(args.scores_dir)
        else:
            scores_dir = Path(args.pred_path).parent / 'scores'
        scores_dir.mkdir(parents=True, exist_ok=True)
        csv_path = scores_dir / 'task_B_metrics.csv'
        
        print(f"\n💾 Saving per-video metrics to {csv_path}")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['video_name', 'accuracy', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_list:
                writer.writerow(row)
                
        # --- NEW: Calculate Overall Class Metrics ---
        if all_pred_phases and all_true_phases:
            all_pred_flat = np.concatenate(all_pred_phases)
            all_true_flat = np.concatenate(all_true_phases)
            
            # 1. Confusion Matrix
            if args.confusion_matrix:
                save_dir = Path(args.save_dir) if args.save_dir else Path(args.pred_path).parent / 'figures'
                save_dir.mkdir(parents=True, exist_ok=True)
                cm_path = save_dir / 'task_B_confusion_matrix.png'
                plot_confusion_matrix_single(all_pred_flat, all_true_flat, cm_path)
            
            # 2. Per-Class F1 Scores
            class_f1s = calculate_class_metrics(all_pred_flat, all_true_flat)
            
            # Save Class F1 Plot
            save_dir = Path(args.save_dir) if args.save_dir else Path(args.pred_path).parent / 'figures'
            save_dir.mkdir(parents=True, exist_ok=True)
            f1_plot_path = save_dir / 'task_B_class_f1_scores.png'
            plot_class_f1_scores(class_f1s, f1_plot_path)
            
            # Save Class F1 CSV
            class_csv_path = scores_dir / 'task_B_class_f1_scores.csv'
            print(f"💾 Saving per-class F1 scores to {class_csv_path}")
            with open(class_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Phase', 'Phase Name', 'F1 Score'])
                for i in range(7):
                    writer.writerow([f'P{i}', PHASE_NAMES[i], f'{class_f1s[i]:.4f}'])
                    print(f"   P{i} ({PHASE_NAMES[i]}): {class_f1s[i]:.4f}")
                    
        print("✅ Batch processing complete.")
            

    elif args.video_id is not None:
        # Load predictions
        print(f"📂 Loading predictions from: {args.pred_path}")
        pred_data = load_prediction_data(args.pred_path)
        
        if 'predictions' not in pred_data:
            print("❌ Error: 'predictions' key not found in .pt file.")
            return

        predictions_list = pred_data['predictions']
        
        split_dir = Path(args.data_dir) / args.split
        files = sorted(list(split_dir.glob('*.npz')))

        target_file = f"video{args.video_id:02d}.npz"
        idx = -1
        for i, f in enumerate(files):
            if f.name == target_file:
                idx = i
                break
        
        if idx == -1:
            print(f"❌ Error: Video {args.video_id} not found in {args.data_dir}/{args.split}")
            return
            
        print(f"   ✓ Found at index {idx} ({target_file})")
        if idx >= len(predictions_list):
            print(f"❌ Error: Index {idx} out of bounds.")
            return

        process_single_video(args.video_id, idx, predictions_list[idx], args, None)
    
    else:
        parser.error("--video_id is required unless --batch is set")

if __name__ == '__main__':
    main()
