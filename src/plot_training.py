"""
Visualize training history and prediction results
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

"""
Visualize training history and prediction results
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_path):
    """Plot training history: loss and metric curves"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_history = history['train']
    val_history = history['val']

    epochs = [h['epoch'] for h in train_history]

    # Extract metrics
    train_loss = [h['losses']['total'] for h in train_history]
    val_loss = [h['losses']['total'] for h in val_history]

    train_acc = [h['metrics']['accuracy'] for h in train_history]
    val_acc = [h['metrics']['accuracy'] for h in val_history]

    train_f1 = [h['metrics']['f1'] for h in train_history]
    val_f1 = [h['metrics']['f1'] for h in val_history]

    train_mae = [h['metrics']['mae'] for h in train_history]
    val_mae = [h['metrics']['mae'] for h in val_history]

    train_rmse = [h['metrics']['rmse'] for h in train_history]
    val_rmse = [h['metrics']['rmse'] for h in val_history]

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - Frame-level Surgical Phase Prediction', fontsize=16)

    # Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Training set')
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation set')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Training set')
    axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation set')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[0, 2].plot(epochs, train_f1, 'b-', label='Training set')
    axes[0, 2].plot(epochs, val_f1, 'r-', label='Validation set')
    axes[0, 2].set_title('Macro-averaged F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # MAE
    axes[1, 0].plot(epochs, train_mae, 'b-', label='Training set')
    axes[1, 0].plot(epochs, val_mae, 'r-', label='Validation set')
    axes[1, 0].set_title('MAE (Regression)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # RMSE
    axes[1, 1].plot(epochs, train_rmse, 'b-', label='Training set')
    axes[1, 1].plot(epochs, val_rmse, 'r-', label='Validation set')
    axes[1, 1].set_title('RMSE (Regression)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Score
    val_scores = [h.get('score', 0) for h in val_history]
    axes[1, 2].plot(epochs, val_scores, 'g-', label='Validation Score')
    axes[1, 2].set_title('Validation Score (F1 + RMSE)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path('results/figures')
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_path / 'training_history.pdf', bbox_inches='tight')
    print(f"Save plots to {save_path / 'training_history.png'} and .pdf")
    plt.show()

def plot_time_predictions(predictions_path, max_videos=5):
    """Plot time prediction comparison chart"""
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

        # Get predicted and true schedules
        pred_sched = predictions['future_schedule'][0].numpy()  # (7, 2)
        true_sched = targets['future_schedule'][0].numpy()  # (7, 2)

        ax = axes[i]

        # Plot prediction vs true for each phase
        x = np.arange(len(phase_names))
        width = 0.35

        # Predicted values
        pred_starts = pred_sched[:, 0]
        pred_durations = pred_sched[:, 1]

        # True values
        true_starts = true_sched[:, 0]
        true_durations = true_sched[:, 1]

        # Plot start times
        ax.bar(x - width/2, pred_starts, width, label='Predicted start time', alpha=0.7, color='blue')
        ax.bar(x - width/2, pred_durations, width, bottom=pred_starts,
               label='Predicted duration', alpha=0.7, color='lightblue')

        ax.bar(x + width/2, true_starts, width, label='True start time', alpha=0.7, color='red')
        ax.bar(x + width/2, true_durations, width, bottom=true_starts,
               label='True duration', alpha=0.7, color='pink')

        ax.set_xlabel('Surgical Phase')
        ax.set_ylabel('Normalized Time')
        ax.set_title(f'Video {video_id} - Surgical Phase Time Prediction Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/time_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_time_errors(predictions_path):
    """Plot time prediction error distribution"""
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

        # Only collect valid predictions (non-negative values)
        valid_mask = true_sched[:, 0] >= 0
        all_pred_starts.extend(pred_sched[valid_mask, 0])
        all_true_starts.extend(true_sched[valid_mask, 0])
        all_pred_durations.extend(pred_sched[valid_mask, 1])
        all_true_durations.extend(true_sched[valid_mask, 1])

    # Calculate errors
    start_errors = np.array(all_pred_starts) - np.array(all_true_starts)
    duration_errors = np.array(all_pred_durations) - np.array(all_true_durations)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Start time error
    axes[0].hist(start_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(start_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(start_errors):.4f}')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Start Time Prediction Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Duration error
    axes[1].hist(duration_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(duration_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(duration_errors):.4f}')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Duration Prediction Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/time_errors.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # Create figures directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    # 1. Plot training history
    print("Plot training loss curves...")
    plot_training_history('results/models/lstm_b1_norm/history.json')

    # 2. Plot time prediction comparison
    print("Plot time prediction comparison...")
    plot_time_predictions('results/models/lstm_b1_norm/val_predictions_epoch001.pt')

    # 3. Plot time prediction error
    print("Plot time prediction error distribution...")
    plot_time_errors('results/models/lstm_b1_norm/val_predictions_epoch001.pt')

    print("Visualization completed! Images saved to results/figures/")