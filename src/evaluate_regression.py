"""
Detailed Regression Task Evaluation Script

Evaluates Task A requirements:
1. Predict the remaining time of the current surgical phase
2. Estimate the start and end times of all upcoming phases in the procedure
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader

from model.data import create_dataloaders
from model.mstcn import MSTCNSurgicalPredictor


class RegressionEvaluator:
    """Detailed evaluator for regression tasks"""
    
    def __init__(self, checkpoint_path, data_dir, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Path to data directory
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        self.epoch = checkpoint['epoch']
        
        # Build model
        self.model = MSTCNSurgicalPredictor(
            feature_dim=self.config['feature_dim'],
            hidden_dim=self.config.get('mstcn_channels', 64),
            num_stages=self.config.get('mstcn_stages', 4),
            num_layers=self.config.get('mstcn_layers', 10),
            dropout=self.config['dropout'],
            num_phases=self.config['num_phases']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded (epoch {self.epoch})")
        
        # Load data
        _, _, self.test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=1,  # Process one video at a time
            num_workers=0,
            normalize_features=self.config.get('normalize_features', True),
            normalize_schedule=self.config.get('normalize_schedule', True),
        )
        
        self.num_phases = self.config['num_phases']
        self.phase_names = [
            'Preparation',
            'CalotTriangleDissection',
            'ClippingCutting',
            'GallbladderDissection',
            'GallbladderPackaging',
            'CleaningCoagulation',
            'GallbladderRetraction'
        ]
    
    def denormalize_schedule(self, schedule, video_length):
        """
        Convert normalized schedule back to frame numbers
        
        Args:
            schedule: (T, 7, 2) normalized schedule in [0, 1]
            video_length: int, total number of frames
            
        Returns:
            schedule_frames: (T, 7, 2) schedule in frame numbers
        """
        return schedule * video_length
    
    def compute_remaining_time(self, schedule, current_frame, current_phase):
        """
        Compute remaining time for current phase
        
        Args:
            schedule: (7, 2) schedule at current frame [start_frame, duration]
            current_frame: int, current frame number
            current_phase: int, current phase ID
            
        Returns:
            remaining_frames: float, remaining frames for current phase
        """
        if current_phase < 0 or current_phase >= 7:
            return 0.0
        
        start_frame = schedule[current_phase, 0]
        duration = schedule[current_phase, 1]
        end_frame = start_frame + duration
        
        remaining = max(0.0, end_frame - current_frame)
        return remaining
    
    def compute_phase_transitions(self, schedule):
        """
        Compute start and end times for all phases
        
        Args:
            schedule: (7, 2) schedule [start_frame, duration]
            
        Returns:
            transitions: dict with phase_id -> (start_frame, end_frame)
        """
        transitions = {}
        for phase_id in range(7):
            start_frame = schedule[phase_id, 0]
            duration = schedule[phase_id, 1]
            end_frame = start_frame + duration
            transitions[phase_id] = {
                'start': float(start_frame),
                'end': float(end_frame),
                'duration': float(duration)
            }
        return transitions
    
    def evaluate_video(self, batch):
        """Evaluate one video"""
        # Extract data
        features = batch['features'].to(self.device)  # (1, T, C)
        true_phases = batch['phase_id'][0].numpy()    # (T,)
        true_schedule = batch['future_schedule'][0].numpy()  # (T, 7, 2)
        video_id = batch['video_id'][0].item()
        video_length = features.shape[1]
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features)
            pred_schedule = predictions['future_schedule'][0].cpu().numpy()  # (T, 7, 2)
        
        # Denormalize if needed
        if self.config.get('normalize_schedule', True):
            pred_schedule = self.denormalize_schedule(pred_schedule, video_length)
            true_schedule = self.denormalize_schedule(true_schedule, video_length)
        
        # Task 1: Remaining time prediction
        remaining_time_errors = []
        remaining_time_predictions = []
        remaining_time_ground_truth = []
        
        for t in range(video_length):
            current_phase = true_phases[t]
            
            # Predicted remaining time
            pred_remaining = self.compute_remaining_time(
                pred_schedule[t], t, current_phase
            )
            
            # True remaining time
            true_remaining = self.compute_remaining_time(
                true_schedule[t], t, current_phase
            )
            
            remaining_time_predictions.append(pred_remaining)
            remaining_time_ground_truth.append(true_remaining)
            
            # Error
            error = abs(pred_remaining - true_remaining)
            remaining_time_errors.append(error)
        
        # Task 2: Phase transition prediction (evaluate at middle frame)
        mid_frame = video_length // 2
        pred_transitions = self.compute_phase_transitions(pred_schedule[mid_frame])
        true_transitions = self.compute_phase_transitions(true_schedule[mid_frame])
        
        # Compute transition errors
        transition_errors = {
            'start': [],
            'end': [],
            'duration': []
        }
        
        for phase_id in range(7):
            pred = pred_transitions[phase_id]
            true = true_transitions[phase_id]
            
            if true['duration'] > 0:  # Only evaluate valid phases
                transition_errors['start'].append(abs(pred['start'] - true['start']))
                transition_errors['end'].append(abs(pred['end'] - true['end']))
                transition_errors['duration'].append(abs(pred['duration'] - true['duration']))
        
        # Aggregate video results
        video_results = {
            'video_id': video_id,
            'video_length': video_length,
            
            # Task 1: Remaining time
            'remaining_time': {
                'mae': float(np.mean(remaining_time_errors)),
                'predictions': [float(x) for x in remaining_time_predictions],
                'ground_truth': [float(x) for x in remaining_time_ground_truth]
            },
            
            # Task 2: Phase transitions
            'phase_transitions': {
                'start_mae': float(np.mean(transition_errors['start'])) if transition_errors['start'] else 0.0,
                'end_mae': float(np.mean(transition_errors['end'])) if transition_errors['end'] else 0.0,
                'duration_mae': float(np.mean(transition_errors['duration'])) if transition_errors['duration'] else 0.0,
                'predictions': {str(k): v for k, v in pred_transitions.items()},
                'ground_truth': {str(k): v for k, v in true_transitions.items()}
            }
        }
        
        return video_results
    
    def evaluate_all(self, save_dir=None):
        """Evaluate all test videos"""
        print("\n" + "="*70)
        print("Evaluating Regression Tasks")
        print("="*70)
        
        all_results = []
        
        # Aggregate metrics
        task1_mae_list = []
        task2_start_mae_list = []
        task2_end_mae_list = []
        task2_duration_mae_list = []
        
        # Per-phase statistics
        phase_errors = defaultdict(lambda: defaultdict(list))
        
        for batch in self.test_loader:
            video_results = self.evaluate_video(batch)
            all_results.append(video_results)
            
            # Aggregate Task 1
            task1_mae_list.append(video_results['remaining_time']['mae'])
            
            # Aggregate Task 2
            task2_start_mae_list.append(video_results['phase_transitions']['start_mae'])
            task2_end_mae_list.append(video_results['phase_transitions']['end_mae'])
            task2_duration_mae_list.append(video_results['phase_transitions']['duration_mae'])
            
            # Per-phase errors
            for phase_id_str, pred in video_results['phase_transitions']['predictions'].items():
                phase_id = int(phase_id_str)
                true = video_results['phase_transitions']['ground_truth'][phase_id_str]
                
                if true['duration'] > 0:
                    phase_errors[phase_id]['start'].append(abs(pred['start'] - true['start']))
                    phase_errors[phase_id]['end'].append(abs(pred['end'] - true['end']))
                    phase_errors[phase_id]['duration'].append(abs(pred['duration'] - true['duration']))
        
        # Compute overall statistics
        summary = {
            'epoch': self.epoch,
            'num_videos': len(all_results),
            
            # Task 1: Remaining Time Prediction
            'task1_remaining_time': {
                'mae_mean': float(np.mean(task1_mae_list)),
                'mae_std': float(np.std(task1_mae_list))
            },
            
            # Task 2: Phase Transition Prediction
            'task2_phase_transitions': {
                'start_mae_mean': float(np.mean(task2_start_mae_list)),
                'start_mae_std': float(np.std(task2_start_mae_list)),
                'end_mae_mean': float(np.mean(task2_end_mae_list)),
                'end_mae_std': float(np.std(task2_end_mae_list)),
                'duration_mae_mean': float(np.mean(task2_duration_mae_list)),
                'duration_mae_std': float(np.std(task2_duration_mae_list))
            },
            
            # Per-phase statistics
            'per_phase_errors': {}
        }
        
        # Compute per-phase statistics
        for phase_id in range(7):
            if phase_id in phase_errors and phase_errors[phase_id]['start']:
                summary['per_phase_errors'][self.phase_names[phase_id]] = {
                    'start_mae': float(np.mean(phase_errors[phase_id]['start'])),
                    'end_mae': float(np.mean(phase_errors[phase_id]['end'])),
                    'duration_mae': float(np.mean(phase_errors[phase_id]['duration']))
                }
        
        # Print summary
        self.print_summary(summary)
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            with open(save_dir / 'regression_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            with open(save_dir / 'regression_detailed.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\n✅ Results saved to: {save_dir}")
        
        return summary, all_results
    
    def print_summary(self, summary):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print(f"TASK A REGRESSION EVALUATION (Epoch {summary['epoch']})")
        print("="*70)
        
        print("\n📊 Task 1: Remaining Time Prediction")
        print("-" * 70)
        task1 = summary['task1_remaining_time']
        print(f"  MAE:  {task1['mae_mean']:.2f} ± {task1['mae_std']:.2f} frames")
        
        print("\n📊 Task 2: Phase Transition Prediction")
        print("-" * 70)
        task2 = summary['task2_phase_transitions']
        print(f"  Start Frame MAE:    {task2['start_mae_mean']:.2f} ± {task2['start_mae_std']:.2f} frames")
        print(f"  End Frame MAE:      {task2['end_mae_mean']:.2f} ± {task2['end_mae_std']:.2f} frames")
        print(f"  Duration MAE:       {task2['duration_mae_mean']:.2f} ± {task2['duration_mae_std']:.2f} frames")
        
        print("\n📊 Per-Phase Transition Errors")
        print("-" * 70)
        print(f"{'PHASE NAME':<30} | {'START MAE':<12} | {'END MAE':<12} | {'DURATION MAE':<12}")
        print("-" * 70)
        
        for phase_name, errors in summary['per_phase_errors'].items():
            print(f"{phase_name:<30} | {errors['start_mae']:>10.2f} f | "
                  f"{errors['end_mae']:>10.2f} f | {errors['duration_mae']:>10.2f} f")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate regression tasks')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/labels/aligned_labels',
                        help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results (default: checkpoint directory)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    # Default save directory: same as checkpoint directory
    if args.save_dir is None:
        args.save_dir = Path(args.checkpoint).parent / 'regression_evaluation'
    
    # Create evaluator
    evaluator = RegressionEvaluator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Evaluate
    summary, detailed_results = evaluator.evaluate_all(save_dir=args.save_dir)
    
    print("\n✅ Evaluation completed!")


if __name__ == '__main__':
    main()
