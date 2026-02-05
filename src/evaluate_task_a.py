"""
优化的回归任务评估脚本 - 直观回应Task A

Task A的两个问题：
1. Predict the remaining time of the current surgical phase
2. Estimate the start and end times of all upcoming phases

输出设计：直接显示预测结果，而不仅仅是误差
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from model.data import create_dataloaders
from model.mstcn import MSTCNSurgicalPredictor


class TaskAEvaluator:
    """Task A评估器 - 直观输出预测结果"""
    
    @staticmethod
    def format_time(seconds):
        """将秒转换为 分:秒 格式"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"
    
    def __init__(self, checkpoint_path, data_dir, device='cuda'):
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
            batch_size=1,
            num_workers=0,
            normalize_features=self.config.get('normalize_features', True),
            normalize_schedule=self.config.get('normalize_schedule', True),
        )
        
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
        """反归一化时间表"""
        return schedule * video_length
    
    def evaluate_video_at_frame(self, pred_schedule, true_schedule, true_phases, 
                                 frame_idx, video_length):
        """
        在特定帧评估Task A的两个问题
        
        Args:
            pred_schedule: (T, 7, 2) 预测的时间表
            true_schedule: (T, 7, 2) 真实的时间表
            true_phases: (T,) 真实的相位ID
            frame_idx: 当前帧索引
            video_length: 视频总长度
            
        Returns:
            dict: Task A的两个答案
        """
        current_phase = int(true_phases[frame_idx])
        pred_sched = pred_schedule[frame_idx]  # (7, 2)
        true_sched = true_schedule[frame_idx]  # (7, 2)
        
        # ========================================
        # Task 1: 预测当前相位的剩余时间
        # ========================================
        pred_start = float(pred_sched[current_phase, 0])
        pred_duration = float(pred_sched[current_phase, 1])
        pred_end = pred_start + pred_duration
        pred_remaining = max(0.0, pred_end - frame_idx)
        
        true_start = float(true_sched[current_phase, 0])
        true_duration = float(true_sched[current_phase, 1])
        true_end = true_start + true_duration
        true_remaining = max(0.0, true_end - frame_idx)
        
        task1_result = {
            'current_frame': int(frame_idx),
            'current_phase_id': int(current_phase),
            'current_phase_name': self.phase_names[current_phase],
            'predicted_remaining_frames': float(pred_remaining),
            'true_remaining_frames': float(true_remaining),
            'error_frames': float(abs(pred_remaining - true_remaining)),
            'predicted_phase_will_end_at_frame': float(pred_end),
            'true_phase_will_end_at_frame': float(true_end)
        }
        
        # ========================================
        # Task 2: 估计所有未来相位的开始和结束时间
        # ========================================
        all_phases_prediction = []
        all_phases_groundtruth = []
        
        for phase_id in range(7):
            # 预测
            pred_phase_start = float(pred_sched[phase_id, 0])
            pred_phase_duration = float(pred_sched[phase_id, 1])
            pred_phase_end = pred_phase_start + pred_phase_duration
            
            # 真实
            true_phase_start = float(true_sched[phase_id, 0])
            true_phase_duration = float(true_sched[phase_id, 1])
            true_phase_end = true_phase_start + true_phase_duration
            
            # 计算误差
            start_error = abs(pred_phase_start - true_phase_start)
            end_error = abs(pred_phase_end - true_phase_end)
            duration_error = abs(pred_phase_duration - true_phase_duration)
            
            # 判断是否已发生、正在发生、或未来
            if frame_idx < true_phase_start:
                status = 'future'
            elif frame_idx >= true_phase_end:
                status = 'past'
            else:
                status = 'current'
            
            phase_info = {
                'phase_id': int(phase_id),
                'phase_name': self.phase_names[phase_id],
                'status': status,
                'prediction': {
                    'start_frame': float(pred_phase_start),
                    'end_frame': float(pred_phase_end),
                    'duration_frames': float(pred_phase_duration)
                },
                'ground_truth': {
                    'start_frame': float(true_phase_start),
                    'end_frame': float(true_phase_end),
                    'duration_frames': float(true_phase_duration)
                },
                'errors': {
                    'start_mae': float(start_error),
                    'end_mae': float(end_error),
                    'duration_mae': float(duration_error)
                }
            }
            
            all_phases_prediction.append(phase_info)
        
        task2_result = {
            'current_frame': int(frame_idx),
            'all_phases': all_phases_prediction
        }
        
        return {
            'task1_remaining_time': task1_result,
            'task2_phase_schedule': task2_result
        }
    
    def evaluate_video(self, batch, sample_frames=5):
        """
        评估一个视频
        
        Args:
            batch: 数据批次
            sample_frames: 采样多少个关键帧进行详细评估
        """
        # Extract data
        features = batch['features'].to(self.device)
        true_phases = batch['phase_id'][0].numpy()
        true_schedule = batch['future_schedule'][0].numpy()
        video_id = batch['video_id'][0].item()
        video_length = features.shape[1]
        
        # Predict
        with torch.no_grad():
            predictions = self.model(features)
            pred_schedule = predictions['future_schedule'][0].cpu().numpy()
        
        # Denormalize
        if self.config.get('normalize_schedule', True):
            pred_schedule = self.denormalize_schedule(pred_schedule, video_length)
            true_schedule = self.denormalize_schedule(true_schedule, video_length)
        
        # ========================================
        # 计算整体统计 (所有帧)
        # ========================================
        all_remaining_errors = []
        phase_transition_errors = defaultdict(lambda: {'start': [], 'end': [], 'duration': []})
        
        for t in range(video_length):
            current_phase = int(true_phases[t])
            
            # Task 1: 剩余时间误差
            pred_start = pred_schedule[t, current_phase, 0]
            pred_duration = pred_schedule[t, current_phase, 1]
            pred_end = pred_start + pred_duration
            pred_remaining = max(0.0, pred_end - t)
            
            true_start = true_schedule[t, current_phase, 0]
            true_duration = true_schedule[t, current_phase, 1]
            true_end = true_start + true_duration
            true_remaining = max(0.0, true_end - t)
            
            all_remaining_errors.append(abs(pred_remaining - true_remaining))
            
            # Task 2: 相位转换误差 (所有相位)
            for phase_id in range(7):
                pred_start = pred_schedule[t, phase_id, 0]
                pred_duration = pred_schedule[t, phase_id, 1]
                pred_end = pred_start + pred_duration
                
                true_start = true_schedule[t, phase_id, 0]
                true_duration = true_schedule[t, phase_id, 1]
                true_end = true_start + true_duration
                
                phase_transition_errors[phase_id]['start'].append(abs(pred_start - true_start))
                phase_transition_errors[phase_id]['end'].append(abs(pred_end - true_end))
                phase_transition_errors[phase_id]['duration'].append(abs(pred_duration - true_duration))
        
        # ========================================
        # 采样关键帧进行详细评估
        # ========================================
        sample_indices = np.linspace(0, video_length-1, sample_frames, dtype=int)
        detailed_frames = []
        
        for frame_idx in sample_indices:
            frame_result = self.evaluate_video_at_frame(
                pred_schedule, true_schedule, true_phases,
                frame_idx, video_length
            )
            detailed_frames.append(frame_result)
        
        # ========================================
        # 汇总结果
        # ========================================
        per_phase_errors = {}
        for phase_id in range(7):
            if phase_transition_errors[phase_id]['start']:
                per_phase_errors[self.phase_names[phase_id]] = {
                    'start_mae': float(np.mean(phase_transition_errors[phase_id]['start'])),
                    'end_mae': float(np.mean(phase_transition_errors[phase_id]['end'])),
                    'duration_mae': float(np.mean(phase_transition_errors[phase_id]['duration']))
                }
        
        video_result = {
            'video_id': int(video_id),
            'video_length': int(video_length),
            
            # 整体统计
            'overall_statistics': {
                'task1_remaining_time_mae': float(np.mean(all_remaining_errors)),
                'per_phase_errors': per_phase_errors
            },
            
            # 关键帧详细预测
            'sampled_frames': detailed_frames
        }
        
        return video_result
    
    def evaluate_all(self, save_dir=None):
        """评估所有测试视频"""
        print("\n" + "="*70)
        print("Task A Evaluation: Direct Prediction Results")
        print("="*70)
        
        all_videos = []
        
        # 统计
        all_task1_mae = []
        all_phase_errors = defaultdict(lambda: {'start': [], 'end': [], 'duration': []})
        
        for batch in self.test_loader:
            video_result = self.evaluate_video(batch, sample_frames=5)
            all_videos.append(video_result)
            
            # 累积统计
            all_task1_mae.append(video_result['overall_statistics']['task1_remaining_time_mae'])
            
            for phase_name, errors in video_result['overall_statistics']['per_phase_errors'].items():
                all_phase_errors[phase_name]['start'].append(errors['start_mae'])
                all_phase_errors[phase_name]['end'].append(errors['end_mae'])
                all_phase_errors[phase_name]['duration'].append(errors['duration_mae'])
        
        # 汇总所有视频
        summary = {
            'epoch': self.epoch,
            'num_videos': len(all_videos),
            
            'task1_remaining_time': {
                'mae_mean': float(np.mean(all_task1_mae)),
                'mae_std': float(np.std(all_task1_mae))
            },
            
            'task2_phase_transitions': {
                phase_name: {
                    'start_mae': float(np.mean(errors['start'])),
                    'end_mae': float(np.mean(errors['end'])),
                    'duration_mae': float(np.mean(errors['duration']))
                }
                for phase_name, errors in all_phase_errors.items()
            }
        }
        
        # 打印汇总
        self.print_summary(summary)
        
        # 打印示例预测
        self.print_example_predictions(all_videos[0])
        
        # 保存结果
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_dir / 'task_a_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            with open(save_dir / 'task_a_detailed.json', 'w') as f:
                json.dump(all_videos, f, indent=2)
            
            print(f"\n✅ Results saved to: {save_dir}")
        
        return summary, all_videos
    
    def print_summary(self, summary):
        """打印汇总统计"""
        print("\n" + "="*70)
        print(f"SUMMARY STATISTICS (Epoch {summary['epoch']})")
        print("="*70)
        
        print("\n📊 Task 1: Current Phase Remaining Time")
        print("-" * 70)
        task1 = summary['task1_remaining_time']
        print(f"  Overall MAE: {task1['mae_mean']:.2f} ± {task1['mae_std']:.2f} seconds")
        print(f"  (平均预测误差约 {task1['mae_mean']:.1f}秒 = {self.format_time(task1['mae_mean'])})")
        
        print("\n📊 Task 2: All Phases Start/End Time Prediction")
        print("-" * 70)
        print(f"{'Phase Name':<30} | {'Start MAE':<15} | {'End MAE':<15} | {'Duration MAE':<15}")
        print("-" * 70)
        
        for phase_name, errors in summary['task2_phase_transitions'].items():
            print(f"{phase_name:<30} | {errors['start_mae']:>8.1f}s ({self.format_time(errors['start_mae']):>5}) | "
                  f"{errors['end_mae']:>8.1f}s ({self.format_time(errors['end_mae']):>5}) | "
                  f"{errors['duration_mae']:>8.1f}s ({self.format_time(errors['duration_mae']):>5})")
        
        print("="*70)
    
    def print_example_predictions(self, video_result):
        """打印一个视频的示例预测"""
        print("\n" + "="*70)
        print(f"EXAMPLE PREDICTIONS (Video {video_result['video_id']}, "
              f"Length: {video_result['video_length']}s = {self.format_time(video_result['video_length'])})")
        print("="*70)
        
        # 取中间的一个采样帧
        mid_sample = video_result['sampled_frames'][len(video_result['sampled_frames']) // 2]
        
        # Task 1
        print("\n🎯 Task 1: Current Phase Remaining Time")
        print("-" * 70)
        task1 = mid_sample['task1_remaining_time']
        print(f"  Current Time: Frame {task1['current_frame']} = {self.format_time(task1['current_frame'])}")
        print(f"  Current Phase: {task1['current_phase_name']}")
        print(f"  ✅ Predicted Remaining: {task1['predicted_remaining_frames']:.1f}s "
              f"(~{self.format_time(task1['predicted_remaining_frames'])})")
        print(f"  📌 True Remaining:      {task1['true_remaining_frames']:.1f}s "
              f"(~{self.format_time(task1['true_remaining_frames'])})")
        print(f"  ❌ Error:               {task1['error_frames']:.1f}s")
        
        # Task 2
        print("\n🎯 Task 2: All Phases Schedule (Absolute Time from Video Start)")
        print("-" * 70)
        print(f"{'Phase':<25} | {'Status':<10} | {'Pred Start':<18} | {'Pred End':<18} | {'Duration':<18}")
        print("-" * 70)
        
        for phase in mid_sample['task2_phase_schedule']['all_phases']:
            pred = phase['prediction']
            status_icon = {'future': '⏳', 'current': '▶️', 'past': '✅'}[phase['status']]
            print(f"{phase['phase_name']:<25} | {status_icon} {phase['status']:<8} | "
                  f"{pred['start_frame']:>6.0f}s ({self.format_time(pred['start_frame']):>5}) | "
                  f"{pred['end_frame']:>6.0f}s ({self.format_time(pred['end_frame']):>5}) | "
                  f"{pred['duration_frames']:>6.0f}s ({self.format_time(pred['duration_frames']):>5})")
        
        print("="*70)
        print("\n💡 说明: 1 frame = 1 second (数据已下采样到1Hz)")
        print("💡 所有时间都是从视频开始的绝对时间戳")


def main():
    parser = argparse.ArgumentParser(description='Task A Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = Path(args.checkpoint).parent / 'task_a_evaluation'
    
    evaluator = TaskAEvaluator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )
    
    summary, detailed = evaluator.evaluate_all(save_dir=args.save_dir)
    
    print("\n✅ Task A Evaluation completed!")
    print(f"\n💡 查看详细预测结果:")
    print(f"   {args.save_dir}/task_a_detailed.json")


if __name__ == '__main__':
    main()
