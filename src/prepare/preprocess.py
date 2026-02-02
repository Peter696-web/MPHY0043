"""
Cholec80 Preprocessing for Surgical Phase Remaining Time Prediction

Functions:
1. Downsample 25Hz phase annotations to 1Hz video frames
2. Generate future phase schedule labels (start_offset, duration)
3. Support dual tasks: current phase remaining time + all future phases timeline
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Phase mapping
PHASE_MAPPING = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}

NUM_PHASES = len(PHASE_MAPPING)


class Cholec80Preprocessor:
    """Cholec80 dataset preprocessor"""
    
    def __init__(self, data_root: str, output_dir: str, fps_original: int = 25, fps_target: int = 1):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.fps_original = fps_original
        self.fps_target = fps_target
        self.downsample_ratio = fps_original // fps_target
        
        # Create output directories
        (self.output_dir / 'aligned_labels').mkdir(parents=True, exist_ok=True)
        
    def load_phase_annotations(self, video_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load phase annotations from file"""
        anno_file = self.data_root / 'phase_annotations' / f'{video_id}-phase.txt'
        if not anno_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_file}")
        
        # Read annotations (skip header "Frame\tPhase")
        data = []
        with open(anno_file, 'r') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    frame_id = int(parts[0])
                    phase_id = PHASE_MAPPING.get(parts[1], -1)
                    data.append((frame_id, phase_id))
        
        frame_ids = np.array([d[0] for d in data])
        phases = np.array([d[1] for d in data])
        return frame_ids, phases
    
    def downsample_labels(self, frames: np.ndarray, phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample labels from 25Hz to 1Hz using majority voting"""
        num_target_frames = len(frames) // self.downsample_ratio
        downsampled_frames = []
        downsampled_phases = []
        
        for i in range(num_target_frames):
            start_idx = i * self.downsample_ratio
            end_idx = start_idx + self.downsample_ratio
            window_phases = phases[start_idx:end_idx]
            
            # Majority voting
            unique, counts = np.unique(window_phases, return_counts=True)
            majority_phase = unique[np.argmax(counts)]
            
            downsampled_frames.append(i + 1)  # Frame numbering starts from 1
            downsampled_phases.append(majority_phase)
        
        return np.array(downsampled_frames), np.array(downsampled_phases)
    
    def extract_phase_segments(self, phases: np.ndarray) -> List[Dict]:
        """Extract phase segments from label sequence"""
        segments = []
        current_phase = phases[0]
        start_frame = 0
        
        for i in range(1, len(phases)):
            if phases[i] != current_phase:
                # Phase transition
                segments.append({
                    'phase_id': int(current_phase),
                    'phase_name': [k for k, v in PHASE_MAPPING.items() if v == current_phase][0],
                    'start_frame': int(start_frame),
                    'end_frame': int(i - 1),
                    'duration': int(i - start_frame)
                })
                current_phase = phases[i]
                start_frame = i
        
        # Add last segment
        segments.append({
            'phase_id': int(current_phase),
            'phase_name': [k for k, v in PHASE_MAPPING.items() if v == current_phase][0],
            'start_frame': int(start_frame),
            'end_frame': int(len(phases) - 1),
            'duration': int(len(phases) - start_frame)
        })
        
        return segments
    
    def compute_remaining_time_labels(self, phases: np.ndarray, segments: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Compute remaining time labels for each frame
        
        Returns dict with:
            - phase_id: (N,) current phase ID
            - future_schedule: (N, 7, 2) future phase schedule
                [:, phase_id, 0] = seconds until start (0=ongoing, -1=completed)
                [:, phase_id, 1] = phase duration in seconds (-1=completed)
        """
        num_frames = len(phases)
        labels = {
            'phase_id': phases.copy(),
            'future_schedule': np.full((num_frames, NUM_PHASES, 2), -1, dtype=np.float32),
        }
        
        # Frame to segment mapping
        frame_to_segment = {}
        for seg_idx, seg in enumerate(segments):
            for frame in range(seg['start_frame'], seg['end_frame'] + 1):
                frame_to_segment[frame] = seg_idx
        
        # Compute future schedule for each frame
        for frame_idx in range(num_frames):
            for phase_id in range(NUM_PHASES):
                phase_segments = [s for s in segments if s['phase_id'] == phase_id]
                
                if not phase_segments:
                    labels['future_schedule'][frame_idx, phase_id, :] = -1
                    continue
                
                phase_seg = phase_segments[0]
                
                if frame_idx < phase_seg['start_frame']:
                    # Future phase
                    start_offset = phase_seg['start_frame'] - frame_idx
                    duration = phase_seg['duration']
                    labels['future_schedule'][frame_idx, phase_id, 0] = start_offset
                    labels['future_schedule'][frame_idx, phase_id, 1] = duration
                    
                elif phase_seg['start_frame'] <= frame_idx <= phase_seg['end_frame']:
                    # Ongoing phase
                    remaining = phase_seg['end_frame'] - frame_idx + 1
                    labels['future_schedule'][frame_idx, phase_id, 0] = 0
                    labels['future_schedule'][frame_idx, phase_id, 1] = remaining
                    
                else:
                    # Completed phase
                    labels['future_schedule'][frame_idx, phase_id, :] = -1
        
        return labels
    
    def process_video(self, video_id: str, save_npy: bool = True) -> Dict:
        """
        Process a single video through the entire pipeline
        
        Args:
            video_id: Video ID, e.g. 'video01'
            save_npy: Whether to save as .npy file

        Returns:
            result: A dictionary containing all processing results
        """
        print(f"\n{'='*60}")
        print(f"process video: {video_id}")
        print(f"{'='*60}")
        
        # 1. Load original labels
        print(f"[1/4] Load original frame (25Hz)...")
        frames_25hz, phases_25hz = self.load_phase_annotations(video_id)
        print(f"  - Original frame: {len(frames_25hz)}")
        print(f"  - Original Frame Range: frame {frames_25hz[0]} - {frames_25hz[-1]}")
        
        # 2. Downsample to 1Hz
        print(f"[2/4] Downsample (25Hz -> 1Hz)...")
        frames_1hz, phases_1hz = self.downsample_labels(frames_25hz, phases_25hz)
        print(f"  - After Downsampling: {len(frames_1hz)}")
        print(f"  - Frame Range: {frames_1hz[0]} - {frames_1hz[-1]}")
        
        # 3. Extract phase segments
        print(f"[3/4] Extract phase segments...")
        segments = self.extract_phase_segments(phases_1hz)
        print(f"  - Phase segments: {len(segments)}")
        for seg in segments:
            print(f"    * {seg['phase_name']:30s} | Frames {seg['start_frame']:4d}-{seg['end_frame']:4d} | Duration: {seg['duration']:4d}")
        
        # 4. Compute future phase schedule labels
        print(f"[4/4] Future Phase Schedule...")
        labels = self.compute_remaining_time_labels(phases_1hz, segments)
        print(f"  - Generated label dimensions:")
        for key, value in labels.items():
            if isinstance(value, np.ndarray):
                print(f"    * {key:35s}: {value.shape}")
        
        # Build result dictionary
        result = {
            'video_id': video_id,
            'num_frames': len(frames_1hz),
            'total_duration_sec': len(frames_1hz),
            'frame_ids': frames_1hz,
            'segments': segments,
            'labels': labels,
            'metadata': {
                'fps_original': self.fps_original,
                'fps_target': self.fps_target,
                'downsample_ratio': self.downsample_ratio,
                'num_phases': NUM_PHASES,
                'phase_mapping': PHASE_MAPPING
            }
        }
        
        # Save results
        if save_npy:
            self._save_results(video_id, result)
        
        print(f"✓ {video_id} processing complete!")
        return result
    
    def _save_results(self, video_id: str, result: Dict):
        """Save processing results (JSON + NPY only)"""
        # Save JSON (metadata + segments) for reading
        json_output = {
            'video_id': result['video_id'],
            'num_frames': result['num_frames'],
            'total_duration_sec': result['total_duration_sec'],
            'segments': result['segments'],
            'metadata': result['metadata']
        }
        
        json_path = self.output_dir / 'aligned_labels' / f'{video_id}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"  → Saved JSON: {json_path}")
        
        # Save labels as NPY (for training)
        npy_path = self.output_dir / 'aligned_labels' / f'{video_id}_labels.npy'
        np.save(npy_path, result['labels'], allow_pickle=True)
        print(f"  → Saved NPY: {npy_path}")
    
    def process_all_videos(self, video_ids: List[str] = None):
        """Process all videos in batch"""
        if video_ids is None:
            phase_dir = self.data_root / 'phase_annotations'
            video_ids = sorted([
                f.stem.replace('-phase', '') 
                for f in phase_dir.glob('*-phase.txt')
            ])
        
        print(f"\n{'='*60}")
        print(f"Starting batch processing for {len(video_ids)} videos")
        print(f"{'='*60}")
        
        all_results = {}
        
        for video_id in video_ids:
            try:
                result = self.process_video(video_id, save_npy=True)
                all_results[video_id] = result
            except Exception as e:
                print(f"✗ Failed to process {video_id}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ All videos processed! Success: {len(all_results)}/{len(video_ids)}")
        print(f"{'='*60}\n")
        
        return all_results


def main():
    """Main function: run preprocessing"""
    DATA_ROOT = './cholec80'
    OUTPUT_DIR = './data/label'
    
    preprocessor = Cholec80Preprocessor(
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR,
        fps_original=25,
        fps_target=1
    )
    
    # Process all 80 videos
    results = preprocessor.process_all_videos()
    
    print("\nPreprocessing complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - aligned_labels/: Aligned label files (JSON + NPY)")


if __name__ == '__main__':
    main()
