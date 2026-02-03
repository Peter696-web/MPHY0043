#!/bin/bash
set -e

# 1). Timeline Visualization
python src/utils/visualize_timeline.py \
    --pred_path results/models/mstcn_phase_aware_v2/val_predictions_best_epoch41.pt \
    --video_id 61 # Specify the video ID to visualize

# 2). Remaining Time Visualization
python src/utils/visualize_video.py \
    --pred_path results/models/mstcn_phase_aware_v2/val_predictions_best_epoch41.pt \
    --video_id 61 # Specify the video ID to visualize

echo "Visualization completed."