#!/bin/bash

python src/train.py \
    --model_type mstcn \
    --save_dir results/models/mstcn_phase_aware_v2 \
    --num_workers 0
