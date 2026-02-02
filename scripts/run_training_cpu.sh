#!/bin/bash
# Training script: bash scripts/run_training_cpu.sh [lstm|transformer]
# Sequence-only training; batch_size fixed to 1

MODEL=${1:-lstm}
BATCH=1

echo "Training ${MODEL^^} | Batch=${BATCH} | Normalized Features | sequence_mode=true"

eval "$(micromamba shell hook --shell bash)"


python src/train.py \
    --model_type ${MODEL} \
    --batch_size ${BATCH} \
    --num_workers 0 \
    --num_epochs 50 \
    --save_dir results/models/${MODEL}_b${BATCH}_norm

echo "Done: results/models/${MODEL}_b${BATCH}_norm"
