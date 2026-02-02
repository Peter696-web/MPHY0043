#!/bin/bash
# MS-TCN++ Training Script - Clean and Simple
# Only saves best model, simplified metrics

cd "$(dirname "$0")/.."

echo "🚀 Starting MS-TCN++ Training"
echo "================================"

python src/train.py \
    --model_type mstcn \
    --data_dir data/processed \
    --save_dir results/models/mstcn_clean \
    \
    --feature_dim 768 \
    --mstcn_channels 64 \
    --mstcn_stages 4 \
    --mstcn_layers 10 \
    --dropout 0.5 \
    --num_phases 7 \
    \
    --loss_alpha 1.0 \
    --loss_beta 1.0 \
    --loss_gamma 0.15 \
    \
    --num_epochs 50 \
    --batch_size 1 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --grad_clip 5.0 \
    --lr_patience 10 \
    \
    --num_workers 0 \
    --seed 42 \
    \
    --print_freq 10

echo ""
echo "✅ Training completed!"
echo "📊 Results saved to: results/models/mstcn_clean/"
