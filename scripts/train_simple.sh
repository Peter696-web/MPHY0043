#!/bin/bash

# 简化的MS-TCN训练脚本

cd /Users/peter/Desktop/MPHY0043

echo "=========================================="
echo "训练MS-TCN++模型（简化参数）"
echo "=========================================="

# 激活环境
export PYTHONPATH=/Users/peter/Desktop/MPHY0043:$PYTHONPATH

# 训练命令（简化参数）
/Users/peter/.local/share/mamba/envs/cholec80_pt/bin/python src/train.py \
    --data_dir data/new_preprocessed/aligned_labels \
    --save_dir results/models/mstcn_simple \
    --channels 64 \
    --stages 4 \
    --layers 10 \
    --epochs 50 \
    --lr 0.0005 \
    --batch_size 4 \
    --seed 42 \
    --workers 0

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: results/models/mstcn_simple"
echo "=========================================="