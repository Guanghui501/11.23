#!/bin/bash
# 详细的参数诊断脚本 - 不重定向输出，显示完整错误

echo "=========================================="
echo "参数诊断测试 (完整输出)"
echo "=========================================="
echo ""

DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

echo "测试配置1: Baseline (No FG + Middle)"
echo "--------------------"

python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection False \
    --early_stopping_patience 150 \
    --output_dir ./test_diagnostic_config1 \
    --num_workers 0 \
    --random_seed 42 2>&1 | head -50

echo ""
echo "如果上面没有错误，说明参数正确"
echo "如果有 'unrecognized arguments' 错误，会显示具体是哪些参数"
