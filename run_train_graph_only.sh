#!/bin/bash

# 设置日志文件
LOG_FILE="./training_graph_only_$(date +%Y%m%d_%H%M%S).log"

nohup python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 128 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection True \
    --use_only_graph_for_prediction True \
    --early_stopping_patience 150 \
    --output_dir ./output_100epochs_7_bs128_sw_ju_graph_only \
    --num_workers 24 \
    --random_seed 7 \
    > "$LOG_FILE" 2>&1 &

echo "训练已启动，日志文件: $LOG_FILE"
echo "使用 tail -f $LOG_FILE 查看训练进度"
