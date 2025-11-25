#!/bin/bash
# 优化版训练脚本 - 带 DynamicFusionModule 权重监控

# 创建日志文件名（包含时间戳）
LOG_FILE="./output_100epochs_7_bs128_sw_ju_dynamic/train_$(date +%Y%m%d_%H%M%S).log"

# 确保输出目录存在
mkdir -p ./output_100epochs_7_bs128_sw_ju_dynamic

echo "=========================================="
echo "开始 DynamicFusionModule 训练"
echo "数据集: JARVIS - MBJ Band Gap"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "关键配置:"
echo "  - DynamicFusionModule: 启用 (layer 2)"
echo "  - 细粒度注意力: 启用"
echo "  - 跨模态注意力: 启用"
echo "  - 权重监控: 自动启用 (每5个epoch)"
echo ""
echo "=========================================="

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

nohup python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 128 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --warmup_steps 2000 \
    --early_stopping_patience 150 \
    \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_num_heads 2 \
    --middle_fusion_dropout 0.1 \
    \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection True \
    \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --cross_modal_dropout 0.1 \
    \
    --output_dir ./output_100epochs_7_bs128_sw_ju_dynamic \
    --num_workers 24 \
    --random_seed 7 \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "=========================================="
echo "✅ 训练已在后台启动"
echo "=========================================="
echo ""
echo "PID: $PID"
echo ""
echo "📊 监控命令:"
echo "  查看日志:      tail -f $LOG_FILE"
echo "  查看权重监控:  grep 'DynamicFusionModule Weight' $LOG_FILE"
echo "  查看进度:      grep 'Epoch:' $LOG_FILE | tail -20"
echo ""
echo "📁 输出文件:"
echo "  模型检查点:    ./output_100epochs_7_bs128_sw_ju_dynamic/mbj_bandgap/*.pt"
echo "  权重日志:      ./output_100epochs_7_bs128_sw_ju_dynamic/mbj_bandgap/fusion_weights.csv"
echo "  训练历史:      ./output_100epochs_7_bs128_sw_ju_dynamic/mbj_bandgap/history_*.json"
echo ""
echo "🔍 分析权重:"
echo "  python analyze_fusion_weights.py --output_dir ./output_100epochs_7_bs128_sw_ju_dynamic/mbj_bandgap/"
echo ""
echo "=========================================="

# 保存 PID 以便后续管理
echo $PID > ./output_100epochs_7_bs128_sw_ju_dynamic/train.pid
