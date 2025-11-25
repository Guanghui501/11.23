#!/bin/bash
# 快速测试单个融合层配置

# 检查参数
if [ "$#" -ne 1 ]; then
    echo "用法: $0 <fusion_layers>"
    echo "示例: $0 \"2\""
    echo "示例: $0 \"2,3\""
    exit 1
fi

FUSION_LAYERS="$1"

echo "=========================================="
echo "快速测试: Fusion Layers = $FUSION_LAYERS"
echo "=========================================="

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

CONFIG_NAME="test_layers_${FUSION_LAYERS//,/_}"
OUTPUT_DIR="./quick_test/$CONFIG_NAME"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --n_train 500 \
    --n_val 50 \
    --n_test 50 \
    --batch_size 64 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --warmup_steps 500 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_middle_fusion True \
    --middle_fusion_layers "$FUSION_LAYERS" \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_num_heads 2 \
    --middle_fusion_dropout 0.1 \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --early_stopping_patience 50 \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 24 \
    --random_seed 123 \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "✅ 训练已启动"
echo "PID: $PID"
echo "日志: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  grep 'Val_MAE' $LOG_FILE | tail -20"
echo ""
