#!/bin/bash

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 基础参数设置
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
BATCH_SIZE=128
EPOCHS=100
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-4
WARMUP_STEPS=2000
ALIGNN_LAYERS=4
GCN_LAYERS=4
HIDDEN_FEATURES=256
GRAPH_DROPOUT=0.15
RANDOM_SEED=42
NUM_WORKERS=24

echo "=========================================="
echo "  融合机制对比实验 - 依次执行"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  - 数据集: $DATASET"
echo "  - 属性: $PROPERTY"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 训练轮数: $EPOCHS"
echo "  - 随机种子: $RANDOM_SEED"
echo ""
echo "将依次运行三个实验:"
echo "  1. 中期融合 → 图预测"
echo "  2. 中期融合 + 细粒度注意力 → 图预测"
echo "  3. 中期融合 + 跨模态注意力 → 图预测"
echo ""
echo "=========================================="
echo ""

# ==================== 实验1: 只用中期融合 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 实验1/3: 中期融合 → 图预测"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_DIR_1="./output_100epochs_42_bs128_middle_fusion_only"
LOG_FILE_1="$OUTPUT_DIR_1/train_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR_1"

echo "配置: 中期融合层2"
echo "输出目录: $OUTPUT_DIR_1"
echo "日志文件: $LOG_FILE_1"
echo ""
echo "开始训练..."
echo ""

python train_with_cross_modal_attention.py \
    --root_dir "$ROOT_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --alignn_layers "$ALIGNN_LAYERS" \
    --gcn_layers "$GCN_LAYERS" \
    --hidden_features "$HIDDEN_FEATURES" \
    --graph_dropout "$GRAPH_DROPOUT" \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --use_only_graph_for_prediction True \
    --early_stopping_patience 150 \
    --output_dir "$OUTPUT_DIR_1" \
    --num_workers "$NUM_WORKERS" \
    --random_seed "$RANDOM_SEED" \
    2>&1 | tee "$LOG_FILE_1"

# 检查第一个实验是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ 实验1完成!"
    echo ""
else
    echo ""
    echo "❌ 实验1失败! 终止后续实验。"
    exit 1
fi

sleep 5

# ==================== 实验2: 中期融合 + 细粒度注意力 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 实验2/3: 中期融合 + 细粒度注意力 → 图预测"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_DIR_2="./output_100epochs_42_bs128_middle_fine_grained"
LOG_FILE_2="$OUTPUT_DIR_2/train_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR_2"

echo "配置: 中期融合层2 + 细粒度注意力(8头)"
echo "输出目录: $OUTPUT_DIR_2"
echo "日志文件: $LOG_FILE_2"
echo ""
echo "开始训练..."
echo ""

python train_with_cross_modal_attention.py \
    --root_dir "$ROOT_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --alignn_layers "$ALIGNN_LAYERS" \
    --gcn_layers "$GCN_LAYERS" \
    --hidden_features "$HIDDEN_FEATURES" \
    --graph_dropout "$GRAPH_DROPOUT" \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection True \
    --use_only_graph_for_prediction True \
    --early_stopping_patience 150 \
    --output_dir "$OUTPUT_DIR_2" \
    --num_workers "$NUM_WORKERS" \
    --random_seed "$RANDOM_SEED" \
    2>&1 | tee "$LOG_FILE_2"

# 检查第二个实验是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ 实验2完成!"
    echo ""
else
    echo ""
    echo "❌ 实验2失败! 终止后续实验。"
    exit 1
fi

sleep 5

# ==================== 实验3: 中期融合 + 跨模态注意力 ====================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 实验3/3: 中期融合 + 跨模态注意力 → 图预测"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

OUTPUT_DIR_3="./output_100epochs_42_bs128_middle_cross_modal"
LOG_FILE_3="$OUTPUT_DIR_3/train_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR_3"

echo "配置: 中期融合层2 + 跨模态注意力(4头)"
echo "输出目录: $OUTPUT_DIR_3"
echo "日志文件: $LOG_FILE_3"
echo ""
echo "开始训练..."
echo ""

python train_with_cross_modal_attention.py \
    --root_dir "$ROOT_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --alignn_layers "$ALIGNN_LAYERS" \
    --gcn_layers "$GCN_LAYERS" \
    --hidden_features "$HIDDEN_FEATURES" \
    --graph_dropout "$GRAPH_DROPOUT" \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --use_only_graph_for_prediction True \
    --early_stopping_patience 150 \
    --output_dir "$OUTPUT_DIR_3" \
    --num_workers "$NUM_WORKERS" \
    --random_seed "$RANDOM_SEED" \
    2>&1 | tee "$LOG_FILE_3"

# 检查第三个实验是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ 实验3完成!"
    echo ""
else
    echo ""
    echo "❌ 实验3失败!"
    exit 1
fi

# ==================== 所有实验完成 ====================
echo ""
echo "=========================================="
echo "  🎉 所有实验完成!"
echo "=========================================="
echo ""
echo "实验结果汇总:"
echo ""
echo "实验1 - 中期融合:"
echo "  输出目录: $OUTPUT_DIR_1"
echo "  日志文件: $LOG_FILE_1"
echo ""
echo "实验2 - 中期融合 + 细粒度:"
echo "  输出目录: $OUTPUT_DIR_2"
echo "  日志文件: $LOG_FILE_2"
echo ""
echo "实验3 - 中期融合 + 跨模态:"
echo "  输出目录: $OUTPUT_DIR_3"
echo "  日志文件: $LOG_FILE_3"
echo ""
echo "=========================================="
echo ""
echo "📊 查看结果对比:"
echo ""
echo "# 实验1最佳MAE:"
echo "grep 'Best test MAE' $LOG_FILE_1 | tail -1"
echo ""
echo "# 实验2最佳MAE:"
echo "grep 'Best test MAE' $LOG_FILE_2 | tail -1"
echo ""
echo "# 实验3最佳MAE:"
echo "grep 'Best test MAE' $LOG_FILE_3 | tail -1"
echo ""
echo "=========================================="
