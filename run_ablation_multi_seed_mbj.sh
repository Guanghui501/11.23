#!/bin/bash

# ============================================================================
# 消融实验自动化脚本（多种子版本 - 串行执行）
# 运行4个实验配置 × 3个随机种子 = 12个训练任务
# 任务一个接一个执行，避免GPU资源冲突
# ============================================================================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 基础配置
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
BASE_OUTPUT_DIR="./ablation_multi_seed_mbj"

# 训练超参数（与用户提供的完全一致）
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-4
WARMUP_STEPS=2000
ALIGNN_LAYERS=4
GCN_LAYERS=4
HIDDEN_FEATURES=256
GRAPH_DROPOUT=0.15
CROSS_MODAL_NUM_HEADS=4
MIDDLE_FUSION_LAYERS=2
FINE_GRAINED_HIDDEN_DIM=256
FINE_GRAINED_NUM_HEADS=8
FINE_GRAINED_DROPOUT=0.2
FINE_GRAINED_USE_PROJECTION=True
EARLY_STOPPING_PATIENCE=150
NUM_WORKERS=24

# 随机种子列表
SEEDS=(42 123 7)

# 公共参数
COMMON_ARGS="
    --root_dir $ROOT_DIR \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --alignn_layers $ALIGNN_LAYERS \
    --gcn_layers $GCN_LAYERS \
    --hidden_features $HIDDEN_FEATURES \
    --graph_dropout $GRAPH_DROPOUT \
    --cross_modal_num_heads $CROSS_MODAL_NUM_HEADS \
    --middle_fusion_layers $MIDDLE_FUSION_LAYERS \
    --fine_grained_hidden_dim $FINE_GRAINED_HIDDEN_DIM \
    --fine_grained_num_heads $FINE_GRAINED_NUM_HEADS \
    --fine_grained_dropout $FINE_GRAINED_DROPOUT \
    --fine_grained_use_projection $FINE_GRAINED_USE_PROJECTION \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --num_workers $NUM_WORKERS
"

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 主日志文件
MAIN_LOG="$BASE_OUTPUT_DIR/launch_log_$(date +%Y%m%d_%H%M%S).txt"

echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🚀 启动消融实验（多种子版本 - 串行执行）" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "时间: $(date)" | tee -a "$MAIN_LOG"
echo "数据集: $DATASET/$PROPERTY" | tee -a "$MAIN_LOG"
echo "实验配置: 4个实验 × 3个种子 = 12个训练任务" | tee -a "$MAIN_LOG"
echo "执行模式: 串行（一个接一个）" | tee -a "$MAIN_LOG"
echo "随机种子: ${SEEDS[@]}" | tee -a "$MAIN_LOG"
echo "基础输出目录: $BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 用于统计完成任务
COMPLETED_COUNT=0
FAILED_COUNT=0

# ============================================================================
# 实验函数：启动单个训练任务（串行执行）
# ============================================================================
run_experiment() {
    local exp_name=$1
    local exp_num=$2
    local seed=$3
    local use_cross_modal=$4
    local use_middle_fusion=$5
    local use_fine_grained=$6

    local output_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"
    local log_file="$output_dir/training.log"

    # 创建输出目录
    mkdir -p "$output_dir"

    echo "============================================================================" | tee -a "$MAIN_LOG"
    echo "[$((COMPLETED_COUNT + FAILED_COUNT + 1))/12] 运行: $exp_name (seed=$seed)" | tee -a "$MAIN_LOG"
    echo "============================================================================" | tee -a "$MAIN_LOG"
    echo "  开始时间: $(date)" | tee -a "$MAIN_LOG"
    echo "  输出目录: $output_dir" | tee -a "$MAIN_LOG"
    echo "  配置: cross_modal=$use_cross_modal, middle_fusion=$use_middle_fusion, fine_grained=$use_fine_grained" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"

    # 同步执行训练（等待完成）
    python train_with_cross_modal_attention.py \
        $COMMON_ARGS \
        --random_seed $seed \
        --use_cross_modal $use_cross_modal \
        --use_middle_fusion $use_middle_fusion \
        --use_fine_grained_attention $use_fine_grained \
        --output_dir "$output_dir" \
        2>&1 | tee "$log_file"

    # 检查退出状态
    local exit_code=${PIPESTATUS[0]}

    echo "" | tee -a "$MAIN_LOG"
    if [ $exit_code -eq 0 ]; then
        echo "✅ 完成: $exp_name (seed=$seed)" | tee -a "$MAIN_LOG"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    else
        echo "❌ 失败: $exp_name (seed=$seed) - 退出码: $exit_code" | tee -a "$MAIN_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    echo "  结束时间: $(date)" | tee -a "$MAIN_LOG"
    echo "  已完成: $COMPLETED_COUNT, 失败: $FAILED_COUNT, 剩余: $((12 - COMPLETED_COUNT - FAILED_COUNT))" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
}

# ============================================================================
# 实验1: Text Simple Concat (Baseline)
# 不使用任何跨模态注意力机制
# ============================================================================

#for seed in "${SEEDS[@]}"; do
#    run_experiment \
#        "Exp-1: Baseline" \
#        1 \
#        $seed \
#        False \
#        False \
#        False
#done

# ============================================================================
# 实验2: +Late Fusion
# 添加晚期跨模态注意力（全局级别融合）
# ============================================================================

#for seed in "${SEEDS[@]}"; do
#    run_experiment \
#        "Exp-2: +Late Fusion" \
#        2 \
#        $seed \
#        True \
#        False \
#        False
#done

# ============================================================================
# 实验3: +Late Fusion +Middle Fusion (创新1)
# Late Fusion + 中期融合（在编码过程中注入文本信息）
# ============================================================================

#for seed in "${SEEDS[@]}"; do
for seed in 123 7; do
    run_experiment \
        "Exp-3: +Middle Fusion" \
        3 \
        $seed \
        True \
        True \
        False
done

# ============================================================================
# 实验4: +Late Fusion +Fine-Grained (创新2)
# Late Fusion + 细粒度注意力（原子-词级别对齐）
# ============================================================================

for seed in "${SEEDS[@]}"; do
    run_experiment \
        "Exp-4: +Fine-Grained" \
        4 \
        $seed \
        True \
        False \
        True
done

# ============================================================================
# 所有实验完成汇总
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🎉 所有实验执行完成！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "结束时间: $(date)" | tee -a "$MAIN_LOG"
echo "总计任务: 12" | tee -a "$MAIN_LOG"
echo "成功完成: $COMPLETED_COUNT" | tee -a "$MAIN_LOG"
echo "执行失败: $FAILED_COUNT" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 生成结果汇总
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 生成结果汇总..." | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

python summarize_multi_seed_results.py --ablation_dir "$BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "✅ 消融实验全部完成！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "查看结果:" | tee -a "$MAIN_LOG"
echo "  - 主日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "  - 简明汇总: $BASE_OUTPUT_DIR/ablation_summary.csv" | tee -a "$MAIN_LOG"
echo "  - 详细结果: $BASE_OUTPUT_DIR/ablation_detailed.csv" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "各实验日志:" | tee -a "$MAIN_LOG"
for exp_num in {1..4}; do
    for seed in "${SEEDS[@]}"; do
        log_file="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}/training.log"
        if [ -f "$log_file" ]; then
            echo "  - exp${exp_num}_seed${seed}: $log_file" | tee -a "$MAIN_LOG"
        fi
    done
done
echo "" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
