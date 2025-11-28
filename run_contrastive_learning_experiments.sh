#!/bin/bash
# 对比学习系统实验脚本
# 验证对比学习是否能有效降低MAE和改善泛化性能

# ============================================================================
# 配置通用参数
# ============================================================================

DATASET="jarvis"
PROPERTY="optb88vdw_bandgap"
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
BASE_OUTPUT_DIR="./contrastive_learning_experiments"
EPOCHS=300
EARLY_STOPPING=100
NUM_WORKERS=24

# 基础配置（从当前最佳设置）
BASE_ARGS=(
    --dataset $DATASET
    --property $PROPERTY
    --root_dir $ROOT_DIR
    --epochs $EPOCHS
    --n_early_stopping $EARLY_STOPPING
    --num_workers $NUM_WORKERS
    --middle_fusion_layers "2,3"
    --middle_fusion_hidden_dim 256
    --middle_fusion_num_heads 4
    --middle_fusion_dropout 0.15
    --graph_dropout 0.2
    --weight_decay 0.001
    --cross_modal_dropout 0.15
    --fine_grained_dropout 0.25
    --learning_rate 0.0005
    --scheduler lambda
    --batch_size 64
)

echo "=========================================="
echo "对比学习系统实验"
echo "目标：验证对比学习是否能降低MAE"
echo "=========================================="
echo ""

# ============================================================================
# Phase 1: 验证对比学习的基本效果
# ============================================================================

echo "=========================================="
echo "Phase 1: 验证对比学习基本效果"
echo "=========================================="
echo ""

# 实验0: Baseline（无对比学习）- 作为对照组
echo "实验0: Baseline（无对比学习）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 0 \
    --output_dir $BASE_OUTPUT_DIR/exp0_baseline_no_contrastive \
    2>&1 | tee $BASE_OUTPUT_DIR/exp0_log.txt

echo "✅ 实验0完成"
echo ""

# 实验1: 启用对比学习（保守配置）
echo "实验1: 对比学习（保守 - weight=0.1）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.1 \
    --contrastive_temperature 0.1 \
    --output_dir $BASE_OUTPUT_DIR/exp1_contrastive_weight_0.1 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp1_log.txt

echo "✅ 实验1完成"
echo ""

# 实验2: 对比学习（中等配置）
echo "实验2: 对比学习（中等 - weight=0.2）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.1 \
    --output_dir $BASE_OUTPUT_DIR/exp2_contrastive_weight_0.2 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp2_log.txt

echo "✅ 实验2完成"
echo ""

# 实验3: 对比学习（激进配置）
echo "实验3: 对比学习（激进 - weight=0.3）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.3 \
    --contrastive_temperature 0.1 \
    --output_dir $BASE_OUTPUT_DIR/exp3_contrastive_weight_0.3 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp3_log.txt

echo "✅ 实验3完成"
echo ""

# ============================================================================
# Phase 2: 温度参数调优
# ============================================================================

echo "=========================================="
echo "Phase 2: 温度参数调优"
echo "=========================================="
echo ""

# 实验4: 低温度（更锐利）
echo "实验4: 对比学习（低温度 - temp=0.07）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.07 \
    --output_dir $BASE_OUTPUT_DIR/exp4_contrastive_temp_0.07 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp4_log.txt

echo "✅ 实验4完成"
echo ""

# 实验5: 高温度（更平滑）
echo "实验5: 对比学习（高温度 - temp=0.15）"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.15 \
    --output_dir $BASE_OUTPUT_DIR/exp5_contrastive_temp_0.15 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp5_log.txt

echo "✅ 实验5完成"
echo ""

# ============================================================================
# Phase 3: 对比学习 + 增强正则化
# ============================================================================

echo "=========================================="
echo "Phase 3: 对比学习 + 增强正则化组合"
echo "=========================================="
echo ""

# 实验6: 对比学习 + 强dropout
echo "实验6: 对比学习 + 强dropout"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.08 \
    --graph_dropout 0.25 \
    --cross_modal_dropout 0.2 \
    --fine_grained_dropout 0.3 \
    --middle_fusion_dropout 0.2 \
    --output_dir $BASE_OUTPUT_DIR/exp6_contrastive_plus_strong_dropout \
    2>&1 | tee $BASE_OUTPUT_DIR/exp6_log.txt

echo "✅ 实验6完成"
echo ""

# 实验7: 对比学习 + 强weight_decay
echo "实验7: 对比学习 + 强weight_decay"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.08 \
    --weight_decay 0.002 \
    --output_dir $BASE_OUTPUT_DIR/exp7_contrastive_plus_strong_weight_decay \
    2>&1 | tee $BASE_OUTPUT_DIR/exp7_log.txt

echo "✅ 实验7完成"
echo ""

# ============================================================================
# Phase 4: 最优组合（基于Phase 1-3结果）
# ============================================================================

echo "=========================================="
echo "Phase 4: 最优组合"
echo "=========================================="
echo ""

# 实验8: 综合最佳配置
echo "实验8: 对比学习综合最佳配置"
python train_with_cross_modal_attention.py \
    "${BASE_ARGS[@]}" \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.08 \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --output_dir $BASE_OUTPUT_DIR/exp8_contrastive_best_combined \
    2>&1 | tee $BASE_OUTPUT_DIR/exp8_log.txt

echo "✅ 实验8完成"
echo ""

# ============================================================================
# 生成对比报告
# ============================================================================

echo "=========================================="
echo "生成实验对比报告"
echo "=========================================="
echo ""

python compare_experiments.py \
    --experiment_dirs $BASE_OUTPUT_DIR/exp* \
    --save_dir $BASE_OUTPUT_DIR/comparison_report

echo ""
echo "=========================================="
echo "Phase 1-4 完成！现在进行深度分析..."
echo "=========================================="
echo ""

# ============================================================================
# 深度分析：对比最佳模型
# ============================================================================

# 找到验证MAE最低的实验（需要手动指定，或写个脚本自动找）
# 这里假设是 exp2

BEST_EXP="exp2_contrastive_weight_0.2"
BASELINE_EXP="exp0_baseline_no_contrastive"

echo "对比最佳模型与baseline的CKA分析..."
python compare_twin_models_cka.py \
    --ckpt_model1 $BASE_OUTPUT_DIR/$BASELINE_EXP/best_model.pt \
    --ckpt_model2 $BASE_OUTPUT_DIR/$BEST_EXP/best_model.pt \
    --model1_name "Baseline (No Contrastive)" \
    --model2_name "Best (With Contrastive)" \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --save_dir $BASE_OUTPUT_DIR/cka_analysis

echo ""
echo "对比最佳模型与baseline的性能分析..."
python analyze_model_performance.py \
    --ckpt_model1 $BASE_OUTPUT_DIR/$BASELINE_EXP/best_model.pt \
    --ckpt_model2 $BASE_OUTPUT_DIR/$BEST_EXP/best_model.pt \
    --model1_name "Baseline (No Contrastive)" \
    --model2_name "Best (With Contrastive)" \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --save_dir $BASE_OUTPUT_DIR/performance_analysis

echo ""
echo "🎉 所有对比学习实验完成！"
echo ""
echo "结果保存在: $BASE_OUTPUT_DIR"
echo ""
echo "关键文件："
echo "  1. 对比报告: $BASE_OUTPUT_DIR/comparison_report/experiments_report.txt"
echo "  2. CKA分析: $BASE_OUTPUT_DIR/cka_analysis/twin_models_cka_report.txt"
echo "  3. 性能分析: $BASE_OUTPUT_DIR/performance_analysis/performance_report.txt"
echo ""
echo "下一步："
echo "  1. 查看对比报告，确认哪个配置验证MAE最低"
echo "  2. 查看CKA分析，确认融合效果是否保持（fused CKA应降低）"
echo "  3. 查看性能分析，确认train-val gap是否缩小"
echo "  4. 如果有效，使用最佳配置进行正式训练"
echo ""
echo "预期改善："
echo "  - 验证MAE: 10 → 7-8 (-20~30%)"
echo "  - Train-Val Gap: 3-4x → 1.5-2.5x"
echo "  - CKA (fused): 0.98 → 0.85-0.92"
