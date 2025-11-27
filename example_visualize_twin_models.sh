#!/bin/bash
# 快速示例：对比基线和SGANet在不同阶段的特征

echo "=========================================="
echo "双模型可视化对比示例"
echo "=========================================="
echo ""

# 设置路径（根据实际情况修改）
CKPT_BASE="./output_baseline/jarvis_mbj_bandgap/best_model.pt"
CKPT_SGA="./output_sganet/jarvis_mbj_bandgap/best_model.pt"

# 示例1: 评估中期融合的独立贡献（推荐）
echo "📊 示例1: BASE阶段 - 评估中期融合的独立贡献"
python visualize_twin_models.py \
    --ckpt_base "$CKPT_BASE" \
    --ckpt_sga "$CKPT_SGA" \
    --dataset jarvis \
    --property mbj_bandgap \
    --feature_stage base \
    --max_samples 1000 \
    --save_dir ./viz_base \
    --device cuda

echo ""
echo "✅ BASE阶段分析完成，结果保存在 ./viz_base/"
echo ""
echo "=========================================="
echo ""

# 示例2: 评估整体模型性能
echo "📊 示例2: FINAL阶段 - 评估整体模型性能"
python visualize_twin_models.py \
    --ckpt_base "$CKPT_BASE" \
    --ckpt_sga "$CKPT_SGA" \
    --dataset jarvis \
    --property mbj_bandgap \
    --feature_stage final \
    --max_samples 1000 \
    --save_dir ./viz_final \
    --device cuda

echo ""
echo "✅ FINAL阶段分析完成，结果保存在 ./viz_final/"
echo ""
echo "=========================================="
echo ""

echo "📈 查看结果："
echo ""
echo "BASE阶段报告:  ./viz_base/comparison_report.txt"
echo "FINAL阶段报告: ./viz_final/comparison_report.txt"
echo ""
echo "对比解读："
echo "  - BASE:  显示中期融合的纯粹贡献"
echo "  - FINAL: 显示所有模块的综合效果"
echo ""
