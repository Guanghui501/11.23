#!/bin/bash
# 快速示例：对比基线和SGANet在不同阶段的特征

echo "=========================================="
echo "双模型可视化对比示例"
echo "=========================================="
echo ""

# 设置路径（根据实际情况修改）
CKPT_BASE="./output_baseline/jarvis_mbj_bandgap/best_model.pt"
CKPT_SGA="./output_sganet/jarvis_mbj_bandgap/best_model.pt"

# 示例1: 评估中期融合的独立贡献
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

# 示例2: 评估完整多模态融合效果（推荐用于论文主结果）
echo "📊 示例2: FUSED阶段 - 完整多模态融合（图+文本）⭐"
python visualize_twin_models.py \
    --ckpt_base "$CKPT_BASE" \
    --ckpt_sga "$CKPT_SGA" \
    --dataset jarvis \
    --property mbj_bandgap \
    --feature_stage fused \
    --max_samples 1000 \
    --save_dir ./viz_fused \
    --device cuda

echo ""
echo "✅ FUSED阶段分析完成，结果保存在 ./viz_fused/"
echo ""
echo "=========================================="
echo ""

# 示例3: (可选) 仅图特征性能
echo "📊 示例3: FINAL阶段 - 仅图特征性能"
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
echo "FUSED阶段报告: ./viz_fused/comparison_report.txt"
echo "FINAL阶段报告: ./viz_final/comparison_report.txt"
echo ""
echo "对比解读："
echo "  - BASE:  中期融合的纯粹贡献（仅图特征，GCN后）"
echo "  - FUSED: 完整多模态融合效果（图+文本，512维）⭐ 论文主结果"
echo "  - FINAL: 仅图特征性能（256维）"
echo ""
