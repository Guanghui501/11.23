#!/bin/bash
# 模型性能对比分析示例脚本
# 用于理解CKA相似度高的情况下，两个模型的实际预测性能差异

# ============================================================================
# 使用场景：
# 当CKA相似度很高（如0.98）时，需要通过性能对比来判断：
# 1. 融合机制是否真的有效（性能是否提升）
# 2. 高CKA相似度是好事还是坏事
# 3. 是否需要调整融合策略
# ============================================================================

python analyze_model_performance.py \
    --ckpt_model1 checkpoints/baseline_mbj_bandgap.pt \
    --ckpt_model2 checkpoints/sganet_middle_fusion_2_3_mbj_bandgap.pt \
    --model1_name "Baseline" \
    --model2_name "SGANet (Middle Fusion)" \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir ./performance_analysis

# 输出文件:
# - performance_analysis/performance_comparison.png    (6个子图的综合对比)
# - performance_analysis/performance_report.txt        (详细分析报告)
# - performance_analysis/performance_metrics.csv       (性能指标CSV)

echo ""
echo "============================================================================"
echo "分析提示:"
echo "============================================================================"
echo "1. 如果 CKA相似度高(0.98) + 性能相同:"
echo "   → 融合机制可能未充分利用，建议调整融合策略"
echo ""
echo "2. 如果 CKA相似度高(0.98) + 性能提升:"
echo "   → 理想情况！融合带来了关键改进，虽然整体表示相似"
echo ""
echo "3. 如果 CKA相似度高(0.98) + 性能下降:"
echo "   → 需要检查模型训练或融合机制实现"
echo ""
echo "4. 查看 text_fine 阶段的低CKA(0.287)是否被后续层抵消"
echo "============================================================================"
