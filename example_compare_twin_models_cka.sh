#!/bin/bash
# 双模型CKA相似度对比示例脚本
# 用于计算baseline模型和SGANet模型在相同特征阶段的CKA相似度

# ============================================================================
# 示例 1: 对比baseline和带中期融合的模型
# ============================================================================

python compare_twin_models_cka.py \
    --ckpt_model1 checkpoints/baseline_mbj_bandgap.pt \
    --ckpt_model2 checkpoints/sganet_middle_fusion_2_3_mbj_bandgap.pt \
    --model1_name "Baseline" \
    --model2_name "SGANet (Middle Fusion 2,3)" \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir ./twin_cka_baseline_vs_middle_fusion

# 输出文件:
# - twin_cka_baseline_vs_middle_fusion/twin_models_cka_comparison.png  (可视化)
# - twin_cka_baseline_vs_middle_fusion/twin_models_cka_report.txt      (详细报告)
# - twin_cka_baseline_vs_middle_fusion/twin_models_cka_scores.csv      (CKA分数)


# ============================================================================
# 示例 2: 对比不同融合层位置的模型
# ============================================================================

python compare_twin_models_cka.py \
    --ckpt_model1 checkpoints/sganet_middle_fusion_1_2_mbj_bandgap.pt \
    --ckpt_model2 checkpoints/sganet_middle_fusion_2_3_mbj_bandgap.pt \
    --model1_name "Middle Fusion [1,2]" \
    --model2_name "Middle Fusion [2,3]" \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir ./twin_cka_fusion_1_2_vs_2_3


# ============================================================================
# 示例 3: 对比不同属性训练的模型
# ============================================================================

python compare_twin_models_cka.py \
    --ckpt_model1 checkpoints/sganet_mbj_bandgap.pt \
    --ckpt_model2 checkpoints/sganet_bulk_modulus_kv.pt \
    --model1_name "SGANet (bandgap)" \
    --model2_name "SGANet (bulk modulus)" \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 32 \
    --max_samples 500 \
    --save_dir ./twin_cka_bandgap_vs_bulk_modulus


# ============================================================================
# 示例 4: 完整配置 - 大样本量，用于论文结果
# ============================================================================

python compare_twin_models_cka.py \
    --ckpt_model1 checkpoints/baseline_final.pt \
    --ckpt_model2 checkpoints/sganet_final.pt \
    --model1_name "Baseline (No Fusion)" \
    --model2_name "SGANet (Full Model)" \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --batch_size 64 \
    --max_samples 2000 \
    --save_dir ./paper_results/twin_cka_baseline_vs_sganet


echo "✅ 所有CKA对比分析完成！"
echo "请查看各个输出目录的结果文件"
