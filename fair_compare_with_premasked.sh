#!/bin/bash
"""
使用预生成的遮挡数据集进行公平对比

步骤：
1. 预生成所有遮挡数据（只需运行一次）
2. 两个模型使用完全相同的遮挡数据进行评估
"""

set -e

echo "========================================================================"
echo "步骤1: 预生成遮挡数据集（只需运行一次）"
echo "========================================================================"

# 设置路径
TEST_DATA="./corrected_test_set/test.pkl"
MASKED_DATA_DIR="./masked_datasets"

# 预生成所有遮挡数据
python pregenerate_masked_dataset.py \
    --input_data "$TEST_DATA" \
    --output_dir "$MASKED_DATA_DIR" \
    --strategies random_token sentence keep_keywords \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seed 42

echo ""
echo "========================================================================"
echo "步骤2: 使用相同的遮挡数据评估两个模型"
echo "========================================================================"

# 模型路径
MODEL1="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
MODEL2="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs64_sw_ju_onlymiddle/mbj_bandgap/best_test_model.pt"

# 评估模型1（中期融合+跨模态+细粒度）
echo ""
echo "评估模型1: 中期融合+跨模态+细粒度"
echo "------------------------------------------------------------------------"

for strategy in random_token sentence keep_keywords; do
    for ratio in 0.0 0.5 1.0; do
        echo "  Strategy: $strategy, Ratio: $ratio"

        # 使用预生成的遮挡数据
        MASKED_FILE="${MASKED_DATA_DIR}/${strategy}_${ratio}.pkl"

        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL1" \
            --masked_data "$MASKED_FILE" \
            --output_file "./results_model1_${strategy}_${ratio}.json" \
            --batch_size 64
    done
done

# 评估模型2（跨模态+细粒度）
echo ""
echo "评估模型2: 跨模态+细粒度"
echo "------------------------------------------------------------------------"

for strategy in random_token sentence keep_keywords; do
    for ratio in 0.0 0.5 1.0; do
        echo "  Strategy: $strategy, Ratio: $ratio"

        # 使用相同的预生成遮挡数据
        MASKED_FILE="${MASKED_DATA_DIR}/${strategy}_${ratio}.pkl"

        python evaluate_with_premasked_data.py \
            --checkpoint "$MODEL2" \
            --masked_data "$MASKED_FILE" \
            --output_file "./results_model2_${strategy}_${ratio}.json" \
            --batch_size 64
    done
done

echo ""
echo "========================================================================"
echo "对比完成！两个模型使用了完全相同的遮挡数据"
echo "========================================================================"
