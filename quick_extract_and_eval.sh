#!/bin/bash
# 快速提取测试集并运行评估
# 这个脚本自动化整个流程：提取测试集 -> 验证 -> 完整评估

set -e  # 遇到错误立即退出

echo "========================================"
echo "测试集提取与评估 - 一键运行"
echo "========================================"
echo ""

# ==================== 修复 GLIBCXX 问题 ====================
echo "检查并修复环境..."

# 获取 conda 环境路径
if [ -z "$CONDA_PREFIX" ]; then
    if [ -d "$HOME/.conda/envs/MatMMFuse" ]; then
        CONDA_PREFIX="$HOME/.conda/envs/MatMMFuse"
    elif [ -d "/public/home/ghzhang/.conda/envs/MatMMFuse" ]; then
        CONDA_PREFIX="/public/home/ghzhang/.conda/envs/MatMMFuse"
    fi
fi

# 设置库路径以避免 GLIBCXX 问题
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "✓ 已设置 LD_LIBRARY_PATH"
else
    echo "⚠ 警告: 未找到 conda 环境，可能遇到库版本问题"
fi

echo ""

# ==================== 配置参数 ====================
PREDICTIONS_CSV="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv"
CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
OUTPUT_DIR="./corrected_test_set"
EVAL_OUTPUT_DIR="./masking_eval_corrected"

EXPECTED_MAE=0.2478  # 训练时的测试集MAE

# ==================== 步骤1: 提取测试集 ====================
echo "========================================"
echo "步骤1: 从CSV提取正确的测试集"
echo "========================================"
echo ""

python extract_test_set_from_csv.py \
    --predictions_csv "$PREDICTIONS_CSV" \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --output_dir "$OUTPUT_DIR"

if [ ! -f "$OUTPUT_DIR/test.pkl" ]; then
    echo ""
    echo "✗ 错误: 测试集提取失败"
    exit 1
fi

echo ""
echo "✓ 测试集提取成功"
echo ""

# ==================== 步骤2: 快速验证 ====================
echo "========================================"
echo "步骤2: 快速验证（0%, 50%, 100%遮挡）"
echo "========================================"
echo ""

python evaluate_text_masking.py \
    --checkpoint "$CHECKPOINT" \
    --test_data "$OUTPUT_DIR/test.pkl" \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./quick_validation \
    --batch_size 64 \
    --seed 42

echo ""
echo "========================================"
echo "验证结果"
echo "========================================"
echo ""

# 提取0%遮挡的MAE
if [ -f "./quick_validation/text_masking_results_random_token.json" ]; then
    ZERO_MASK_MAE=$(python -c "
import json
with open('./quick_validation/text_masking_results_random_token.json') as f:
    data = json.load(f)
    print(data['mae'][0])
")

    echo "训练时测试集MAE: $EXPECTED_MAE"
    echo "评估时0%遮挡MAE: $ZERO_MASK_MAE"
    echo ""

    # 计算差异百分比
    DIFF_PCT=$(python -c "print(abs($ZERO_MASK_MAE - $EXPECTED_MAE) / $EXPECTED_MAE * 100)")

    echo "差异: ${DIFF_PCT}%"
    echo ""

    # 判断是否通过验证
    IS_VALID=$(python -c "print('yes' if abs($ZERO_MASK_MAE - $EXPECTED_MAE) / $EXPECTED_MAE < 0.1 else 'no')")

    if [ "$IS_VALID" = "yes" ]; then
        echo "✓ 验证通过！测试集正确"
        echo ""
    else
        echo "⚠ 警告: 0%遮挡MAE与训练时差异较大"
        echo "可能的原因："
        echo "  1. 使用了错误的checkpoint"
        echo "  2. 模型代码与训练时不一致"
        echo "  3. predictions CSV与checkpoint不匹配"
        echo ""
        read -p "是否继续完整评估? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "已取消"
            exit 1
        fi
    fi
else
    echo "⚠ 警告: 找不到验证结果JSON文件"
    echo ""
fi

# ==================== 步骤3: 完整评估（所有策略）====================
echo "========================================"
echo "步骤3: 运行完整评估（5种策略）"
echo "========================================"
echo ""

strategies=("random_token" "random_word" "random_chunk" "sentence" "keep_keywords")

for strategy in "${strategies[@]}"; do
    echo "----------------------------------------"
    echo "评估策略: $strategy"
    echo "----------------------------------------"
    echo ""

    python evaluate_text_masking.py \
        --checkpoint "$CHECKPOINT" \
        --test_data "$OUTPUT_DIR/test.pkl" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy "$strategy" \
        --output_dir "$EVAL_OUTPUT_DIR/$strategy" \
        --batch_size 64 \
        --seed 42

    echo ""
    echo "✓ $strategy 评估完成"
    echo ""
done

# ==================== 步骤4: 生成对比报告 ====================
echo "========================================"
echo "步骤4: 生成策略对比报告"
echo "========================================"
echo ""

python compare_masking_strategies.py --input_dir "$EVAL_OUTPUT_DIR"

# ==================== 完成 ====================
echo ""
echo "========================================"
echo "✓ 所有评估完成！"
echo "========================================"
echo ""
echo "结果位置:"
echo "  • 测试集: $OUTPUT_DIR/test.pkl"
echo "  • 评估结果: $EVAL_OUTPUT_DIR/"
echo "  • 对比报告: $EVAL_OUTPUT_DIR/strategy_comparison_report.txt"
echo "  • 可视化图表: $EVAL_OUTPUT_DIR/strategy_comparison.png"
echo ""
echo "查看对比报告:"
echo "  cat $EVAL_OUTPUT_DIR/strategy_comparison_report.txt"
echo ""
echo "查看各策略结果:"
for strategy in "${strategies[@]}"; do
    echo "  cat $EVAL_OUTPUT_DIR/$strategy/text_masking_report_$strategy.txt"
done
echo ""
