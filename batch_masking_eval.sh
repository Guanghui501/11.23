#!/bin/bash
# 批量文本遮挡评估脚本
# 一次性测试所有5种遮挡策略

# 默认参数（可根据需要修改）
CHECKPOINT="${1:-./results/jarvis/formation_energy/best_model.pt}"
PREPROCESSED_DIR="${2:-./preprocessed_data}"
DATASET="${3:-jarvis}"
PROPERTY="${4:-formation_energy}"
OUTPUT_BASE_DIR="${5:-./masking_evaluation}"

echo "=========================================="
echo "批量文本遮挡评估"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "数据目录: $PREPROCESSED_DIR"
echo "数据集: $DATASET"
echo "属性: $PROPERTY"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "=========================================="
echo ""

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint文件不存在: $CHECKPOINT"
    echo ""
    echo "用法:"
    echo "  ./batch_masking_eval.sh <checkpoint> <preprocessed_dir> <dataset> <property> <output_dir>"
    echo ""
    echo "示例:"
    echo "  ./batch_masking_eval.sh ./results/jarvis/formation_energy/best_model.pt ./preprocessed_data jarvis formation_energy ./masking_eval"
    exit 1
fi

# 遮挡策略列表
strategies=("random_token" "random_word" "random_chunk" "sentence" "keep_keywords")

# 计数器
total=${#strategies[@]}
current=0

# 遍历所有策略
for strategy in "${strategies[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total] 测试策略: $strategy"
    echo "=========================================="

    output_dir="$OUTPUT_BASE_DIR/${strategy}"

    python evaluate_text_masking.py \
        --checkpoint "$CHECKPOINT" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy "$strategy" \
        --output_dir "$output_dir" \
        --batch_size 64

    if [ $? -eq 0 ]; then
        echo "✓ $strategy 评估完成"
    else
        echo "✗ $strategy 评估失败"
    fi
done

echo ""
echo "=========================================="
echo "所有策略测试完成！"
echo "=========================================="
echo ""
echo "结果保存在: $OUTPUT_BASE_DIR"
echo ""
echo "生成的文件:"
for strategy in "${strategies[@]}"; do
    output_dir="$OUTPUT_BASE_DIR/${strategy}"
    echo "  [$strategy]"
    echo "    - $output_dir/text_masking_results_${strategy}.json"
    echo "    - $output_dir/text_masking_report_${strategy}.txt"
    echo "    - $output_dir/text_masking_analysis_${strategy}.png"
done

echo ""
echo "接下来你可以:"
echo "  1. 查看各策略的报告: cat $OUTPUT_BASE_DIR/*/text_masking_report_*.txt"
echo "  2. 查看可视化图表: 打开 $OUTPUT_BASE_DIR/*/*.png"
echo "  3. 使用Python脚本对比结果（参见 TEXT_MASKING_EVALUATION_GUIDE.md）"
echo ""
