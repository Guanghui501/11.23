#!/bin/bash
# 文本遮挡评估 - MBJ Band Gap模型
# 用法: 将此脚本复制到你的服务器上运行

# ==================== 路径配置 ====================
CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
DATA_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"

# 预处理数据目录（如果有的话）
# 如果你已经有预处理数据，请设置这个路径
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"  # 修改为你的预处理数据路径

# 输出目录
OUTPUT_DIR="./masking_evaluation_mbj_bandgap"

# 评估脚本路径（假设与此脚本在同一目录）
EVAL_SCRIPT="./evaluate_text_masking.py"

# ==================== 检查文件 ====================
echo "=========================================="
echo "文本遮挡评估 - MBJ Band Gap"
echo "=========================================="
echo ""

# 检查checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint文件不存在"
    echo "   路径: $CHECKPOINT"
    echo ""
    echo "请检查:"
    echo "  1. 路径是否正确"
    echo "  2. 文件是否存在"
    exit 1
fi
echo "✓ Checkpoint: $CHECKPOINT"

# 检查评估脚本
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ 错误: 评估脚本不存在"
    echo "   路径: $EVAL_SCRIPT"
    echo ""
    echo "请确保 evaluate_text_masking.py 在当前目录"
    exit 1
fi
echo "✓ 评估脚本: $EVAL_SCRIPT"

# 检查预处理数据
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "⚠ 警告: 预处理数据目录不存在: $PREPROCESSED_DIR"
    echo ""
    echo "评估脚本需要预处理数据。请："
    echo "  1. 如果已有预处理数据，修改脚本中的 PREPROCESSED_DIR 路径"
    echo "  2. 如果没有，需要先运行预处理脚本生成预处理数据"
    echo ""
    echo "预处理数据目录结构应为："
    echo "  $PREPROCESSED_DIR/"
    echo "    jarvis/"
    echo "      mbj_bandgap/"
    echo "        train.pkl"
    echo "        val.pkl"
    echo "        test.pkl"
    echo ""
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ 预处理数据目录: $PREPROCESSED_DIR"
fi

echo ""

# ==================== 运行评估 ====================
echo "=========================================="
echo "开始评估"
echo "=========================================="
echo ""

# 遮挡策略列表
strategies=("random_token" "random_word" "random_chunk" "sentence" "keep_keywords")

total=${#strategies[@]}
current=0

for strategy in "${strategies[@]}"; do
    current=$((current + 1))
    echo ""
    echo "=========================================="
    echo "[$current/$total] 策略: $strategy"
    echo "=========================================="

    output_dir="$OUTPUT_DIR/${strategy}"

    python "$EVAL_SCRIPT" \
        --checkpoint "$CHECKPOINT" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy "$strategy" \
        --output_dir "$output_dir" \
        --batch_size 64 \
        --seed 42

    if [ $? -eq 0 ]; then
        echo "✓ $strategy 评估完成"
    else
        echo "✗ $strategy 评估失败"
        echo ""
        echo "可能的原因:"
        echo "  1. 预处理数据路径不正确"
        echo "  2. Checkpoint与数据不匹配"
        echo "  3. 内存不足（尝试减小 --batch_size）"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "所有策略评估完成"
echo "=========================================="
echo ""

# ==================== 生成对比报告 ====================
echo "生成对比报告..."
if [ -f "./compare_masking_strategies.py" ]; then
    python ./compare_masking_strategies.py --input_dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ 对比报告生成完成"
    fi
else
    echo "⚠ 警告: compare_masking_strategies.py 不存在，跳过对比报告生成"
fi

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
for strategy in "${strategies[@]}"; do
    output_dir="$OUTPUT_DIR/${strategy}"
    if [ -d "$output_dir" ]; then
        echo "  [$strategy]"
        echo "    - $output_dir/text_masking_results_${strategy}.json"
        echo "    - $output_dir/text_masking_report_${strategy}.txt"
        echo "    - $output_dir/text_masking_analysis_${strategy}.png"
    fi
done

if [ -f "$OUTPUT_DIR/strategy_comparison.png" ]; then
    echo ""
    echo "  [对比结果]"
    echo "    - $OUTPUT_DIR/strategy_comparison.png"
    echo "    - $OUTPUT_DIR/strategy_comparison_report.txt"
    echo "    - $OUTPUT_DIR/strategy_comparison_summary.csv"
fi

echo ""
