#!/bin/bash
# Checkpoint修复和评估一体化脚本
# 用法: 在你的服务器上运行此脚本

CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
OUTPUT_DIR="./masking_evaluation_mbj_bandgap"

echo "=========================================="
echo "Checkpoint修复和评估"
echo "=========================================="
echo ""
echo "Checkpoint: $CHECKPOINT"
echo ""

# 步骤1: 检查checkpoint
echo "=========================================="
echo "步骤1: 检查Checkpoint内容"
echo "=========================================="
echo ""

python inspect_checkpoint.py "$CHECKPOINT"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 检查checkpoint失败"
    exit 1
fi

echo ""
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 步骤2: 创建配置文件
echo ""
echo "=========================================="
echo "步骤2: 创建模型配置文件"
echo "=========================================="
echo ""

CONFIG_FILE="./mbj_bandgap_model_config.json"

python create_config_from_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --output "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 创建配置失败"
    echo ""
    echo "如果checkpoint已包含model_config，可以跳过此步骤"
    echo "按Enter继续..."
    read
fi

# 步骤3: 快速测试
echo ""
echo "=========================================="
echo "步骤3: 快速测试（验证配置）"
echo "=========================================="
echo ""
echo "只测试random_token策略，遮挡率: 0%, 50%, 100%"
echo ""

TEST_OUTPUT="./test_masking_output"

if [ -f "$CONFIG_FILE" ]; then
    # 使用配置文件
    python evaluate_text_masking.py \
        --checkpoint "$CHECKPOINT" \
        --config_file "$CONFIG_FILE" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy random_token \
        --masking_ratios 0.0 0.5 1.0 \
        --output_dir "$TEST_OUTPUT" \
        --batch_size 64 \
        --seed 42
else
    # 不使用配置文件（checkpoint可能已包含model_config）
    python evaluate_text_masking.py \
        --checkpoint "$CHECKPOINT" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy random_token \
        --masking_ratios 0.0 0.5 1.0 \
        --output_dir "$TEST_OUTPUT" \
        --batch_size 64 \
        --seed 42
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 快速测试成功！"
    echo ""
    echo "结果: $TEST_OUTPUT/text_masking_report_random_token.txt"
    echo ""
else
    echo ""
    echo "❌ 快速测试失败"
    echo ""
    echo "请检查："
    echo "  1. 预处理数据是否存在"
    echo "  2. 配置文件是否正确"
    echo "  3. 上面的错误信息"
    exit 1
fi

# 步骤4: 询问是否运行完整评估
echo ""
echo "=========================================="
echo "步骤4: 完整评估（可选）"
echo "=========================================="
echo ""
echo "快速测试成功！是否运行完整评估（5种策略，11个遮挡率）？"
echo "预计时间: 30-60分钟"
echo ""
read -p "继续完整评估? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "开始完整评估..."
    echo ""

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

        if [ -f "$CONFIG_FILE" ]; then
            python evaluate_text_masking.py \
                --checkpoint "$CHECKPOINT" \
                --config_file "$CONFIG_FILE" \
                --preprocessed_dir "$PREPROCESSED_DIR" \
                --dataset "$DATASET" \
                --property "$PROPERTY" \
                --masking_strategy "$strategy" \
                --output_dir "$output_dir" \
                --batch_size 64 \
                --seed 42
        else
            python evaluate_text_masking.py \
                --checkpoint "$CHECKPOINT" \
                --preprocessed_dir "$PREPROCESSED_DIR" \
                --dataset "$DATASET" \
                --property "$PROPERTY" \
                --masking_strategy "$strategy" \
                --output_dir "$output_dir" \
                --batch_size 64 \
                --seed 42
        fi

        if [ $? -eq 0 ]; then
            echo "✓ $strategy 评估完成"
        else
            echo "✗ $strategy 评估失败"
        fi
    done

    # 生成对比报告
    echo ""
    echo "=========================================="
    echo "生成对比报告"
    echo "=========================================="
    echo ""

    python compare_masking_strategies.py --input_dir "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "评估完成！"
    echo "=========================================="
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "查看对比报告:"
    echo "  cat $OUTPUT_DIR/strategy_comparison_report.txt"
    echo ""

else
    echo ""
    echo "已跳过完整评估"
    echo ""
    echo "你可以稍后运行："
    echo "  ./run_masking_eval_mbj_bandgap.sh"
    echo ""
fi

echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
