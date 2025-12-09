#!/bin/bash
# 快速测试脚本 - 只测试一个策略以验证配置
# 用法: 在你的服务器上运行此脚本

CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"  # 修改为你的预处理数据路径
DATASET="jarvis"
PROPERTY="mbj_bandgap"
OUTPUT_DIR="./test_masking_output"

echo "=========================================="
echo "快速测试 - 文本遮挡评估"
echo "=========================================="
echo ""
echo "只测试 random_token 策略，遮挡率: 0%, 50%, 100%"
echo ""

python evaluate_text_masking.py \
    --checkpoint "$CHECKPOINT" \
    --preprocessed_dir "$PREPROCESSED_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 64 \
    --seed 42

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 测试成功！"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "接下来你可以："
    echo "  1. 查看结果: cat $OUTPUT_DIR/text_masking_report_random_token.txt"
    echo "  2. 运行完整评估: ./run_masking_eval_mbj_bandgap.sh"
else
    echo ""
    echo "✗ 测试失败"
    echo ""
    echo "请检查："
    echo "  1. Checkpoint路径: $CHECKPOINT"
    echo "  2. 预处理数据路径: $PREPROCESSED_DIR"
    echo "  3. 预处理数据目录结构:"
    echo "     $PREPROCESSED_DIR/jarvis/mbj_bandgap/test.pkl"
    echo ""
    echo "如果没有预处理数据，请先运行数据预处理脚本"
fi
