#!/bin/bash

###############################################################################
# 图-文本检索评估脚本
# 用于评估模型的图-文本对齐能力 (R@1, R@5, R@10)
###############################################################################

set -e  # 遇到错误立即退出

# ======================== 配置参数 ========================

# 模型检查点路径
CHECKPOINT="checkpoints/best_model.pt"

# 数据集配置
DATASET_PATH="your_dataset_path"
TARGET_PROPERTY="target_property"

# 评估参数
SPLIT="val"              # 评估哪个数据集: train, val, test
MAX_SAMPLES=1000         # 最多评估多少样本 (1000 足够快且准确)
K_VALUES="1 5 10 20"     # 计算哪些 K 值
BATCH_SIZE=32

# 输出目录
OUTPUT_DIR="./retrieval_results"

# 设备
DEVICE="cuda"

# ======================== 函数定义 ========================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

# ======================== 主流程 ========================

print_header "🎯 图-文本检索评估"

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 找不到检查点文件: $CHECKPOINT"
    echo "请修改脚本中的 CHECKPOINT 变量"
    exit 1
fi

echo "📋 评估配置:"
echo "  - 检查点: $CHECKPOINT"
echo "  - 数据集: $SPLIT"
echo "  - 样本数: $MAX_SAMPLES"
echo "  - K 值: $K_VALUES"
echo "  - 输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ======================== 运行评估 ========================

print_header "🚀 开始评估..."

python evaluate_retrieval.py \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --max_samples $MAX_SAMPLES \
    --k_values $K_VALUES \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# ======================== 检查结果 ========================

print_header "📊 评估完成！"

if [ -f "$OUTPUT_DIR/retrieval_results.json" ]; then
    echo "✅ 结果文件已生成:"
    echo ""
    echo "📄 JSON 结果:"
    cat "$OUTPUT_DIR/retrieval_results.json" | python -m json.tool
    echo ""
    echo "📊 生成的可视化:"
    ls -lh "$OUTPUT_DIR"/*.png
else
    echo "❌ 未找到结果文件"
    exit 1
fi

print_header "🎉 全部完成！"

echo "查看结果:"
echo "  - JSON: $OUTPUT_DIR/retrieval_results.json"
echo "  - 相似度矩阵: $OUTPUT_DIR/similarity_matrix.png"
echo "  - 检索指标图: $OUTPUT_DIR/retrieval_metrics.png"
echo ""

# ======================== 可选：打开结果图 ========================

# 如果在图形界面环境，可以自动打开图片
if command -v xdg-open &> /dev/null; then
    read -p "是否打开可视化图表? (y/n): " choice
    if [ "$choice" = "y" ]; then
        xdg-open "$OUTPUT_DIR/similarity_matrix.png" &
        xdg-open "$OUTPUT_DIR/retrieval_metrics.png" &
    fi
fi

echo "✨ Done!"
