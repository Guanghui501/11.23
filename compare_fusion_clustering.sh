#!/bin/bash
# 快速对比中期融合的聚类效果

echo "========================================"
echo "中期融合聚类效果对比"
echo "========================================"

# 配置参数
DATA_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
OUTPUT_DIR="fusion_clustering_analysis"

# 模型路径（需要修改为实际路径）
MODEL_WITHOUT_FUSION="path/to/model_without_middle_fusion.pth"
MODEL_WITH_FUSION="path/to/model_with_middle_fusion.pth"

echo ""
echo "📋 配置:"
echo "  数据目录: $DATA_DIR"
echo "  数据集: $DATASET"
echo "  属性: $PROPERTY"
echo "  无融合模型: $MODEL_WITHOUT_FUSION"
echo "  有融合模型: $MODEL_WITH_FUSION"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查模型文件是否存在
if [ ! -f "$MODEL_WITHOUT_FUSION" ]; then
    echo "❌ 错误: 找不到无融合模型: $MODEL_WITHOUT_FUSION"
    echo "提示: 请先训练或指定正确的模型路径"
    exit 1
fi

if [ ! -f "$MODEL_WITH_FUSION" ]; then
    echo "❌ 错误: 找不到有融合模型: $MODEL_WITH_FUSION"
    echo "提示: 请先训练或指定正确的模型路径"
    exit 1
fi

# 运行可视化
echo "🚀 开始分析..."
python visualize_middle_fusion_clustering.py \
    --checkpoint_without_fusion "$MODEL_WITHOUT_FUSION" \
    --checkpoint_with_fusion "$MODEL_WITH_FUSION" \
    --data_dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --n_samples 1000 \
    --reduction_method tsne \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 分析完成！"
    echo "📊 结果保存在: $OUTPUT_DIR"
    echo ""
    echo "生成的图像:"
    echo "  - clustering_comparison.png : 聚类对比图"
    echo "  - metrics_comparison.png    : 指标对比图"
else
    echo ""
    echo "❌ 分析失败，请检查错误信息"
fi
