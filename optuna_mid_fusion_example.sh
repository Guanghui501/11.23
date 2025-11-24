#!/bin/bash
# 示例：使用 Optuna 专注于中期融合参数调优

echo "================================================"
echo "中期融合（Mid-level Fusion）超参数调优示例"
echo "================================================"
echo ""
echo "此脚本演示如何使用 Optuna 优化 ALIGNN 模型的中期融合参数。"
echo ""
echo "中期融合的作用："
echo "  - 在图编码的中间层插入文本信息"
echo "  - 允许文本特征调制节点表示"
echo "  - 提供比纯后期融合更丰富的多模态交互"
echo ""
echo "可调参数："
echo "  1. use_middle_fusion: 是否启用中期融合"
echo "  2. middle_fusion_layers: 在哪些层插入融合 (如 '2' 或 '1,3')"
echo "  3. middle_fusion_hidden_dim: 融合模块的隐藏维度"
echo "  4. middle_fusion_num_heads: 注意力头数"
echo "  5. middle_fusion_dropout: Dropout 率"
echo ""
echo "================================================"
echo ""

# 设置参数
N_TRIALS=${1:-50}
N_JOBS=${2:-1}
DATASET=${3:-user_data}
TARGET=${4:-target}
OUTPUT_DIR=${5:-mid_fusion_optuna_results}

echo "运行参数："
echo "  试验次数: $N_TRIALS"
echo "  并行作业: $N_JOBS"
echo "  数据集: $DATASET"
echo "  目标: $TARGET"
echo "  输出目录: $OUTPUT_DIR"
echo ""
echo "================================================"
echo ""

# 步骤 1: 运行 Optuna 优化
echo "步骤 1: 开始 Optuna 超参数搜索..."
echo ""

python train_optuna.py \
    --n_trials $N_TRIALS \
    --n_jobs $N_JOBS \
    --dataset $DATASET \
    --target $TARGET \
    --output_dir $OUTPUT_DIR \
    --n_epochs 100

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 优化失败！"
    exit 1
fi

echo ""
echo "================================================"
echo ""

# 步骤 2: 显示最佳参数
echo "步骤 2: 最佳超参数"
echo ""

if [ -f "$OUTPUT_DIR/best_params.json" ]; then
    echo "完整参数文件: $OUTPUT_DIR/best_params.json"
    echo ""

    # 提取中期融合相关参数
    echo "中期融合参数："
    cat "$OUTPUT_DIR/best_params.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
params = data.get('best_params', {})

mid_fusion_params = {
    'use_middle_fusion': params.get('use_middle_fusion', 'N/A'),
    'middle_fusion_layers': params.get('middle_fusion_layers', 'N/A'),
    'middle_fusion_hidden_dim': params.get('middle_fusion_hidden_dim', 'N/A'),
    'middle_fusion_num_heads': params.get('middle_fusion_num_heads', 'N/A'),
    'middle_fusion_dropout': params.get('middle_fusion_dropout', 'N/A'),
}

for k, v in mid_fusion_params.items():
    print(f'  {k}: {v}')

print(f\"\\n最佳验证 MAE: {data.get('best_value', 'N/A')}\")
"
else
    echo "⚠️  找不到最佳参数文件"
fi

echo ""
echo "================================================"
echo ""

# 步骤 3: 查看可视化结果
echo "步骤 3: 查看可视化结果"
echo ""

if [ -f "$OUTPUT_DIR/optimization_history.html" ]; then
    echo "✓ 生成的可视化文件："
    echo "  - $OUTPUT_DIR/optimization_history.html (优化历史)"
    echo "  - $OUTPUT_DIR/param_importances.html (参数重要性)"
    echo "  - $OUTPUT_DIR/parallel_coordinate.html (并行坐标图)"
    echo ""
    echo "在浏览器中打开这些文件以查看详细分析"
else
    echo "⚠️  可视化文件未生成 (可能需要安装 plotly)"
    echo "    运行: pip install plotly kaleido"
fi

echo ""
echo "================================================"
echo ""

# 步骤 4: 使用最佳参数训练
echo "步骤 4: 使用最佳参数进行完整训练"
echo ""
echo "运行以下命令开始完整训练："
echo ""
echo "  python train_with_best_params.py \\"
echo "      --best_params $OUTPUT_DIR/best_params.json \\"
echo "      --epochs 500 \\"
echo "      --dataset $DATASET \\"
echo "      --target $TARGET \\"
echo "      --output_dir ${OUTPUT_DIR}_best_model"
echo ""
echo "================================================"
echo ""

echo "✅ Optuna 优化完成！"
echo ""
echo "下一步："
echo "  1. 查看可视化结果了解参数重要性"
echo "  2. 运行上述命令进行完整训练"
echo "  3. 比较中期融合 vs 后期融合的性能差异"
echo ""
