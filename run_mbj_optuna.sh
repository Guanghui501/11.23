#!/bin/bash
# MBJ Bandgap Optuna 超参数调优快速启动脚本

echo "========================================================================================================="
echo "                        MBJ Bandgap 超参数调优 - Optuna 自动优化                                        "
echo "========================================================================================================="
echo ""
echo "此脚本使用 Optuna 自动寻找 MBJ Bandgap 预测的最佳超参数组合"
echo ""
echo "优化参数包括："
echo "  ✓ 模型架构: ALIGNN层数、GCN层数、隐藏层维度"
echo "  ✓ 训练参数: 学习率、权重衰减、批次大小、dropout"
echo "  ✓ 跨模态注意力: 隐藏维度、注意力头数、dropout"
echo "  ✓ 细粒度注意力: 注意力头数、dropout"
echo "  ✓ 中期融合: 融合层位置、隐藏维度、注意力头数、dropout"
echo ""
echo "========================================================================================================="
echo ""

# 设置参数
N_TRIALS=${1:-50}
N_EPOCHS=${2:-100}
N_JOBS=${3:-1}
ROOT_DIR=${4:-"../dataset/"}
OUTPUT_DIR=${5:-"mbj_optuna_results"}
EARLY_STOPPING=${6:-20}

echo "运行参数："
echo "  试验次数:         $N_TRIALS"
echo "  每次试验轮数:     $N_EPOCHS"
echo "  并行作业数:       $N_JOBS"
echo "  数据集目录:       $ROOT_DIR"
echo "  输出目录:         $OUTPUT_DIR"
echo "  早停轮数:         $EARLY_STOPPING"
echo ""
echo "========================================================================================================="
echo ""

# 检查数据集
if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ 错误: 找不到数据集目录 $ROOT_DIR"
    echo ""
    echo "请确保："
    echo "  1. 数据集目录存在"
    echo "  2. 目录包含 mbj_bandgap 数据"
    echo "  3. 数据格式正确（CIF 文件 + description.csv）"
    echo ""
    exit 1
fi

# 检查Python环境
echo "步骤 1: 检查依赖..."
python3 -c "import optuna" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 未安装 Optuna"
    echo "   安装方式: pip install optuna plotly kaleido"
    exit 1
fi
echo "✓ Optuna 已安装"

python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 未安装 PyTorch"
    exit 1
fi
echo "✓ PyTorch 已安装"

python3 -c "import dgl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 未安装 DGL"
    exit 1
fi
echo "✓ DGL 已安装"

echo ""
echo "========================================================================================================="
echo ""

# 运行优化
echo "步骤 2: 开始 Optuna 优化..."
echo ""

python train_mbj_with_optuna.py \
    --root_dir "$ROOT_DIR" \
    --n_trials $N_TRIALS \
    --n_epochs $N_EPOCHS \
    --n_jobs $N_JOBS \
    --output_dir "$OUTPUT_DIR" \
    --early_stopping $EARLY_STOPPING

if [ $? -ne 0 ]; then
    echo ""
    echo "========================================================================================================="
    echo "❌ 优化失败！"
    echo "========================================================================================================="
    exit 1
fi

echo ""
echo "========================================================================================================="
echo "✅ 优化完成！"
echo "========================================================================================================="
echo ""

# 显示结果
if [ -f "$OUTPUT_DIR/best_params_mbj.json" ]; then
    echo "步骤 3: 最佳参数"
    echo ""
    echo "完整参数文件: $OUTPUT_DIR/best_params_mbj.json"
    echo ""

    # 提取关键参数
    python3 << EOF
import json
with open('$OUTPUT_DIR/best_params_mbj.json', 'r') as f:
    data = json.load(f)

print("最佳验证 MAE: {:.6f} eV\n".format(data['best_value']))
print("关键参数:")
params = data['best_params']

# 模型架构
print("  模型架构:")
print(f"    ALIGNN 层数: {params.get('alignn_layers', 'N/A')}")
print(f"    GCN 层数: {params.get('gcn_layers', 'N/A')}")
print(f"    隐藏层维度: {params.get('hidden_features', 'N/A')}")

# 训练参数
print("\n  训练参数:")
print(f"    学习率: {params.get('learning_rate', 'N/A')}")
print(f"    权重衰减: {params.get('weight_decay', 'N/A')}")
print(f"    批次大小: {params.get('batch_size', 'N/A')}")
print(f"    Graph Dropout: {params.get('graph_dropout', 'N/A')}")

# 融合设置
print("\n  融合设置:")
print(f"    跨模态注意力: {params.get('use_cross_modal_attention', 'N/A')}")
print(f"    细粒度注意力: {params.get('use_fine_grained_attention', 'N/A')}")
print(f"    中期融合: {params.get('use_middle_fusion', 'N/A')}")

if params.get('use_middle_fusion'):
    print(f"    中期融合层: {params.get('middle_fusion_layers', 'N/A')}")
EOF

else
    echo "⚠️  找不到最佳参数文件"
fi

echo ""
echo "========================================================================================================="
echo ""

# 显示可视化
if [ -f "$OUTPUT_DIR/mbj_optimization_history.html" ]; then
    echo "步骤 4: 可视化结果"
    echo ""
    echo "生成的可视化文件："
    echo "  ✓ $OUTPUT_DIR/mbj_optimization_history.html (优化历史)"
    echo "  ✓ $OUTPUT_DIR/mbj_param_importances.html (参数重要性)"
    echo "  ✓ $OUTPUT_DIR/mbj_parallel_coordinate.html (并行坐标图)"
    echo ""
    echo "在浏览器中打开这些文件以查看详细分析"
else
    echo "提示: 可视化文件未生成"
    echo "      安装 plotly: pip install plotly kaleido"
fi

echo ""
echo "========================================================================================================="
echo ""

# 下一步提示
echo "步骤 5: 使用最佳参数训练完整模型"
echo ""
echo "运行以下命令进行完整训练（500 epochs）："
echo ""
echo "  python train_with_best_params.py \\"
echo "      --best_params $OUTPUT_DIR/best_params_mbj.json \\"
echo "      --epochs 500 \\"
echo "      --dataset user_data \\"
echo "      --target target \\"
echo "      --output_dir mbj_best_model"
echo ""
echo "========================================================================================================="
echo ""

echo "🎉 完成！MBJ Bandgap 超参数优化已结束。"
echo ""

# 显示使用统计
if [ -f "$OUTPUT_DIR/all_trials_mbj.csv" ]; then
    echo "试验统计："
    python3 << EOF
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/all_trials_mbj.csv')
completed = df[df['state'] == 'COMPLETE']
pruned = df[df['state'] == 'PRUNED']
print(f"  完成的试验: {len(completed)}")
print(f"  剪枝的试验: {len(pruned)}")
if len(completed) > 0:
    print(f"  最佳 MAE: {completed['value'].min():.6f} eV")
    print(f"  最差 MAE: {completed['value'].max():.6f} eV")
    print(f"  平均 MAE: {completed['value'].mean():.6f} eV")
EOF
    echo ""
fi

echo "========================================================================================================="
