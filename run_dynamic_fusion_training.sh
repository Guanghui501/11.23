#!/bin/bash
# DynamicFusionModule 训练 - 使用 train_with_cross_modal_attention.py

set -e  # Exit on error

echo "=========================================="
echo "DynamicFusionModule 训练"
echo "使用 train_with_cross_modal_attention.py"
echo "=========================================="
echo ""

# 验证集成
echo "1️⃣ 验证集成..."
if python test_integration.py; then
    echo "✅ 集成验证通过"
else
    echo "❌ 集成验证失败，请检查配置"
    exit 1
fi

echo ""
echo "=========================================="
echo "🚀 启动 DynamicFusionModule 训练"
echo "=========================================="
echo ""

# 训练配置
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_dropout 0.1 \
    \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --cross_modal_dropout 0.1 \
    \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.0 \
    \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --early_stopping_patience 20 \
    \
    --output_dir ./output_dynamic_fusion/ \
    --random_seed 123

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""

# 显示结果
if [ -d "./output_dynamic_fusion/formation_energy_peratom/" ]; then
    RESULT_DIR="./output_dynamic_fusion/formation_energy_peratom/"

    echo "📊 生成的文件："
    echo ""
    ls -lh "$RESULT_DIR"/*.pt 2>/dev/null || echo "  (无检查点文件)"
    ls -lh "$RESULT_DIR"/*.csv 2>/dev/null || echo "  (无权重日志)"
    ls -lh "$RESULT_DIR"/*.json 2>/dev/null || echo "  (无历史记录)"

    echo ""
    echo "查看权重统计："
    if [ -f "$RESULT_DIR/fusion_weights.csv" ]; then
        echo "  cat $RESULT_DIR/fusion_weights.csv"
        echo ""
        echo "最后一次权重记录："
        tail -1 "$RESULT_DIR/fusion_weights.csv"
    else
        echo "  权重日志文件未找到（可能训练轮数 < 5）"
    fi
fi

echo ""
echo "分析结果："
echo "  python analyze_fusion_weights.py --output_dir $RESULT_DIR"
