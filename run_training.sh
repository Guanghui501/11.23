#!/bin/bash
# DynamicFusionModule 训练启动脚本

set -e  # Exit on error

echo "=========================================="
echo "DynamicFusionModule 训练启动"
echo "=========================================="
echo ""

# 1. 验证集成
echo "1️⃣ 验证集成..."
if python test_integration.py; then
    echo "✅ 集成验证通过"
else
    echo "❌ 集成验证失败，请检查配置"
    exit 1
fi

echo ""
echo "=========================================="
echo ""

# 2. 询问训练模式
echo "请选择训练模式："
echo "1) 快速测试 (5 epochs, 小数据集)"
echo "2) 标准训练 (100 epochs)"
echo "3) 自定义"
echo ""
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动快速测试..."
        python train.py \
            --config config_dynamic_fusion.json \
            --n_train 100 \
            --n_val 20 \
            --n_test 20 \
            --epochs 5 \
            --output_dir ./output_test
        ;;
    2)
        echo ""
        echo "🚀 启动标准训练..."
        python train.py \
            --config config_dynamic_fusion.json \
            --epochs 100 \
            --output_dir ./output_dynamic_fusion
        ;;
    3)
        echo ""
        read -p "Epochs: " epochs
        read -p "Output directory: " outdir
        echo ""
        echo "🚀 启动自定义训练..."
        python train.py \
            --config config_dynamic_fusion.json \
            --epochs $epochs \
            --output_dir $outdir
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""

# 3. 显示结果
if [ -f "$outdir/fusion_weights.csv" ] || [ -f "./output_dynamic_fusion/fusion_weights.csv" ] || [ -f "./output_test/fusion_weights.csv" ]; then
    echo "📊 生成的文件："
    echo ""

    # 找到输出目录
    if [ -d "$outdir" ]; then
        RESULT_DIR="$outdir"
    elif [ -d "./output_dynamic_fusion" ]; then
        RESULT_DIR="./output_dynamic_fusion"
    else
        RESULT_DIR="./output_test"
    fi

    echo "输出目录: $RESULT_DIR"
    echo ""
    ls -lh "$RESULT_DIR"/*.pt 2>/dev/null || echo "  (无检查点文件)"
    ls -lh "$RESULT_DIR"/*.csv 2>/dev/null || echo "  (无权重日志)"
    ls -lh "$RESULT_DIR"/*.json 2>/dev/null || echo "  (无历史记录)"

    echo ""
    echo "查看权重统计："
    echo "  cat $RESULT_DIR/fusion_weights.csv"
    echo ""
    echo "查看最后的权重："
    if [ -f "$RESULT_DIR/fusion_weights.csv" ]; then
        tail -1 "$RESULT_DIR/fusion_weights.csv"
    fi
fi

echo ""
echo "下一步："
echo "1. 分析权重演化: python -c \"import pandas as pd; df=pd.read_csv('$RESULT_DIR/fusion_weights.csv'); print(df)\""
echo "2. 查看训练历史: cat $RESULT_DIR/history_val.json"
echo "3. 加载最佳模型: 见 TRAINING_COMMANDS.md"
