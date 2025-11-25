#!/bin/bash
# 调查权重数据缺失的原因

echo "=========================================="
echo "调查权重数据缺失问题"
echo "=========================================="
echo ""

BASE_DIR="./fusion_layer_search"

# 检查每个配置
for config in layers_1 layers_2 layers_3 layers_2_3 layers_1_2_3; do
    echo "检查配置: $config"
    echo "----------------------------------------"

    # 检查目录是否存在
    if [ -d "$BASE_DIR/$config" ]; then
        echo "✓ 目录存在"

        # 检查训练日志
        LOG_FILE=$(ls $BASE_DIR/$config/train_*.log 2>/dev/null | head -1)
        if [ -f "$LOG_FILE" ]; then
            echo "✓ 日志文件存在: $LOG_FILE"

            # 检查是否有权重监控输出
            if grep -q "DynamicFusionModule Weight" "$LOG_FILE"; then
                echo "✓ 找到权重监控输出"
                echo "  示例:"
                grep "DynamicFusionModule Weight" "$LOG_FILE" | head -3
            else
                echo "✗ 未找到权重监控输出"
                echo "  可能原因: epochs < 5 或监控未启用"
            fi

            # 检查 epochs 数
            total_epochs=$(grep -c "Epoch:" "$LOG_FILE" 2>/dev/null || echo "0")
            echo "  训练 epochs: $total_epochs"

        else
            echo "✗ 日志文件不存在"
        fi

        # 检查 fusion_weights.csv
        CSV_FILE="$BASE_DIR/$config/mbj_bandgap/fusion_weights.csv"
        if [ -f "$CSV_FILE" ]; then
            echo "✓ fusion_weights.csv 存在"
            echo "  行数: $(wc -l < "$CSV_FILE")"
            echo "  内容预览:"
            head -3 "$CSV_FILE"
        else
            echo "✗ fusion_weights.csv 不存在"
            echo "  路径: $CSV_FILE"
        fi

        # 检查输出目录结构
        echo "  目录结构:"
        ls -la "$BASE_DIR/$config/mbj_bandgap/" 2>/dev/null | head -10

    else
        echo "✗ 目录不存在: $BASE_DIR/$config"
    fi

    echo ""
    echo "=========================================="
    echo ""
done

# 检查基线
echo "检查基线配置: baseline_no_fusion"
echo "----------------------------------------"
if [ -d "$BASE_DIR/baseline_no_fusion" ]; then
    LOG_FILE=$(ls $BASE_DIR/baseline_no_fusion/train_*.log 2>/dev/null | head -1)
    if [ -f "$LOG_FILE" ]; then
        echo "✓ 基线日志存在"
        echo "  验证和测试 MAE:"
        grep "Best_val_mae\|Best_test_mae" "$LOG_FILE" | tail -5
    fi
fi

echo ""
echo "=========================================="
echo "诊断总结"
echo "=========================================="
