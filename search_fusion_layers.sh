#!/bin/bash
# 融合层位置搜索 - 阶段1：快速筛选
# 测试不同的 middle_fusion_layers 配置 + 无融合基线

echo "=========================================="
echo "DynamicFusionModule - 融合层位置搜索"
echo "数据集: JARVIS (多属性)"
echo "策略: 小数据快速筛选 (500 样本, 20 epochs)"
echo "=========================================="
echo ""

# 环境配置
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 基础输出目录
BASE_OUTPUT_DIR="./fusion_layer_search"
mkdir -p "$BASE_OUTPUT_DIR"

# 属性列表
PROPERTIES=(
    "mbj_bandgap"
    "bulk_modulus_kv"
)

# 测试配置列表
FUSION_LAYERS_LIST=(
    "1,2"      # 双层融合（第1和第2层）
    "2,3"      # 双层融合（第2和第3层）
)

echo "待测试属性:"
for prop in "${PROPERTIES[@]}"; do
    echo "  - $prop"
done
echo ""

echo "测试配置:"
for layers in "${FUSION_LAYERS_LIST[@]}"; do
    echo "  - Fusion layers: $layers"
done
echo ""
echo "=========================================="
echo ""

# 遍历每个属性
for PROPERTY in "${PROPERTIES[@]}"; do

    echo "=========================================="
    echo "开始测试属性: $PROPERTY"
    echo "=========================================="
    echo ""

    # 为每个属性创建单独的结果文件
    PROPERTY_OUTPUT_DIR="$BASE_OUTPUT_DIR/$PROPERTY"
    mkdir -p "$PROPERTY_OUTPUT_DIR"

    RESULTS_FILE="$PROPERTY_OUTPUT_DIR/results_summary.csv"
    echo "fusion_layers,best_val_mae,best_test_mae,final_w_graph,final_w_text,ratio" > "$RESULTS_FILE"

    # 遍历每个配置
    for FUSION_LAYERS in "${FUSION_LAYERS_LIST[@]}"; do

        # 创建配置特定的输出目录
        CONFIG_NAME="layers_${FUSION_LAYERS//,/_}"  # 将逗号替换为下划线

        OUTPUT_DIR="$PROPERTY_OUTPUT_DIR/$CONFIG_NAME"
        LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

        mkdir -p "$OUTPUT_DIR"

        echo "----------------------------------------"
        echo "属性: $PROPERTY"
        echo "🧪 测试配置: Fusion Layers = $FUSION_LAYERS"
        echo "输出目录: $OUTPUT_DIR"
        echo "日志文件: $LOG_FILE"
        echo "----------------------------------------"
        echo ""

        # 使用中期融合
        USE_MIDDLE_FUSION="True"
        MIDDLE_FUSION_LAYERS_ARG="--middle_fusion_layers $FUSION_LAYERS"

        # 运行训练
        python train_with_cross_modal_attention.py \
            --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
            --dataset jarvis \
            --property $PROPERTY \
        \
        --n_train 500 \
        --n_val 50 \
        --n_test 50 \
        \
        --batch_size 64 \
        --epochs 20 \
        --learning_rate 1e-3 \
        --weight_decay 5e-4 \
        --warmup_steps 500 \
        \
        --alignn_layers 4 \
        --gcn_layers 4 \
        --hidden_features 256 \
        --graph_dropout 0.15 \
        \
        --use_middle_fusion $USE_MIDDLE_FUSION \
        $MIDDLE_FUSION_LAYERS_ARG \
        --middle_fusion_hidden_dim 128 \
        --middle_fusion_num_heads 2 \
        --middle_fusion_dropout 0.1 \
        \
        --use_fine_grained_attention True \
        --fine_grained_hidden_dim 256 \
        --fine_grained_num_heads 8 \
        --fine_grained_dropout 0.2 \
        --fine_grained_use_projection True \
        \
        --use_cross_modal True \
        --cross_modal_num_heads 4 \
        --cross_modal_dropout 0.1 \
        \
        --early_stopping_patience 50 \
        --output_dir "$OUTPUT_DIR" \
        --num_workers 24 \
        --random_seed 123 \
        > "$LOG_FILE" 2>&1

        echo "✅ 训练完成: $PROPERTY - $CONFIG_NAME"
        echo ""

        # 提取结果
        echo "📊 提取结果..."

        # 从日志中提取最佳 MAE
        BEST_VAL_MAE=$(grep "Best_val_mae:" "$LOG_FILE" | tail -1 | awk '{print $2}' | sed 's/,//')
        BEST_TEST_MAE=$(grep "Best_test_mae:" "$LOG_FILE" | tail -1 | awk '{print $2}')

        # 从 fusion_weights.csv 中提取最终权重
        FUSION_WEIGHTS_FILE="$OUTPUT_DIR/$PROPERTY/fusion_weights.csv"

        if [ -f "$FUSION_WEIGHTS_FILE" ]; then
            # 读取最后一行（最终权重）
            LAST_LINE=$(tail -1 "$FUSION_WEIGHTS_FILE")

            # 提取各列（根据CSV格式调整）
            # 假设格式: epoch,layer_X_w_graph,layer_X_w_text,layer_X_eff_ratio
            # 我们取第一个 layer 的权重作为代表
            FINAL_W_GRAPH=$(echo "$LAST_LINE" | cut -d',' -f2)
            FINAL_W_TEXT=$(echo "$LAST_LINE" | cut -d',' -f3)
            FINAL_RATIO=$(echo "$LAST_LINE" | cut -d',' -f4)
        else
            FINAL_W_GRAPH="N/A"
            FINAL_W_TEXT="N/A"
            FINAL_RATIO="N/A"
        fi

        # 显示结果
        echo "  最佳验证 MAE: $BEST_VAL_MAE"
        echo "  最佳测试 MAE: $BEST_TEST_MAE"
        echo "  最终 w_graph: $FINAL_W_GRAPH"
        echo "  最终 w_text:  $FINAL_W_TEXT"
        echo "  图/文本比例: $FINAL_RATIO"
        echo ""

        # 保存到结果文件
        echo "$FUSION_LAYERS,$BEST_VAL_MAE,$BEST_TEST_MAE,$FINAL_W_GRAPH,$FINAL_W_TEXT,$FINAL_RATIO" >> "$RESULTS_FILE"

        echo "=========================================="
        echo ""

    done  # 结束 FUSION_LAYERS 循环

    # 属性汇总
    echo ""
    echo "=========================================="
    echo "✅ 属性 $PROPERTY 的所有配置测试完成！"
    echo "=========================================="
    echo ""

    echo "📊 结果汇总 ($PROPERTY):"
    echo ""
    column -t -s',' "$RESULTS_FILE"
    echo ""

    echo "🏆 最佳配置（按验证 MAE 排序）:"
    echo ""
    (head -1 "$RESULTS_FILE" && tail -n +2 "$RESULTS_FILE" | sort -t',' -k2 -n) | column -t -s','
    echo ""

    echo "=========================================="
    echo ""

done  # 结束 PROPERTY 循环

# 总体汇总
echo ""
echo "=========================================="
echo "✅ 所有属性和配置测试完成！"
echo "=========================================="
echo ""

echo "📁 详细结果位置: $BASE_OUTPUT_DIR/"
echo ""

# 为每个属性生成分析命令
for PROPERTY in "${PROPERTIES[@]}"; do
    echo "🔍 分析命令 ($PROPERTY):"
    echo ""
    echo "  # 对比所有配置:"
    echo "  python compare_search_results.py --search_dir $BASE_OUTPUT_DIR/$PROPERTY/"
    echo ""
    for FUSION_LAYERS in "${FUSION_LAYERS_LIST[@]}"; do
        CONFIG_NAME="layers_${FUSION_LAYERS//,/_}"
        echo "  # 查看 $FUSION_LAYERS 的权重演化:"
        echo "  python analyze_fusion_weights.py --output_dir $BASE_OUTPUT_DIR/$PROPERTY/$CONFIG_NAME/$PROPERTY/"
        echo ""
    done
    echo "=========================================="
    echo ""
done

echo "💡 下一步:"
echo "  1. 查看上面每个属性的排序结果"
echo "  2. 比较两种配置的性能差异"
echo "  3. 为每个属性选择最佳的 fusion_layers 配置"
echo "  4. 使用最佳配置进行后续实验"
echo ""
