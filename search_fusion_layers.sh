#!/bin/bash
# 融合层位置搜索 - 阶段1：快速筛选
# 测试不同的 middle_fusion_layers 配置 + 无融合基线

echo "=========================================="
echo "DynamicFusionModule - 融合层位置搜索"
echo "数据集: JARVIS MBJ Band Gap"
echo "策略: 小数据快速筛选 (500 样本, 20 epochs)"
echo "=========================================="
echo ""

# 环境配置
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 基础输出目录
BASE_OUTPUT_DIR="./fusion_layer_search"
mkdir -p "$BASE_OUTPUT_DIR"

# 结果文件
RESULTS_FILE="$BASE_OUTPUT_DIR/results_summary.csv"
echo "fusion_layers,best_val_mae,best_test_mae,final_w_graph,final_w_text,ratio" > "$RESULTS_FILE"

# 测试配置列表
FUSION_LAYERS_LIST=(
    "none"     # 基线：不使用中期融合（DynamicFusionModule）
    "1"        # 早期融合（第1层）
    "2"        # 中期融合（第2层）- 你的原始配置
    "3"        # 后期融合（第3层）
    "2,3"      # 双层融合（第2和第3层）
    "1,2,3"    # 全层融合（第1、2、3层）
)

echo "测试配置:"
for layers in "${FUSION_LAYERS_LIST[@]}"; do
    if [ "$layers" == "none" ]; then
        echo "  - 基线: 无中期融合 (DynamicFusionModule)"
    else
        echo "  - Fusion layers: $layers"
    fi
done
echo ""
echo "=========================================="
echo ""

# 遍历每个配置
for FUSION_LAYERS in "${FUSION_LAYERS_LIST[@]}"; do

    # 创建配置特定的输出目录
    if [ "$FUSION_LAYERS" == "none" ]; then
        CONFIG_NAME="baseline_no_fusion"
    else
        CONFIG_NAME="layers_${FUSION_LAYERS//,/_}"  # 将逗号替换为下划线
    fi

    OUTPUT_DIR="$BASE_OUTPUT_DIR/$CONFIG_NAME"
    LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "$OUTPUT_DIR"

    echo "----------------------------------------"
    if [ "$FUSION_LAYERS" == "none" ]; then
        echo "🧪 测试配置: 基线（无中期融合）"
    else
        echo "🧪 测试配置: Fusion Layers = $FUSION_LAYERS"
    fi
    echo "输出目录: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
    echo "----------------------------------------"
    echo ""

    # 根据配置决定参数
    if [ "$FUSION_LAYERS" == "none" ]; then
        # 基线：不使用中期融合
        USE_MIDDLE_FUSION="False"
        MIDDLE_FUSION_LAYERS_ARG=""
    else
        # 使用中期融合
        USE_MIDDLE_FUSION="True"
        MIDDLE_FUSION_LAYERS_ARG="--middle_fusion_layers $FUSION_LAYERS"
    fi

    # 运行训练
    python train_with_cross_modal_attention.py \
        --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
        --dataset jarvis \
        --property mbj_bandgap \
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

    echo "✅ 训练完成: $CONFIG_NAME"
    echo ""

    # 提取结果
    echo "📊 提取结果..."

    # 从日志中提取最佳 MAE
    BEST_VAL_MAE=$(grep "Best_val_mae:" "$LOG_FILE" | tail -1 | awk '{print $2}' | sed 's/,//')
    BEST_TEST_MAE=$(grep "Best_test_mae:" "$LOG_FILE" | tail -1 | awk '{print $2}')

    # 从 fusion_weights.csv 中提取最终权重
    FUSION_WEIGHTS_FILE="$OUTPUT_DIR/mbj_bandgap/fusion_weights.csv"

    if [ "$FUSION_LAYERS" != "none" ] && [ -f "$FUSION_WEIGHTS_FILE" ]; then
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
    if [ "$FINAL_W_GRAPH" != "N/A" ]; then
        echo "  最终 w_graph: $FINAL_W_GRAPH"
        echo "  最终 w_text:  $FINAL_W_TEXT"
        echo "  图/文本比例: $FINAL_RATIO"
    fi
    echo ""

    # 保存到结果文件
    echo "$FUSION_LAYERS,$BEST_VAL_MAE,$BEST_TEST_MAE,$FINAL_W_GRAPH,$FINAL_W_TEXT,$FINAL_RATIO" >> "$RESULTS_FILE"

    echo "=========================================="
    echo ""

done

# 最终汇总
echo ""
echo "=========================================="
echo "✅ 所有配置测试完成！"
echo "=========================================="
echo ""

echo "📊 结果汇总:"
echo ""
column -t -s',' "$RESULTS_FILE"
echo ""

echo "🏆 最佳配置（按验证 MAE 排序）:"
echo ""
(head -1 "$RESULTS_FILE" && tail -n +2 "$RESULTS_FILE" | sort -t',' -k2 -n) | column -t -s','
echo ""

echo "📊 与基线对比:"
echo ""
BASELINE_MAE=$(grep "^none," "$RESULTS_FILE" | cut -d',' -f2)
if [ -n "$BASELINE_MAE" ]; then
    echo "  基线（无融合）MAE: $BASELINE_MAE"
    echo ""
    echo "  各配置相对基线的提升:"
    echo ""

    while IFS=',' read -r layers val_mae test_mae w_g w_t ratio; do
        if [ "$layers" != "none" ] && [ "$layers" != "fusion_layers" ]; then
            # 计算相对提升（百分比）
            improvement=$(echo "scale=2; ($BASELINE_MAE - $val_mae) / $BASELINE_MAE * 100" | bc)

            if (( $(echo "$improvement > 0" | bc -l) )); then
                echo "    Layers $layers: ↓ $improvement% (更好)"
            elif (( $(echo "$improvement < 0" | bc -l) )); then
                improvement_abs=$(echo "$improvement * -1" | bc)
                echo "    Layers $layers: ↑ $improvement_abs% (更差)"
            else
                echo "    Layers $layers: 持平"
            fi
        fi
    done < "$RESULTS_FILE"
fi
echo ""

echo "=========================================="
echo ""

echo "📁 详细结果位置: $BASE_OUTPUT_DIR/"
echo ""

echo "🔍 分析命令:"
echo ""
echo "  # 对比所有配置:"
echo "  python compare_search_results.py --search_dir $BASE_OUTPUT_DIR/"
echo ""
for FUSION_LAYERS in "${FUSION_LAYERS_LIST[@]}"; do
    if [ "$FUSION_LAYERS" == "none" ]; then
        CONFIG_NAME="baseline_no_fusion"
        echo "  # 查看基线（无融合）:"
        echo "  cat $BASE_OUTPUT_DIR/$CONFIG_NAME/train_*.log | grep 'MAE'"
    else
        CONFIG_NAME="layers_${FUSION_LAYERS//,/_}"
        echo "  # 查看 $FUSION_LAYERS 的权重演化:"
        echo "  python analyze_fusion_weights.py --output_dir $BASE_OUTPUT_DIR/$CONFIG_NAME/mbj_bandgap/"
    fi
    echo ""
done

echo "=========================================="
echo ""

echo "💡 下一步:"
echo "  1. 查看上面的排序结果和与基线的对比"
echo "  2. 如果所有融合配置都优于基线 → 融合有效！"
echo "  3. 选择最佳的 fusion_layers 配置"
echo "  4. 使用最佳配置进行阶段2（中等数据精细调整）"
echo ""
