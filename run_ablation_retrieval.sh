#!/bin/bash

###############################################################################
# 消融实验：比较不同融合策略的检索性能
# 用于验证中期融合、细粒度注意力等机制对图-文本对齐的作用
###############################################################################

set -e

# ======================== 配置 ========================

# 数据集配置
SPLIT="val"
MAX_SAMPLES=1000
K_VALUES="1 5 10"
DEVICE="cuda"

# 输出目录
BASE_OUTPUT_DIR="./retrieval_ablation_results"
mkdir -p "$BASE_OUTPUT_DIR"

# 要评估的模型列表
declare -A MODELS=(
    ["no_fusion"]="checkpoints/no_fusion_best.pt"
    ["middle_fusion"]="checkpoints/middle_fusion_best.pt"
    ["cross_modal"]="checkpoints/cross_modal_best.pt"
    ["fine_grained"]="checkpoints/fine_grained_best.pt"
    ["full_model"]="checkpoints/full_model_best.pt"
)

# 模型标签（用于报告）
declare -A LABELS=(
    ["no_fusion"]="基线（无融合）"
    ["middle_fusion"]="中期融合"
    ["cross_modal"]="跨模态注意力"
    ["fine_grained"]="细粒度注意力"
    ["full_model"]="完整模型"
)

# ======================== 函数定义 ========================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

evaluate_model() {
    local model_key=$1
    local checkpoint=${MODELS[$model_key]}
    local label=${LABELS[$model_key]}
    local output_dir="$BASE_OUTPUT_DIR/$model_key"

    print_header "🔍 评估: $label"

    if [ ! -f "$checkpoint" ]; then
        echo "⚠️  跳过（文件不存在）: $checkpoint"
        return
    fi

    echo "检查点: $checkpoint"
    echo "输出目录: $output_dir"
    echo ""

    python evaluate_retrieval.py \
        --checkpoint "$checkpoint" \
        --split "$SPLIT" \
        --max_samples $MAX_SAMPLES \
        --k_values $K_VALUES \
        --output_dir "$output_dir" \
        --device "$DEVICE" \
        --no_visualize  # 不生成每个模型的图表（最后统一生成对比图）

    echo "✅ 完成: $label"
}

# ======================== 主流程 ========================

print_header "🎯 消融实验：检索性能评估"

echo "📋 评估配置:"
echo "  - 数据集: $SPLIT"
echo "  - 样本数: $MAX_SAMPLES"
echo "  - K 值: $K_VALUES"
echo "  - 输出目录: $BASE_OUTPUT_DIR"
echo ""
echo "🔬 待评估模型:"
for key in "${!MODELS[@]}"; do
    echo "  - $key: ${LABELS[$key]}"
done
echo ""

# 评估每个模型
for key in no_fusion middle_fusion cross_modal fine_grained full_model; do
    if [[ -n "${MODELS[$key]}" ]]; then
        evaluate_model "$key"
    fi
done

# ======================== 汇总结果 ========================

print_header "📊 汇总所有结果"

SUMMARY_FILE="$BASE_OUTPUT_DIR/summary.txt"
SUMMARY_JSON="$BASE_OUTPUT_DIR/summary.json"

echo "生成汇总报告..."
echo ""

# 创建汇总报告
{
    echo "============================================================"
    echo "消融实验：检索性能汇总"
    echo "============================================================"
    echo ""
    echo "模型配置                      R@1       R@5       R@10"
    echo "------------------------------------------------------------"
} > "$SUMMARY_FILE"

# 创建 JSON 汇总
echo "{" > "$SUMMARY_JSON"
echo "  \"models\": {" >> "$SUMMARY_JSON"

first=true
for key in no_fusion middle_fusion cross_modal fine_grained full_model; do
    result_file="$BASE_OUTPUT_DIR/$key/retrieval_results.json"

    if [ -f "$result_file" ]; then
        label=${LABELS[$key]}

        # 提取 R@1, R@5, R@10
        r1=$(python -c "import json; d=json.load(open('$result_file')); print(f\"{d['metrics']['avg_R@1']*100:.2f}\")")
        r5=$(python -c "import json; d=json.load(open('$result_file')); print(f\"{d['metrics']['avg_R@5']*100:.2f}\")")
        r10=$(python -c "import json; d=json.load(open('$result_file')); print(f\"{d['metrics']['avg_R@10']*100:.2f}\")")

        # 写入文本报告
        printf "%-30s %6s%%   %6s%%   %6s%%\n" "$label" "$r1" "$r5" "$r10" >> "$SUMMARY_FILE"

        # 写入 JSON
        if [ "$first" = false ]; then
            echo "," >> "$SUMMARY_JSON"
        fi
        first=false

        echo "    \"$key\": {" >> "$SUMMARY_JSON"
        echo "      \"label\": \"$label\"," >> "$SUMMARY_JSON"
        echo "      \"R@1\": $r1," >> "$SUMMARY_JSON"
        echo "      \"R@5\": $r5," >> "$SUMMARY_JSON"
        echo "      \"R@10\": $r10" >> "$SUMMARY_JSON"
        echo -n "    }" >> "$SUMMARY_JSON"
    fi
done

echo "" >> "$SUMMARY_JSON"
echo "  }" >> "$SUMMARY_JSON"
echo "}" >> "$SUMMARY_JSON"

# 显示汇总
cat "$SUMMARY_FILE"
echo ""

# ======================== 生成对比图 ========================

print_header "📊 生成对比可视化"

# 创建 Python 脚本生成对比图
cat > "$BASE_OUTPUT_DIR/plot_comparison.py" <<'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# 读取汇总数据
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

models = data['models']
labels = [models[k]['label'] for k in models.keys()]
r1_values = [models[k]['R@1'] for k in models.keys()]
r5_values = [models[k]['R@5'] for k in models.keys()]
r10_values = [models[k]['R@10'] for k in models.keys()]

# 创建图表
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width, r1_values, width, label='R@1',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x, r5_values, width, label='R@5',
               color='coral', alpha=0.8)
bars3 = ax.bar(x + width, r10_values, width, label='R@10',
               color='mediumseagreen', alpha=0.8)

ax.set_xlabel('模型配置', fontsize=13, fontweight='bold')
ax.set_ylabel('Recall@K (%)', fontsize=13, fontweight='bold')
ax.set_title('消融实验：不同融合策略的检索性能对比',
            fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# 添加数值标签
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.tight_layout()
plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight')
print(f"✅ 对比图已保存: {sys.argv[2]}")
EOF

python "$BASE_OUTPUT_DIR/plot_comparison.py" \
    "$SUMMARY_JSON" \
    "$BASE_OUTPUT_DIR/retrieval_comparison.png"

# ======================== 完成 ========================

print_header "🎉 消融实验完成！"

echo "📁 结果文件:"
echo "  - 汇总报告: $SUMMARY_FILE"
echo "  - JSON 数据: $SUMMARY_JSON"
echo "  - 对比图: $BASE_OUTPUT_DIR/retrieval_comparison.png"
echo ""
echo "📊 详细结果:"
ls -lh "$BASE_OUTPUT_DIR"/*/retrieval_results.json
echo ""

# 分析结论
print_header "💡 关键发现"

cat <<EOF
基于检索性能的分析：

1️⃣  如果 "中期融合" 比 "无融合" 的 R@1 高 20%+：
   ✅ 中期融合显著提高了图-文本对齐能力

2️⃣  如果 "细粒度注意力" 比 "中期融合" 的 R@1 高 10%+：
   ✅ 原子级别的对齐进一步增强了检索能力

3️⃣  如果 "完整模型" 的 R@1 达到 80%+：
   🏆 模型成功实现了强对齐，可以投入使用

4️⃣  如果所有模型的 R@1 都 <40%：
   ⚠️  需要检查：
      - 对比学习损失是否启用
      - 特征投影维度是否匹配
      - 训练是否充分收敛

查看详细报告: cat $SUMMARY_FILE
EOF

echo ""
echo "✨ Done!"
