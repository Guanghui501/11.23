#!/bin/bash
# MBJ Band Gap 数据预处理快捷脚本
# 用法: 在你的服务器上运行此脚本

# ==================== 路径配置 ====================
# 数据集根目录（包含jarvis/mbj_bandgap/cif/和description.csv）
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"

# 预处理数据输出目录
OUTPUT_DIR="/public/home/ghzhang/preprocessed_data"

# 数据集和属性
DATASET="jarvis"
PROPERTY="mbj_bandgap"

# ==================== 图构建参数 ====================
CUTOFF=8.0
MAX_NEIGHBORS=12

# ==================== 数据分割参数 ====================
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
SEED=42

echo "=========================================="
echo "MBJ Band Gap 数据预处理"
echo "=========================================="
echo ""
echo "配置:"
echo "  数据集根目录: $ROOT_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  数据集: $DATASET"
echo "  属性: $PROPERTY"
echo ""
echo "图构建参数:"
echo "  Cutoff: $CUTOFF Å"
echo "  Max neighbors: $MAX_NEIGHBORS"
echo ""
echo "数据分割:"
echo "  训练集: ${TRAIN_RATIO} (80%)"
echo "  验证集: ${VAL_RATIO} (10%)"
echo "  测试集: ${TEST_RATIO} (10%)"
echo "  随机种子: $SEED"
echo "=========================================="
echo ""

# 检查数据集目录
if [ ! -d "$ROOT_DIR/jarvis/mbj_bandgap" ]; then
    echo "❌ 错误: 数据集目录不存在"
    echo "   路径: $ROOT_DIR/jarvis/mbj_bandgap"
    echo ""
    echo "请检查:"
    echo "  1. ROOT_DIR 路径是否正确"
    echo "  2. 数据集是否已下载"
    echo ""
    echo "预期目录结构:"
    echo "  $ROOT_DIR/"
    echo "    jarvis/"
    echo "      mbj_bandgap/"
    echo "        cif/"
    echo "          JVASP-1.cif"
    echo "          JVASP-2.cif"
    echo "          ..."
    echo "        description.csv"
    exit 1
fi

echo "✓ 数据集目录: $ROOT_DIR/jarvis/mbj_bandgap"
echo ""

# 检查预处理脚本
if [ ! -f "./preprocess_dataset.py" ]; then
    echo "❌ 错误: 预处理脚本不存在"
    echo "   路径: ./preprocess_dataset.py"
    echo ""
    echo "请确保 preprocess_dataset.py 在当前目录"
    exit 1
fi

echo "✓ 预处理脚本: ./preprocess_dataset.py"
echo ""

# 提示用户确认
echo "开始预处理..."
echo "预计时间: 取决于数据集大小，可能需要10-30分钟"
echo ""
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "运行预处理"
echo "=========================================="
echo ""

# 运行预处理
python preprocess_dataset.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --root_dir "$ROOT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --cutoff "$CUTOFF" \
    --max_neighbors "$MAX_NEIGHBORS" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "预处理完成！"
    echo "=========================================="
    echo ""
    echo "预处理数据已保存到:"
    echo "  $OUTPUT_DIR/jarvis/mbj_bandgap/"
    echo ""
    echo "包含文件:"
    echo "  - train.pkl (训练集)"
    echo "  - val.pkl   (验证集)"
    echo "  - test.pkl  (测试集)"
    echo "  - README.txt (数据说明)"
    echo ""
    echo "接下来你可以:"
    echo ""
    echo "1. 运行文本遮挡评估:"
    echo "   ./quick_test_masking.sh"
    echo "   或"
    echo "   ./run_masking_eval_mbj_bandgap.sh"
    echo ""
    echo "2. 训练模型 (使用预处理数据):"
    echo "   python train_with_cross_modal_attention.py \\"
    echo "       --dataset jarvis \\"
    echo "       --property mbj_bandgap \\"
    echo "       --use_preprocessed True \\"
    echo "       --preprocessed_dir $OUTPUT_DIR"
    echo ""
else
    echo ""
    echo "❌ 预处理失败"
    echo ""
    echo "可能的原因:"
    echo "  1. 数据集路径不正确"
    echo "  2. CIF文件缺失或格式错误"
    echo "  3. 内存不足"
    echo "  4. 依赖包未安装"
    echo ""
    echo "请检查上面的错误信息"
fi
