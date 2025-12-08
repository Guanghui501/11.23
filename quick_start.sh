#!/bin/bash
# 快速启动脚本 - 训练方案1和方案2

echo "========================================="
echo "方案1和方案2快速训练脚本"
echo "========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

echo "检测到的Python版本:"
python --version
echo ""

# 选择方案
echo "请选择训练方案:"
echo "1) 方案1: 中期+细粒度+门控融合 (推荐，最高效)"
echo "2) 方案2: 中期+细粒度+单向注意力"
echo "3) 两个方案都训练（用于对比）"
read -p "输入选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "========================================="
        echo "开始训练方案1"
        echo "========================================="
        python train_solution1.py \
            --dataset jarvis \
            --property formation_energy \
            --batch_size 64 \
            --epochs 300 \
            --learning_rate 0.001 \
            --output_dir experiments/solution1
        ;;
    2)
        echo ""
        echo "========================================="
        echo "开始训练方案2"
        echo "========================================="
        python train_solution2.py \
            --dataset jarvis \
            --property formation_energy \
            --batch_size 64 \
            --epochs 300 \
            --learning_rate 0.001 \
            --output_dir experiments/solution2
        ;;
    3)
        echo ""
        echo "========================================="
        echo "开始训练方案1"
        echo "========================================="
        python train_solution1.py \
            --dataset jarvis \
            --property formation_energy \
            --batch_size 64 \
            --epochs 300 \
            --output_dir experiments/solution1 &
        PID1=$!

        echo ""
        echo "========================================="
        echo "开始训练方案2"
        echo "========================================="
        python train_solution2.py \
            --dataset jarvis \
            --property formation_energy \
            --batch_size 64 \
            --epochs 300 \
            --output_dir experiments/solution2 &
        PID2=$!

        echo ""
        echo "两个训练任务已在后台启动"
        echo "方案1 PID: $PID1"
        echo "方案2 PID: $PID2"
        echo "使用 'tail -f experiments/solution1/train.log' 查看日志"

        wait $PID1 $PID2
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
echo "结果保存在 experiments/ 目录"
echo "使用以下命令查看结果:"
echo "  cat experiments/solution1/model_config.json"
echo "  cat experiments/solution1/gate_history.json"
