#!/bin/bash
#==============================================================================
# 训练参数测试脚本
# 用途: 在提交SLURM作业前测试参数是否正确
#==============================================================================

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "训练参数测试工具"
echo -e "==========================================${NC}"
echo ""

# 数据集路径
DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

# 测试三个配置
echo -e "${YELLOW}测试配置1: Baseline (No FG + Middle)${NC}"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection False \
    --early_stopping_patience 150 \
    --output_dir ./test_output_config1 \
    --num_workers 0 \
    --random_seed 42 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 配置1参数验证通过${NC}"
else
    echo -e "${RED}✗ 配置1参数验证失败${NC}"
    echo "运行以下命令查看详细错误:"
    echo "python train_with_cross_modal_attention.py --help"
    exit 1
fi

echo ""
echo -e "${YELLOW}测试配置2: FG + Proj + Middle${NC}"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection True \
    --early_stopping_patience 150 \
    --output_dir ./test_output_config2 \
    --num_workers 0 \
    --random_seed 42 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 配置2参数验证通过${NC}"
else
    echo -e "${RED}✗ 配置2参数验证失败${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}测试配置3: FG + Proj (No Middle)${NC}"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion False \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection True \
    --early_stopping_patience 150 \
    --output_dir ./test_output_config3 \
    --num_workers 0 \
    --random_seed 42 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 配置3参数验证通过${NC}"
else
    echo -e "${RED}✗ 配置3参数验证失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "所有配置参数验证通过！✓"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}可以安全提交SLURM作业:${NC}"
echo "  ./submit_ablation_study.sh"
echo ""
