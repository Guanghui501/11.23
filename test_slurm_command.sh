#!/bin/bash
# 测试SLURM脚本中实际使用的命令
# 直接从submit_ablation_study.sh复制的参数

echo "=========================================="
echo "测试SLURM脚本中的实际命令"
echo "=========================================="
echo ""

# 配置（从submit_ablation_study.sh复制）
DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"
property="mbj_bandgap"
seed=42
output_dir="./test_slurm_output"
use_fg="False"
use_proj="False"
use_middle="True"

echo "配置1参数: FG=$use_fg, Proj=$use_proj, Middle=$use_middle"
echo ""

# 完全复制submit_ablation_study.sh第179-208行的命令
echo "执行命令 (带--help验证参数):"
python train_with_cross_modal_attention.py \
    --root_dir ${DATA_ROOT} \
    --dataset jarvis \
    --property ${property} \
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
    --use_middle_fusion ${use_middle} \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention ${use_fg} \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection ${use_proj} \
    --early_stopping_patience 150 \
    --output_dir ${output_dir} \
    --num_workers 24 \
    --random_seed ${seed} \
    --help 2>&1 | head -20

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 参数验证成功 (退出码: $EXIT_CODE)"
else
    echo "✗ 参数验证失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "重新运行显示完整错误:"
    python train_with_cross_modal_attention.py \
        --root_dir ${DATA_ROOT} \
        --dataset jarvis \
        --property ${property} \
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
        --use_middle_fusion ${use_middle} \
        --middle_fusion_layers 2 \
        --use_fine_grained_attention ${use_fg} \
        --middle_fusion_dropout 0.35 \
        --fine_grained_hidden_dim 256 \
        --fine_grained_num_heads 8 \
        --fine_grained_dropout 0.35 \
        --fine_grained_use_projection ${use_proj} \
        --early_stopping_patience 150 \
        --output_dir ${output_dir} \
        --num_workers 24 \
        --random_seed ${seed} 2>&1
fi

echo ""
echo "现在测试配置2: FG=True, Proj=True, Middle=True"
use_fg="True"
use_proj="True"
use_middle="True"

python train_with_cross_modal_attention.py \
    --root_dir ${DATA_ROOT} \
    --dataset jarvis \
    --property ${property} \
    --use_middle_fusion ${use_middle} \
    --use_fine_grained_attention ${use_fg} \
    --fine_grained_use_projection ${use_proj} \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 配置2参数OK"
else
    echo "✗ 配置2参数失败"
fi

echo ""
echo "测试配置3: FG=True, Proj=True, Middle=False"
use_middle="False"

python train_with_cross_modal_attention.py \
    --root_dir ${DATA_ROOT} \
    --dataset jarvis \
    --property ${property} \
    --use_middle_fusion ${use_middle} \
    --use_fine_grained_attention ${use_fg} \
    --fine_grained_use_projection ${use_proj} \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 配置3参数OK"
else
    echo "✗ 配置3参数失败"
fi
