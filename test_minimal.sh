#!/bin/bash
# 最小参数集测试 - 逐步添加参数找出问题

echo "=========================================="
echo "最小参数测试"
echo "=========================================="
echo ""

DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

echo "步骤1: 测试只有必需参数"
echo "--------------------"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --help 2>&1 | grep -E "(error|unrecognized)" || echo "✓ 基础参数OK"

echo ""
echo "步骤2: 添加训练参数"
echo "--------------------"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --help 2>&1 | grep -E "(error|unrecognized)" || echo "✓ 训练参数OK"

echo ""
echo "步骤3: 添加模型参数"
echo "--------------------"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.15 \
    --help 2>&1 | grep -E "(error|unrecognized)" || echo "✓ 模型参数OK"

echo ""
echo "步骤4: 添加融合参数"
echo "--------------------"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal False \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --middle_fusion_dropout 0.35 \
    --help 2>&1 | grep -E "(error|unrecognized)" || echo "✓ 融合参数OK"

echo ""
echo "步骤5: 添加Fine-grained参数"
echo "--------------------"
python train_with_cross_modal_attention.py \
    --root_dir "$DATA_ROOT" \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_fine_grained_attention False \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection False \
    --help 2>&1 | grep -E "(error|unrecognized)" || echo "✓ Fine-grained参数OK"

echo ""
echo "步骤6: 完整参数测试（不实际运行，只验证参数）"
echo "--------------------"

# 打印完整的命令用于调试
cat > /tmp/test_cmd.sh <<'CMDEOF'
python train_with_cross_modal_attention.py \
    --root_dir "/public/home/ghzhang/crysmmnet-main-2/dataset" \
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
    --output_dir ./test_output \
    --num_workers 0 \
    --random_seed 42
CMDEOF

echo "生成的命令保存在 /tmp/test_cmd.sh"
echo "参数数量: $(grep -o '\-\-' /tmp/test_cmd.sh | wc -l)"

bash /tmp/test_cmd.sh --help 2>&1 | grep -E "(error|unrecognized)"
if [ $? -eq 0 ]; then
    echo "✗ 发现参数错误"
    echo ""
    echo "重新运行显示完整错误:"
    bash /tmp/test_cmd.sh 2>&1 | head -100
else
    echo "✓ 完整参数验证通过"
fi
