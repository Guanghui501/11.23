#!/bin/bash
# 测试heredoc中的命令是否正确

echo "测试heredoc中的Python命令..."

DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"
property="mbj_bandgap"
seed=42
use_fg="False"
use_proj="False"
use_middle="True"
output_dir="./test_heredoc"

# 模拟sbatch heredoc
TEST_SCRIPT=$(cat <<'OUTER_EOF'
#!/bin/bash
echo "测试开始..."

# 这里模拟submit_ablation_study.sh中的Python命令
python train_with_cross_modal_attention.py --root_dir ${DATA_ROOT} --dataset jarvis --property ${property} --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --batch_size 64 --epochs 100 --learning_rate 5e-4 --weight_decay 1e-3 --warmup_steps 2000 --alignn_layers 4 --gcn_layers 4 --hidden_features 256 --graph_dropout 0.15 --use_cross_modal False --cross_modal_num_heads 2 --use_middle_fusion ${use_middle} --middle_fusion_layers 2 --use_fine_grained_attention ${use_fg} --middle_fusion_dropout 0.35 --fine_grained_hidden_dim 256 --fine_grained_num_heads 8 --fine_grained_dropout 0.35 --fine_grained_use_projection ${use_proj} --early_stopping_patience 150 --output_dir ${output_dir} --num_workers 24 --random_seed ${seed} --help

echo "退出码: $?"
OUTER_EOF
)

# 展开变量
eval "cat <<EOF
$TEST_SCRIPT
EOF
" > /tmp/test_heredoc.sh

chmod +x /tmp/test_heredoc.sh

echo "生成的脚本:"
echo "================================"
cat /tmp/test_heredoc.sh
echo "================================"
echo ""
echo "执行测试..."
/tmp/test_heredoc.sh 2>&1 | head -30

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✓ heredoc测试成功！"
else
    echo ""
    echo "✗ heredoc测试失败"
fi
