#!/bin/bash
#SBATCH --job-name=crystal_train          # 作业名称
#SBATCH --partition=gpu                   # 分区名称（根据你的集群修改）
#SBATCH --nodes=1                         # 节点数
#SBATCH --ntasks=1                        # 任务数
#SBATCH --cpus-per-task=24                # 每个任务的CPU核心数
#SBATCH --gres=gpu:1                      # 每个任务使用1个GPU
#SBATCH --mem=64G                         # 内存
#SBATCH --time=48:00:00                   # 最大运行时间 (48小时)
#SBATCH --array=0-5                       # 作业数组: 2个属性 × 3个种子 = 6个任务
#SBATCH --output=logs/slurm_%A_%a.out     # 标准输出 (%A=作业ID, %a=数组索引)
#SBATCH --error=logs/slurm_%A_%a.err      # 标准错误

# ============================================
# 配置部分
# ============================================

# 定义要训练的属性列表
PROPERTIES=("bulk_modulus_kv" "shear_modulus_gv")

# 定义随机种子列表
RANDOM_SEEDS=(42 7 123)

# Hugging Face镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建日志目录
mkdir -p logs

# ============================================
# 计算当前任务的属性和种子
# ============================================

# 总共有 2个属性 × 3个种子 = 6个任务
# SLURM_ARRAY_TASK_ID 范围: 0-5

NUM_PROPERTIES=${#PROPERTIES[@]}  # 2
NUM_SEEDS=${#RANDOM_SEEDS[@]}     # 3

# 计算当前任务对应的属性和种子索引
PROPERTY_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

# 获取实际的属性名和种子值
PROPERTY=${PROPERTIES[$PROPERTY_IDX]}
SEED=${RANDOM_SEEDS[$SEED_IDX]}

# 创建输出目录
OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_onlymiddle_${PROPERTY}_quantext"
mkdir -p "$OUTPUT_DIR"

# ============================================
# 打印任务信息
# ============================================

echo "=========================================="
echo "SLURM 作业信息:"
echo "  作业ID: $SLURM_JOB_ID"
echo "  数组任务ID: $SLURM_ARRAY_TASK_ID"
echo "  节点: $SLURM_NODELIST"
echo "  GPU设备: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo "训练任务信息:"
echo "  属性: $PROPERTY"
echo "  随机种子: $SEED"
echo "  输出目录: $OUTPUT_DIR"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ============================================
# 激活环境（根据你的环境修改）
# ============================================

# 如果使用conda环境，取消下面的注释并修改环境名
# source ~/.bashrc
# conda activate your_env_name

# 如果使用模块系统，取消下面的注释
# module load cuda/11.8
# module load python/3.9

# ============================================
# 运行训练
# ============================================

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main-2/dataset \
    --dataset jarvis \
    --property "$PROPERTY" \
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
    --output_dir "$OUTPUT_DIR" \
    --num_workers 24 \
    --random_seed "$SEED"

# ============================================
# 记录完成信息
# ============================================

EXIT_CODE=$?
echo "=========================================="
echo "任务完成信息:"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  退出码: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
