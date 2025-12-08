#!/bin/bash
#==============================================================================
# SLURM 消融实验批量提交脚本
# 用途: 测试Fine-grained Attention和Projection对模型性能的影响
#==============================================================================

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

#==============================================================================
# 配置部分
#==============================================================================

# 训练属性和随机种子
PROPERTIES=("mbj_bandgap")
RANDOM_SEEDS=(42)

# SLURM资源配置
SLURM_PARTITION=""              # 留空则使用默认分区，或设置如 "gpu"
SLURM_TIME="48:00:00"           # 最大运行时间
SLURM_GPUS=1                    # GPU数量
SLURM_NODES=1                   # 节点数
SLURM_NTASKS=1                  # 任务数

# Conda环境
CONDA_ENV="sganet"

# 数据集路径
DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

# 共同训练参数
COMMON_PARAMS=(
    --root_dir "$DATA_ROOT"
    --dataset jarvis
    --train_ratio 0.8
    --val_ratio 0.1
    --test_ratio 0.1
    --batch_size 64
    --epochs 100
    --learning_rate 5e-4
    --weight_decay 1e-3
    --warmup_steps 2000
    --alignn_layers 4
    --gcn_layers 4
    --hidden_features 256
    --graph_dropout 0.15
    --use_cross_modal False
    --cross_modal_num_heads 2
    --middle_fusion_layers 2
    --middle_fusion_dropout 0.35
    --fine_grained_hidden_dim 256
    --fine_grained_num_heads 8
    --fine_grained_dropout 0.35
    --early_stopping_patience 150
    --num_workers 24
)

#==============================================================================
# 消融实验配置定义
#==============================================================================

# 配置1: 基线（无Fine-grained, 有Middle fusion）
CONFIG_1_NAME="baseline"
CONFIG_1_DESC="Baseline: No Fine-grained + With Middle fusion"
CONFIG_1_SUFFIX="onlymiddle"
CONFIG_1_PARAMS=(
    --use_fine_grained_attention False
    --fine_grained_use_projection False
    --use_middle_fusion True
)

# 配置2: Fine-grained + Projection + Middle fusion
CONFIG_2_NAME="fg_proj_middle"
CONFIG_2_DESC="Fine-grained + Projection + Middle fusion"
CONFIG_2_SUFFIX="middle_fg_proj"
CONFIG_2_PARAMS=(
    --use_fine_grained_attention True
    --fine_grained_use_projection True
    --use_middle_fusion True
)

# 配置3: Fine-grained + Projection, 无Middle fusion
CONFIG_3_NAME="fg_proj_nomiddle"
CONFIG_3_DESC="Fine-grained + Projection, No Middle fusion"
CONFIG_3_SUFFIX="fg_proj_nomiddle"
CONFIG_3_PARAMS=(
    --use_fine_grained_attention True
    --fine_grained_use_projection True
    --use_middle_fusion False
)

#==============================================================================
# 函数定义
#==============================================================================

# 提交单个SLURM作业的函数
# 参数: $1=作业名, $2=输出目录, $3=属性, $4=种子, $5=配置参数数组名, $6=依赖作业ID(可选)
submit_job() {
    local job_name=$1
    local output_dir=$2
    local property=$3
    local seed=$4
    local config_params_name=$5
    local dependency_id=$6

    # 创建输出目录
    mkdir -p "$output_dir"

    # 构建依赖参数
    local dependency_flag=""
    if [ -n "$dependency_id" ]; then
        dependency_flag="--dependency=afterok:${dependency_id}"
    fi

    # 构建分区参数
    local partition_flag=""
    if [ -n "$SLURM_PARTITION" ]; then
        partition_flag="#SBATCH -p ${SLURM_PARTITION}"
    fi

    # 获取配置参数（通过名称引用）
    local -n config_params=$config_params_name

    # 提交作业
    local job_submit=$(sbatch $dependency_flag <<EOF
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N ${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH -o ${output_dir}/%x-%j.out
#SBATCH -e ${output_dir}/%x-%j.err
#SBATCH -t ${SLURM_TIME}
${partition_flag}

#==============================================================================
# 作业执行脚本
#==============================================================================

# 清理模块环境
module purge

# 激活Conda环境
source ~/.bashrc
conda activate ${CONDA_ENV}

# 环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# 打印作业信息
echo "=========================================="
echo "SLURM 作业信息"
echo "=========================================="
echo "作业 ID:       \${SLURM_JOB_ID}"
echo "作业名称:      \${SLURM_JOB_NAME}"
echo "节点:          \${SLURM_NODELIST}"
echo "GPU:           \${CUDA_VISIBLE_DEVICES}"
echo "开始时间:      \$(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "训练配置:"
echo "  属性:        ${property}"
echo "  随机种子:    ${seed}"
echo "  输出目录:    ${output_dir}"
echo "=========================================="
echo ""

# 执行训练
python train_with_cross_modal_attention.py \\
    ${COMMON_PARAMS[@]} \\
    --property "${property}" \\
    --random_seed "${seed}" \\
    --output_dir "${output_dir}" \\
    ${config_params[@]}

# 记录完成状态
EXIT_CODE=\$?
echo ""
echo "=========================================="
echo "作业完成信息"
echo "=========================================="
echo "结束时间:      \$(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码:        \${EXIT_CODE}"
echo "=========================================="

exit \${EXIT_CODE}
EOF
)

    # 提取并返回作业ID
    local job_id=$(echo "$job_submit" | grep -oP 'Submitted batch job \K\d+')
    echo "$job_id"
}

# 打印配置信息的函数
print_config_info() {
    local config_num=$1
    local config_name=$2
    local config_desc=$3
    local property=$4
    local seed=$5
    local output_dir=$6
    local dependency_id=$7

    echo ""
    echo -e "${CYAN}=========================================="
    echo "提交训练任务 ${config_num}/3"
    echo -e "==========================================${NC}"
    echo -e "${GREEN}配置名称:${NC}   $config_name"
    echo -e "${GREEN}配置描述:${NC}   $config_desc"
    echo -e "${GREEN}属性:${NC}       $property"
    echo -e "${GREEN}随机种子:${NC}   $seed"
    echo -e "${GREEN}输出目录:${NC}   $output_dir"
    if [ -n "$dependency_id" ]; then
        echo -e "${YELLOW}依赖作业:${NC}   $dependency_id (等待其完成)${NC}"
    else
        echo -e "${GREEN}依赖作业:${NC}   无（立即开始）"
    fi
}

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}=========================================="
echo "SLURM 消融实验批量提交工具"
echo -e "==========================================${NC}"
echo ""
echo -e "${GREEN}实验配置:${NC}"
echo "  属性:           ${PROPERTIES[@]}"
echo "  随机种子:       ${RANDOM_SEEDS[@]}"
echo "  配置数量:       3种"
echo "  执行方式:       串行（依赖链）"
echo "  总任务数:       $((${#PROPERTIES[@]} * ${#RANDOM_SEEDS[@]} * 3))"
echo ""
echo -e "${GREEN}消融实验设计:${NC}"
echo "  1. ${CONFIG_1_DESC}"
echo "  2. ${CONFIG_2_DESC}"
echo "  3. ${CONFIG_3_DESC}"
echo ""
echo -e "${GREEN}资源配置:${NC}"
echo "  GPU:            ${SLURM_GPUS}"
echo "  最大时间:       ${SLURM_TIME}"
echo "  Conda环境:      ${CONDA_ENV}"
echo -e "${BLUE}==========================================${NC}"

# 用于追踪作业链
PREV_JOB_ID=""
ALL_JOB_IDS=()
JOB_CONFIGS=()

# 循环遍历每个属性和种子
for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do

        # ===== 配置1: 基线 =====
        OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${CONFIG_1_SUFFIX}_${PROPERTY}_quantext"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_1_NAME}"

        print_config_info "1" "$CONFIG_1_NAME" "$CONFIG_1_DESC" "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" "CONFIG_1_PARAMS" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_1_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置2: Fine-grained + Projection + Middle =====
        OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${CONFIG_2_SUFFIX}_${PROPERTY}_quantext"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_2_NAME}"

        print_config_info "2" "$CONFIG_2_NAME" "$CONFIG_2_DESC" "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" "CONFIG_2_PARAMS" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_2_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

        # ===== 配置3: Fine-grained + Projection, 无Middle =====
        OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${CONFIG_3_SUFFIX}_${PROPERTY}_quantext"
        JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_3_NAME}"

        print_config_info "3" "$CONFIG_3_NAME" "$CONFIG_3_DESC" "$PROPERTY" "$SEED" "$OUTPUT_DIR" "$PREV_JOB_ID"

        JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" "CONFIG_3_PARAMS" "$PREV_JOB_ID")

        if [ -n "$JOB_ID" ]; then
            echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
            PREV_JOB_ID=$JOB_ID
            ALL_JOB_IDS+=($JOB_ID)
            JOB_CONFIGS+=("${CONFIG_3_DESC}")
        else
            echo -e "${RED}✗ 作业提交失败！${NC}"
            exit 1
        fi
        sleep 1

    done
done

#==============================================================================
# 汇总信息
#==============================================================================

echo ""
echo -e "${GREEN}=========================================="
echo "所有作业提交完成！"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}提交的作业列表（串行执行顺序）:${NC}"
for i in "${!ALL_JOB_IDS[@]}"; do
    echo "  $((i+1)). 作业ID: ${ALL_JOB_IDS[$i]} - ${JOB_CONFIGS[$i]}"
done

echo ""
echo -e "${YELLOW}管理命令:${NC}"
echo "  查看所有作业:          squeue -u \$USER"
echo "  查看详细信息:          squeue -u \$USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"
echo "  查看依赖关系:          squeue -u \$USER -o '%.18i %.30j %.8T %.10r'"
echo "  取消所有作业:          scancel -u \$USER"
echo "  取消作业链:            scancel ${ALL_JOB_IDS[@]}"
echo ""
echo -e "${YELLOW}监控命令:${NC}"
echo "  实时监控:              watch -n 10 'squeue -u \$USER'"
echo "  查看第一个作业日志:    tail -f output_*baseline*/train_*.out"
echo ""
echo -e "${GREEN}========================================${NC}"
