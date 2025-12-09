#!/bin/bash
#==============================================================================
# SLURM 消融实验提交脚本（使用自定义sbatch路径）
# 用法: 在脚本开头设置 REAL_SBATCH 为真正的sbatch路径
#==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

#==============================================================================
# 重要：设置真正的 sbatch 路径
#==============================================================================
# 运行 ./find_real_sbatch.sh 找到真正的sbatch路径，然后在这里设置
# 常见路径：
#   /opt/slurm/bin/sbatch
#   /usr/local/bin/sbatch
#   /opt/ohpc/pub/scheduler/slurm/bin/sbatch

REAL_SBATCH=""  # 留空则自动查找

# 如果未设置，尝试自动查找
if [ -z "$REAL_SBATCH" ]; then
    echo "自动查找真正的sbatch..."

    # 尝试常见路径
    for path in /opt/slurm/bin/sbatch /usr/local/bin/sbatch /opt/ohpc/pub/scheduler/slurm/bin/sbatch; do
        if [ -f "$path" ] && [ -x "$path" ]; then
            # 测试是否需要sudo
            TEST_OUTPUT=$($path --version 2>&1)
            if [[ ! "$TEST_OUTPUT" =~ "sudo" ]]; then
                REAL_SBATCH="$path"
                echo "✓ 找到: $REAL_SBATCH"
                break
            fi
        fi
    done

    if [ -z "$REAL_SBATCH" ]; then
        echo -e "${RED}=========================================="
        echo "错误：未找到可用的sbatch"
        echo -e "==========================================${NC}"
        echo ""
        echo "请运行以下命令找到真正的sbatch："
        echo "  ./find_real_sbatch.sh"
        echo ""
        echo "然后在本脚本开头设置："
        echo "  REAL_SBATCH=\"/path/to/real/sbatch\""
        echo ""
        exit 1
    fi
fi

# 验证sbatch可用
echo "验证sbatch: $REAL_SBATCH"
$REAL_SBATCH --version 2>&1 | head -3
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ sbatch不可用或需要sudo${NC}"
    echo "请检查路径: $REAL_SBATCH"
    exit 1
fi
echo ""

#==============================================================================
# 配置
#==============================================================================

WORK_DIR="${HOME}/ablation_experiments"
PROPERTIES=("shear_modulus_gv")
RANDOM_SEEDS=(7)
SLURM_PARTITION=""
SLURM_GPUS=1
SLURM_NODES=1
SLURM_NTASKS=1
CONDA_ENV="sganet"
DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

# 配置定义
declare -A CONFIGS
CONFIGS[1_name]="baseline"
CONFIGS[1_desc]="Baseline: No Fine-grained + With Middle fusion"
CONFIGS[1_suffix]="onlymiddle"
CONFIGS[1_fg]="False"
CONFIGS[1_proj]="False"
CONFIGS[1_middle]="True"
CONFIGS[1_cross]="False"

CONFIGS[2_name]="fg_proj_middle"
CONFIGS[2_desc]="Fine-grained + Projection + Middle fusion"
CONFIGS[2_suffix]="middle_fg_proj"
CONFIGS[2_fg]="True"
CONFIGS[2_proj]="True"
CONFIGS[2_middle]="True"
CONFIGS[2_cross]="False"

CONFIGS[3_name]="fg_proj_crossmodal_nomiddle"
CONFIGS[3_desc]="Fine-grained + Projection + Cross-modal, No Middle fusion"
CONFIGS[3_suffix]="fg_proj_crossmodal_nomiddle"
CONFIGS[3_fg]="True"
CONFIGS[3_proj]="True"
CONFIGS[3_middle]="False"
CONFIGS[3_cross]="True"

CONFIGS[4_name]="fg_proj_crossmodal_middle"
CONFIGS[4_desc]="Fine-grained + Projection + Cross-modal + Middle fusion (Full)"
CONFIGS[4_suffix]="fg_proj_crossmodal_middle"
CONFIGS[4_fg]="True"
CONFIGS[4_proj]="True"
CONFIGS[4_middle]="True"
CONFIGS[4_cross]="True"

#==============================================================================
# 创建SLURM作业文件
#==============================================================================

create_job_file() {
    local job_file=$1
    local job_name=$2
    local output_dir=$3
    local property=$4
    local seed=$5
    local use_fg=$6
    local use_proj=$7
    local use_middle=$8
    local use_cross_modal=$9

    local partition_line=""
    if [ -n "$SLURM_PARTITION" ]; then
        partition_line="#SBATCH -p ${SLURM_PARTITION}"
    fi

    cat > "$job_file" <<JOBFILE
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N ${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --gpus=${SLURM_GPUS}
#SBATCH -o ${output_dir}/%x-%j.out
#SBATCH -e ${output_dir}/%x-%j.err
${partition_line}

# 激活Conda环境
source ~/.bashrc 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ${CONDA_ENV}

# 环境变量
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "SLURM 作业信息"
echo "=========================================="
echo "作业 ID:       \${SLURM_JOB_ID}"
echo "作业名称:      \${SLURM_JOB_NAME}"
echo "开始时间:      \$(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 执行训练
python train_with_cross_modal_attention.py \\
    --root_dir ${DATA_ROOT} \\
    --dataset jarvis \\
    --property ${property} \\
    --train_ratio 0.8 \\
    --val_ratio 0.1 \\
    --test_ratio 0.1 \\
    --batch_size 64 \\
    --epochs 100 \\
    --learning_rate 5e-4 \\
    --weight_decay 1e-3 \\
    --warmup_steps 2000 \\
    --alignn_layers 4 \\
    --gcn_layers 4 \\
    --hidden_features 256 \\
    --graph_dropout 0.15 \\
    --use_cross_modal ${use_cross_modal} \\
    --cross_modal_num_heads 2 \\
    --use_middle_fusion ${use_middle} \\
    --middle_fusion_layers 2 \\
    --use_fine_grained_attention ${use_fg} \\
    --middle_fusion_dropout 0.35 \\
    --fine_grained_hidden_dim 256 \\
    --fine_grained_num_heads 8 \\
    --fine_grained_dropout 0.35 \\
    --fine_grained_use_projection ${use_proj} \\
    --early_stopping_patience 150 \\
    --output_dir ${output_dir} \\
    --num_workers 24 \\
    --random_seed ${seed}

EXIT_CODE=\$?
echo "结束时间: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "退出码: \${EXIT_CODE}"
exit \${EXIT_CODE}
JOBFILE

    chmod +x "$job_file"
}

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}=========================================="
echo "SLURM 消融实验批量提交工具"
echo "（使用真正的sbatch: $REAL_SBATCH）"
echo -e "==========================================${NC}"
echo ""

mkdir -p "$WORK_DIR" || {
    echo -e "${RED}✗ 无法创建目录: $WORK_DIR${NC}"
    exit 1
}

SCRIPT_DIR="$WORK_DIR/job_scripts"
mkdir -p "$SCRIPT_DIR"

echo -e "${GREEN}工作目录:${NC} $WORK_DIR"
echo -e "${GREEN}脚本目录:${NC} $SCRIPT_DIR"
echo -e "${GREEN}sbatch路径:${NC} $REAL_SBATCH"
echo ""

PREV_JOB_ID=""
ALL_JOB_IDS=()
JOB_CONFIGS=()

for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do
        for CONFIG_NUM in 1 2 3 4; do

            CONFIG_NAME="${CONFIGS[${CONFIG_NUM}_name]}"
            CONFIG_DESC="${CONFIGS[${CONFIG_NUM}_desc]}"
            CONFIG_SUFFIX="${CONFIGS[${CONFIG_NUM}_suffix]}"
            CONFIG_FG="${CONFIGS[${CONFIG_NUM}_fg]}"
            CONFIG_PROJ="${CONFIGS[${CONFIG_NUM}_proj]}"
            CONFIG_MIDDLE="${CONFIGS[${CONFIG_NUM}_middle]}"
            CONFIG_CROSS="${CONFIGS[${CONFIG_NUM}_cross]}"

            OUTPUT_DIR="${WORK_DIR}/output_100epochs_${SEED}_bs64_sw_ju_${CONFIG_SUFFIX}_${PROPERTY}_quantext"
            JOB_NAME="train_${PROPERTY}_seed${SEED}_${CONFIG_NAME}"
            JOB_FILE="${SCRIPT_DIR}/${JOB_NAME}.sbatch"

            echo -e "${CYAN}=========================================="
            echo "提交任务 ${CONFIG_NUM}/4: ${CONFIG_NAME}"
            echo -e "==========================================${NC}"
            echo -e "${GREEN}配置:${NC} $CONFIG_DESC"

            mkdir -p "$OUTPUT_DIR"

            echo "创建作业文件: $JOB_FILE"
            create_job_file "$JOB_FILE" "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                           "$CONFIG_FG" "$CONFIG_PROJ" "$CONFIG_MIDDLE" "$CONFIG_CROSS"

            # 使用真正的sbatch提交
            if [ -n "$PREV_JOB_ID" ]; then
                echo "提交作业（依赖: $PREV_JOB_ID）..."
                JOB_SUBMIT=$($REAL_SBATCH --dependency=afterok:${PREV_JOB_ID} "$JOB_FILE" 2>&1)
            else
                echo "提交作业（无依赖）..."
                JOB_SUBMIT=$($REAL_SBATCH "$JOB_FILE" 2>&1)
            fi

            JOB_ID=$(echo "$JOB_SUBMIT" | grep -oP 'Submitted batch job \K\d+')

            if [ -n "$JOB_ID" ]; then
                echo -e "${GREEN}✓ 作业已提交: ID = ${JOB_ID}${NC}"
                PREV_JOB_ID=$JOB_ID
                ALL_JOB_IDS+=($JOB_ID)
                JOB_CONFIGS+=("$CONFIG_DESC")
            else
                echo -e "${RED}✗ 提交失败: $JOB_SUBMIT${NC}"
                exit 1
            fi

            echo ""
            sleep 1
        done
    done
done

echo -e "${GREEN}=========================================="
echo "所有作业提交完成！"
echo -e "==========================================${NC}"
echo ""
echo -e "${BLUE}提交的作业:${NC}"
for i in "${!ALL_JOB_IDS[@]}"; do
    echo "  $((i+1)). ID=${ALL_JOB_IDS[$i]} - ${JOB_CONFIGS[$i]}"
done
echo ""
echo -e "${YELLOW}作业文件: ${SCRIPT_DIR}/${NC}"
echo -e "${YELLOW}管理命令:${NC}"
echo "  查看队列: squeue -u \$USER"
echo "  取消作业: scancel ${ALL_JOB_IDS[@]}"
