#!/bin/bash
#==============================================================================
# 消融实验监控脚本
# 用途: 监控Fine-grained Attention消融实验的训练进度
#==============================================================================

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

#==============================================================================
# 配置
#==============================================================================

PROPERTIES=("mbj_bandgap")
RANDOM_SEEDS=(42)

# 配置定义
declare -A CONFIGS
CONFIGS[1]="onlymiddle|Baseline (No FG + Middle)"
CONFIGS[2]="middle_fg_proj|FG + Proj + Middle"
CONFIGS[3]="fg_proj_nomiddle|FG + Proj + No Middle"

#==============================================================================
# 函数定义
#==============================================================================

# 获取训练进度信息
get_training_progress() {
    local output_dir=$1
    local config_name=$2

    if [ ! -d "$output_dir" ]; then
        echo -e "${RED}    ✗ 目录不存在${NC}"
        return 1
    fi

    if [ -f "$output_dir/best_model.pth" ]; then
        # 训练已完成
        echo -e "${GREEN}    ✓ 训练完成${NC}"

        if [ -f "$output_dir/training_log.csv" ]; then
            # 读取最佳验证指标
            local best_line=$(tail -n 1 "$output_dir/training_log.csv")
            local best_epoch=$(echo "$best_line" | cut -d',' -f1)
            local train_mae=$(echo "$best_line" | cut -d',' -f3)
            local val_mae=$(echo "$best_line" | cut -d',' -f4)
            local test_mae=$(echo "$best_line" | cut -d',' -f5 2>/dev/null)

            echo "      最佳Epoch:       $best_epoch"
            echo "      训练MAE:         $train_mae"
            echo "      验证MAE:         $val_mae"
            [ -n "$test_mae" ] && echo "      测试MAE:         $test_mae"
        fi

    elif [ -f "$output_dir/training_log.csv" ]; then
        # 训练进行中
        echo -e "${YELLOW}    ⟳ 训练中${NC}"

        local line_count=$(wc -l < "$output_dir/training_log.csv")
        local current_epoch=$((line_count - 1))  # 减去标题行

        if [ $current_epoch -gt 0 ]; then
            local last_line=$(tail -n 1 "$output_dir/training_log.csv")
            local epoch_num=$(echo "$last_line" | cut -d',' -f1)
            local train_mae=$(echo "$last_line" | cut -d',' -f3)
            local val_mae=$(echo "$last_line" | cut -d',' -f4)

            echo "      当前Epoch:       $epoch_num / 100"
            echo "      进度:            $((epoch_num * 100 / 100))%"
            echo "      当前训练MAE:     $train_mae"
            echo "      当前验证MAE:     $val_mae"

            # 查找最佳验证MAE
            local best_val_mae=$(tail -n +2 "$output_dir/training_log.csv" | cut -d',' -f4 | sort -n | head -n 1)
            echo "      最佳验证MAE:     $best_val_mae"
        else
            echo "      等待第一个epoch完成..."
        fi

    else
        # 还未开始或刚开始
        # 检查SLURM日志
        local slurm_logs=$(ls "$output_dir"/train_*.out 2>/dev/null | head -n 1)
        if [ -n "$slurm_logs" ]; then
            echo -e "${YELLOW}    ○ 作业已启动，等待训练开始${NC}"

            # 显示最后几行日志
            local last_lines=$(tail -n 3 "$slurm_logs" 2>/dev/null)
            if [ -n "$last_lines" ]; then
                echo "      最新日志:"
                echo "$last_lines" | sed 's/^/        /'
            fi
        else
            echo -e "${CYAN}    ○ 等待作业启动${NC}"
        fi
    fi
}

# 检查作业状态
check_slurm_status() {
    local job_pattern=$1
    local status=$(squeue -u $USER -o "%.18i %.30j %.8T" 2>/dev/null | grep "$job_pattern" | head -n 1)

    if [ -n "$status" ]; then
        local job_id=$(echo "$status" | awk '{print $1}')
        local job_state=$(echo "$status" | awk '{print $3}')

        case $job_state in
            RUNNING)
                echo -e "${GREEN}运行中${NC} (ID: $job_id)"
                ;;
            PENDING)
                echo -e "${YELLOW}等待中${NC} (ID: $job_id)"
                ;;
            COMPLETING)
                echo -e "${CYAN}完成中${NC} (ID: $job_id)"
                ;;
            *)
                echo -e "${MAGENTA}$job_state${NC} (ID: $job_id)"
                ;;
        esac
    else
        echo -e "${NC}已完成或未提交${NC}"
    fi
}

# 获取最近的错误信息
check_errors() {
    local output_dir=$1

    local error_log=$(ls "$output_dir"/train_*.err 2>/dev/null | head -n 1)
    if [ -f "$error_log" ] && [ -s "$error_log" ]; then
        echo -e "${RED}    ⚠ 发现错误日志:${NC}"
        tail -n 5 "$error_log" | sed 's/^/      /'
        return 1
    fi
    return 0
}

#==============================================================================
# 主程序
#==============================================================================

clear
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗"
echo "║        Fine-grained Attention 消融实验监控面板               ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# SLURM作业概览
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📊 SLURM 作业状态概览${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

RUNNING=$(squeue -u $USER -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue -u $USER -t PENDING 2>/dev/null | wc -l)
RUNNING=$((RUNNING > 0 ? RUNNING - 1 : 0))
PENDING=$((PENDING > 0 ? PENDING - 1 : 0))

echo -e "  运行中作业: ${GREEN}$RUNNING${NC}"
echo -e "  等待中作业: ${YELLOW}$PENDING${NC}"
echo ""

if [ $RUNNING -gt 0 ] || [ $PENDING -gt 0 ]; then
    echo -e "${YELLOW}当前队列:${NC}"
    squeue -u $USER -o "  %.10i %.25j %.8T %.10M %.10L" 2>/dev/null | head -n 10
    echo ""
fi

# 遍历每个属性和种子
for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do

        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${MAGENTA}🔬 实验组: $PROPERTY (seed=$SEED)${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # 遍历3个配置
        for config_id in {1..3}; do
            IFS='|' read -r suffix desc <<< "${CONFIGS[$config_id]}"

            OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${suffix}_${PROPERTY}_quantext"
            JOB_PATTERN="train_${PROPERTY}_seed${SEED}"

            echo -e "${BLUE}  ┌─ 配置 $config_id: ${desc}${NC}"
            echo -e "${BLUE}  ├─ 输出目录:${NC} $OUTPUT_DIR"

            # SLURM状态
            echo -n -e "${BLUE}  ├─ SLURM状态:${NC} "
            check_slurm_status "$JOB_PATTERN"

            # 训练进度
            echo -e "${BLUE}  └─ 训练进度:${NC}"
            get_training_progress "$OUTPUT_DIR" "$desc"

            # 检查错误
            check_errors "$OUTPUT_DIR"

            echo ""
        done

    done
done

#==============================================================================
# 结果对比（如果都完成了）
#==============================================================================

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}📈 结果对比 (仅显示已完成的实验)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

printf "%-40s | %-12s | %-12s | %-12s\n" "配置" "训练MAE" "验证MAE" "测试MAE"
echo "--------------------------------------------------------------------------------"

for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do
        for config_id in {1..3}; do
            IFS='|' read -r suffix desc <<< "${CONFIGS[$config_id]}"
            OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${suffix}_${PROPERTY}_quantext"

            if [ -f "$OUTPUT_DIR/training_log.csv" ]; then
                local last_line=$(tail -n 1 "$OUTPUT_DIR/training_log.csv")
                local train_mae=$(echo "$last_line" | cut -d',' -f3)
                local val_mae=$(echo "$last_line" | cut -d',' -f4)
                local test_mae=$(echo "$last_line" | cut -d',' -f5 2>/dev/null)

                printf "%-40s | %-12s | %-12s | %-12s\n" \
                    "${desc:0:40}" \
                    "${train_mae:-N/A}" \
                    "${val_mae:-N/A}" \
                    "${test_mae:-N/A}"
            fi
        done
    done
done

echo ""

#==============================================================================
# 帮助信息
#==============================================================================

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}💡 常用命令${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  实时监控:           watch -n 10 './monitor_ablation_study.sh'"
echo "  查看详细日志:       tail -f output_*_onlymiddle_*/train_*.out"
echo "  查看所有作业:       squeue -u \$USER"
echo "  取消所有作业:       scancel -u \$USER"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${NC}最后更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
