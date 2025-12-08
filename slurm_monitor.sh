#!/bin/bash
# SLURM作业监控脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# 任务配置
PROPERTIES=("bulk_modulus_kv" "shear_modulus_gv")
RANDOM_SEEDS=(42 7 123)

echo -e "${BLUE}=========================================="
echo "SLURM 训练任务监控面板"
echo -e "==========================================${NC}"

# 显示当前用户的所有作业
echo -e "\n${GREEN}当前运行的作业:${NC}"
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "  无运行中的作业"

# 统计作业状态
echo -e "\n${GREEN}作业统计:${NC}"
RUNNING=$(squeue -u $USER -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue -u $USER -t PENDING 2>/dev/null | wc -l)
RUNNING=$((RUNNING > 0 ? RUNNING - 1 : 0))  # 减去标题行
PENDING=$((PENDING > 0 ? PENDING - 1 : 0))

echo -e "  运行中: ${GREEN}$RUNNING${NC}"
echo -e "  等待中: ${YELLOW}$PENDING${NC}"

# 检查输出目录
echo -e "\n${GREEN}=========================================="
echo "训练进度概览:"
echo -e "==========================================${NC}"

TOTAL_TASKS=0
COMPLETED_TASKS=0

for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do
        OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_onlymiddle_${PROPERTY}_quantext"
        TOTAL_TASKS=$((TOTAL_TASKS + 1))

        echo -e "\n${CYAN}[任务 $TOTAL_TASKS] $PROPERTY (seed=$SEED)${NC}"
        echo "  输出目录: $OUTPUT_DIR"

        if [ -d "$OUTPUT_DIR" ]; then
            # 检查是否有最佳模型
            if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
                echo -e "  状态: ${GREEN}✓ 已完成${NC}"
                COMPLETED_TASKS=$((COMPLETED_TASKS + 1))

                # 尝试读取最佳验证MAE
                if [ -f "$OUTPUT_DIR/training_log.csv" ]; then
                    BEST_MAE=$(tail -n 1 "$OUTPUT_DIR/training_log.csv" | cut -d',' -f4 2>/dev/null)
                    if [ ! -z "$BEST_MAE" ]; then
                        echo "  最佳验证MAE: $BEST_MAE"
                    fi
                fi
            elif [ -f "$OUTPUT_DIR/training_log.csv" ]; then
                echo -e "  状态: ${YELLOW}⟳ 训练中${NC}"

                # 显示最新epoch
                LAST_EPOCH=$(tail -n 1 "$OUTPUT_DIR/training_log.csv" | cut -d',' -f1 2>/dev/null)
                LAST_VAL_MAE=$(tail -n 1 "$OUTPUT_DIR/training_log.csv" | cut -d',' -f4 2>/dev/null)

                if [ ! -z "$LAST_EPOCH" ]; then
                    echo "  当前Epoch: $LAST_EPOCH / 100"
                    echo "  当前验证MAE: $LAST_VAL_MAE"
                fi
            else
                echo -e "  状态: ${YELLOW}○ 等待启动${NC}"
            fi
        else
            echo -e "  状态: ${RED}✗ 未开始${NC}"
        fi
    done
done

echo -e "\n${GREEN}=========================================="
echo -e "总体进度: $COMPLETED_TASKS / $TOTAL_TASKS 已完成${NC}"
echo -e "==========================================${NC}"

# 检查最近的日志文件
echo -e "\n${GREEN}最近的日志文件:${NC}"
if [ -d "logs" ]; then
    ls -lht logs/slurm_*.out 2>/dev/null | head -6 | awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'
else
    echo "  logs/ 目录不存在"
fi

# 显示帮助命令
echo -e "\n${BLUE}=========================================="
echo "常用命令:"
echo -e "==========================================${NC}"
echo -e "  实时监控:         ${CYAN}watch -n 5 './slurm_monitor.sh'${NC}"
echo -e "  查看特定日志:     ${CYAN}tail -f logs/slurm_<JOB_ID>_<TASK_ID>.out${NC}"
echo -e "  查看错误日志:     ${CYAN}tail -f logs/slurm_<JOB_ID>_<TASK_ID>.err${NC}"
echo -e "  取消所有作业:     ${CYAN}scancel -u \$USER${NC}"
echo -e "  GPU使用情况:      ${CYAN}squeue -u \$USER -o '%.18i %.9P %.8T %b'${NC}"

# 检查是否有错误日志
echo -e "\n${YELLOW}检查错误日志:${NC}"
if [ -d "logs" ]; then
    ERROR_COUNT=$(find logs -name "slurm_*.err" -type f ! -size 0 2>/dev/null | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo -e "  ${RED}警告: 发现 $ERROR_COUNT 个非空错误日志${NC}"
        echo "  查看错误: ls -lhS logs/slurm_*.err | grep -v ' 0 '"
    else
        echo -e "  ${GREEN}✓ 无错误日志${NC}"
    fi
fi

echo ""
