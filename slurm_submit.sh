#!/bin/bash
# SLURM作业提交和管理辅助脚本

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 创建必要的目录
mkdir -p logs

echo -e "${BLUE}=========================================="
echo "SLURM 批量训练作业提交工具"
echo -e "==========================================${NC}"

# 显示作业配置
echo -e "\n${GREEN}作业配置:${NC}"
echo "  属性: bulk_modulus_kv, shear_modulus_gv"
echo "  随机种子: 42, 7, 123"
echo "  总任务数: 6 (2属性 × 3种子)"
echo "  每个任务: 1 GPU, 24 CPU核心, 64GB内存"
echo "  最大运行时间: 48小时"

echo -e "\n${YELLOW}提示: 请确保已修改 submit_slurm_batch.sh 中的以下内容:${NC}"
echo "  1. #SBATCH --partition=gpu  → 改为你的集群分区名"
echo "  2. 取消注释并配置环境激活命令（conda/module）"
echo "  3. 检查GPU资源配置是否符合集群规范"

read -p "$(echo -e ${GREEN}是否继续提交作业? [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}取消提交${NC}"
    exit 0
fi

# 提交作业
echo -e "\n${BLUE}正在提交作业...${NC}"
JOB_ID=$(sbatch submit_slurm_batch.sh | grep -oP '\d+')

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}作业提交失败！${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 作业提交成功！${NC}"
echo -e "  作业ID: ${BLUE}$JOB_ID${NC}"
echo -e "  数组任务: ${BLUE}${JOB_ID}_[0-5]${NC}"

# 显示管理命令
echo -e "\n${GREEN}=========================================="
echo "作业管理命令:"
echo -e "==========================================${NC}"
echo -e "  查看作业状态:     ${BLUE}squeue -j $JOB_ID${NC}"
echo -e "  查看所有任务:     ${BLUE}squeue -u \$USER${NC}"
echo -e "  取消作业:         ${BLUE}scancel $JOB_ID${NC}"
echo -e "  取消单个任务:     ${BLUE}scancel ${JOB_ID}_<0-5>${NC}"
echo -e "  查看作业详情:     ${BLUE}scontrol show job $JOB_ID${NC}"
echo -e "  查看实时日志:     ${BLUE}tail -f logs/slurm_${JOB_ID}_<0-5>.out${NC}"
echo -e "  查看所有日志:     ${BLUE}ls -lh logs/slurm_${JOB_ID}_*.{out,err}${NC}"

echo -e "\n${GREEN}任务ID对应关系:${NC}"
echo "  任务0: bulk_modulus_kv, seed=42"
echo "  任务1: bulk_modulus_kv, seed=7"
echo "  任务2: bulk_modulus_kv, seed=123"
echo "  任务3: shear_modulus_gv, seed=42"
echo "  任务4: shear_modulus_gv, seed=7"
echo "  任务5: shear_modulus_gv, seed=123"

# 等待几秒后显示状态
sleep 2
echo -e "\n${BLUE}当前作业状态:${NC}"
squeue -j $JOB_ID

echo -e "\n${GREEN}提示: 使用以下命令监控所有任务:${NC}"
echo -e "  ${BLUE}watch -n 5 'squeue -u \$USER'${NC}"
