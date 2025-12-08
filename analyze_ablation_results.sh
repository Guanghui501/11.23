#!/bin/bash
#==============================================================================
# 消融实验结果分析脚本
# 用途: 分析并对比Fine-grained Attention消融实验的结果
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
declare -A CONFIG_NAMES
declare -A CONFIG_DESCS
CONFIG_NAMES[1]="onlymiddle"
CONFIG_NAMES[2]="middle_fg_proj"
CONFIG_NAMES[3]="fg_proj_nomiddle"

CONFIG_DESCS[1]="Baseline (No FG + Middle)"
CONFIG_DESCS[2]="FG + Proj + Middle"
CONFIG_DESCS[3]="FG + Proj + No Middle"

OUTPUT_FILE="ablation_study_results_$(date +%Y%m%d_%H%M%S).txt"

#==============================================================================
# 函数定义
#==============================================================================

# 提取最佳结果
extract_best_results() {
    local output_dir=$1
    local config_name=$2

    if [ ! -f "$output_dir/training_log.csv" ]; then
        echo "N/A|N/A|N/A|N/A|N/A"
        return
    fi

    # 读取所有数据（跳过标题行）
    local data=$(tail -n +2 "$output_dir/training_log.csv")

    # 找到最佳验证MAE的行
    local best_line=$(echo "$data" | sort -t',' -k4 -n | head -n 1)

    if [ -z "$best_line" ]; then
        echo "N/A|N/A|N/A|N/A|N/A"
        return
    fi

    local epoch=$(echo "$best_line" | cut -d',' -f1)
    local train_loss=$(echo "$best_line" | cut -d',' -f2)
    local train_mae=$(echo "$best_line" | cut -d',' -f3)
    local val_mae=$(echo "$best_line" | cut -d',' -f4)
    local test_mae=$(echo "$best_line" | cut -d',' -f5 2>/dev/null)

    echo "$epoch|$train_mae|$val_mae|$test_mae|$train_loss"
}

# 计算改进百分比
calculate_improvement() {
    local baseline=$1
    local current=$2

    if [ "$baseline" == "N/A" ] || [ "$current" == "N/A" ]; then
        echo "N/A"
        return
    fi

    # 使用bc进行浮点运算
    local improvement=$(echo "scale=2; ($baseline - $current) / $baseline * 100" | bc)
    echo "$improvement"
}

#==============================================================================
# 主程序
#==============================================================================

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗"
echo "║           Fine-grained Attention 消融实验结果分析            ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 输出到文件和屏幕
exec > >(tee "$OUTPUT_FILE")

for PROPERTY in "${PROPERTIES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do

        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${MAGENTA}实验组: $PROPERTY (seed=$SEED)${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # 存储结果
        declare -A RESULTS

        # 收集所有配置的结果
        for config_id in {1..3}; do
            suffix=${CONFIG_NAMES[$config_id]}
            OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_${suffix}_${PROPERTY}_quantext"

            RESULTS[$config_id]=$(extract_best_results "$OUTPUT_DIR" "$suffix")
        done

        # 提取基线结果（配置1）
        IFS='|' read -r baseline_epoch baseline_train_mae baseline_val_mae baseline_test_mae baseline_train_loss <<< "${RESULTS[1]}"

        # 打印详细结果表格
        echo -e "${GREEN}详细结果对比:${NC}"
        echo ""
        printf "%-35s | %-8s | %-12s | %-12s | %-12s | %-12s\n" \
            "配置" "Epoch" "训练MAE" "验证MAE" "测试MAE" "训练Loss"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for config_id in {1..3}; do
            desc=${CONFIG_DESCS[$config_id]}
            IFS='|' read -r epoch train_mae val_mae test_mae train_loss <<< "${RESULTS[$config_id]}"

            printf "%-35s | %-8s | %-12s | %-12s | %-12s | %-12s\n" \
                "$desc" \
                "${epoch:-N/A}" \
                "${train_mae:-N/A}" \
                "${val_mae:-N/A}" \
                "${test_mae:-N/A}" \
                "${train_loss:-N/A}"
        done

        echo ""
        echo ""

        # 打印相对于基线的改进
        echo -e "${GREEN}相对于基线的改进 (负值表示性能下降):${NC}"
        echo ""
        printf "%-35s | %-15s | %-15s | %-15s\n" \
            "配置" "训练MAE改进%" "验证MAE改进%" "测试MAE改进%"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        for config_id in {2..3}; do
            desc=${CONFIG_DESCS[$config_id]}
            IFS='|' read -r epoch train_mae val_mae test_mae train_loss <<< "${RESULTS[$config_id]}"

            train_improve=$(calculate_improvement "$baseline_train_mae" "$train_mae")
            val_improve=$(calculate_improvement "$baseline_val_mae" "$val_mae")
            test_improve=$(calculate_improvement "$baseline_test_mae" "$test_mae")

            printf "%-35s | %-15s | %-15s | %-15s\n" \
                "$desc" \
                "${train_improve:+${train_improve}%}" \
                "${val_improve:+${val_improve}%}" \
                "${test_improve:+${test_improve}%}"
        done

        echo ""
        echo ""

        # 关键发现
        echo -e "${YELLOW}关键发现:${NC}"
        echo ""

        # 分析配置2 (FG + Proj + Middle)
        IFS='|' read -r epoch2 train_mae2 val_mae2 test_mae2 _ <<< "${RESULTS[2]}"
        if [ "$val_mae2" != "N/A" ] && [ "$baseline_val_mae" != "N/A" ]; then
            val_improve2=$(calculate_improvement "$baseline_val_mae" "$val_mae2")
            if (( $(echo "$val_improve2 > 0" | bc -l) )); then
                echo -e "  ${GREEN}✓${NC} 添加Fine-grained Attention + Projection + Middle fusion 改进了 ${val_improve2}%"
            else
                echo -e "  ${RED}✗${NC} 添加Fine-grained Attention + Projection + Middle fusion 降低了 ${val_improve2#-}%"
            fi
        fi

        # 分析配置3 (FG + Proj, No Middle)
        IFS='|' read -r epoch3 train_mae3 val_mae3 test_mae3 _ <<< "${RESULTS[3]}"
        if [ "$val_mae3" != "N/A" ] && [ "$baseline_val_mae" != "N/A" ]; then
            val_improve3=$(calculate_improvement "$baseline_val_mae" "$val_mae3")
            if (( $(echo "$val_improve3 > 0" | bc -l) )); then
                echo -e "  ${GREEN}✓${NC} 使用Fine-grained Attention + Projection (无Middle) 改进了 ${val_improve3}%"
            else
                echo -e "  ${RED}✗${NC} 使用Fine-grained Attention + Projection (无Middle) 降低了 ${val_improve3#-}%"
            fi
        fi

        # 比较配置2和配置3
        if [ "$val_mae2" != "N/A" ] && [ "$val_mae3" != "N/A" ]; then
            diff_2_3=$(calculate_improvement "$val_mae3" "$val_mae2")
            if (( $(echo "$diff_2_3 > 0" | bc -l) )); then
                echo -e "  ${GREEN}✓${NC} Middle fusion带来了额外 ${diff_2_3}% 的改进"
            elif (( $(echo "$diff_2_3 < 0" | bc -l) )); then
                echo -e "  ${RED}✗${NC} 去除Middle fusion反而改进了 ${diff_2_3#-}%"
            else
                echo -e "  ${YELLOW}○${NC} Middle fusion没有明显影响"
            fi
        fi

        echo ""

        # 最佳配置
        echo -e "${GREEN}最佳配置:${NC}"
        best_config=1
        best_val_mae=$baseline_val_mae

        for config_id in {2..3}; do
            IFS='|' read -r _ _ val_mae _ _ <<< "${RESULTS[$config_id]}"
            if [ "$val_mae" != "N/A" ]; then
                if [ "$best_val_mae" == "N/A" ] || (( $(echo "$val_mae < $best_val_mae" | bc -l) )); then
                    best_config=$config_id
                    best_val_mae=$val_mae
                fi
            fi
        done

        echo -e "  ${CYAN}${CONFIG_DESCS[$best_config]}${NC}"
        echo -e "  验证MAE: ${GREEN}$best_val_mae${NC}"

        echo ""
        echo ""

    done
done

#==============================================================================
# 推荐建议
#==============================================================================

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📋 推荐建议${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "基于以上消融实验结果:"
echo ""
echo "1. 如果Fine-grained Attention带来了改进:"
echo "   → 建议在最终模型中使用Fine-grained Attention"
echo "   → 确认是否需要Projection层（对比配置2和配置3）"
echo ""
echo "2. 如果Middle fusion带来了额外改进:"
echo "   → 建议同时使用Fine-grained和Middle fusion"
echo ""
echo "3. 如果性能下降:"
echo "   → 检查位置编码是否正确添加"
echo "   → 考虑调整dropout率或attention heads数量"
echo "   → 验证训练是否收敛"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}结果已保存到: $OUTPUT_FILE${NC}"
