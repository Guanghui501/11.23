#!/bin/bash
# 查找真正的SLURM sbatch命令

echo "=========================================="
echo "查找真正的 SLURM sbatch"
echo "=========================================="
echo ""

# 检查wrapper内容
echo "检查 /usr/bin/sbatch 内容："
echo "----------------------------------------"
file /usr/bin/sbatch
echo ""
echo "前20行内容："
head -20 /usr/bin/sbatch 2>/dev/null || echo "无法读取（可能需要权限）"
echo ""

# 查找所有sbatch文件
echo "查找系统中所有的 sbatch："
echo "----------------------------------------"
SBATCH_FILES=$(find /usr /opt /home /public -name sbatch -type f 2>/dev/null | grep -v ".sh")

if [ -z "$SBATCH_FILES" ]; then
    echo "未找到其他sbatch文件"
else
    for f in $SBATCH_FILES; do
        echo "找到: $f"
        ls -lh "$f"
        file "$f"
        echo ""
    done
fi

# 查找slurm相关目录
echo "查找 SLURM 安装目录："
echo "----------------------------------------"
SLURM_DIRS=$(find /usr /opt /home /public -type d -name "*slurm*" 2>/dev/null | head -10)

if [ -z "$SLURM_DIRS" ]; then
    echo "未找到slurm目录"
else
    for d in $SLURM_DIRS; do
        echo "目录: $d"
        if [ -f "$d/bin/sbatch" ]; then
            echo "  ✓ 包含 sbatch: $d/bin/sbatch"
            ls -lh "$d/bin/sbatch"
        fi
    done
fi

echo ""
echo "=========================================="
echo "尝试使用可能的sbatch路径"
echo "=========================================="
echo ""

# 尝试常见路径
POSSIBLE_PATHS=(
    "/opt/slurm/bin/sbatch"
    "/usr/local/bin/sbatch"
    "/usr/slurm/bin/sbatch"
    "/opt/ohpc/pub/scheduler/slurm/bin/sbatch"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "测试: $path"
        file "$path"

        # 创建测试作业
        TEST_JOB="/tmp/test_real_sbatch_$$.sh"
        cat > "$TEST_JOB" <<'EOF'
#!/bin/bash
#SBATCH -J test_real
#SBATCH -N 1
echo "Test with real sbatch"
EOF

        echo "尝试提交测试作业..."
        $path "$TEST_JOB" 2>&1 | head -5
        rm -f "$TEST_JOB"
        echo ""
    fi
done

echo ""
echo "=========================================="
echo "建议"
echo "=========================================="
echo ""
echo "1. 查看 /usr/bin/sbatch 的内容找到真正的sbatch"
echo "   cat /usr/bin/sbatch"
echo ""
echo "2. 如果找到真正的sbatch路径，在脚本中使用："
echo "   REAL_SBATCH=/opt/slurm/bin/sbatch"
echo "   \$REAL_SBATCH your_job.sh"
echo ""
echo "3. 或者联系管理员配置sudo规则，允许无密码运行sbatch"
