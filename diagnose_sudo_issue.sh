#!/bin/bash
# SLURM sudo问题诊断脚本

echo "=========================================="
echo "SLURM sudo问题诊断"
echo "=========================================="
echo ""

# 测试1: 检查sbatch命令
echo "测试1: 检查sbatch命令"
echo "----------------------------------------"
which sbatch
ls -l $(which sbatch)
echo ""

# 测试2: 检查sbatch是否是别名
echo "测试2: 检查sbatch别名"
echo "----------------------------------------"
alias sbatch 2>/dev/null || echo "sbatch 不是别名"
type sbatch
echo ""

# 测试3: 测试简单的sbatch（无heredoc）
echo "测试3: 提交最简单的测试作业"
echo "----------------------------------------"
echo "#!/bin/bash
echo 'Hello World'
" > /tmp/test_sbatch_$$.sh

echo "运行: sbatch /tmp/test_sbatch_$$.sh"
sbatch /tmp/test_sbatch_$$.sh 2>&1 | head -5
RESULT=$?
echo "返回码: $RESULT"
rm -f /tmp/test_sbatch_$$.sh
echo ""

# 测试4: 测试heredoc方式的sbatch
echo "测试4: 测试heredoc方式提交"
echo "----------------------------------------"
echo "运行: sbatch <<EOF ... EOF"
TEST_OUTPUT=$(sbatch <<'EOF' 2>&1
#!/bin/bash
#SBATCH -J test_heredoc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o /tmp/test-%j.out

echo "Test heredoc"
sleep 5
EOF
)
echo "$TEST_OUTPUT"
echo ""

# 测试5: 检查目录权限
echo "测试5: 检查目录创建权限"
echo "----------------------------------------"
TEST_DIR="/public/home/ghzhang/ablation_experiments/test_$$"
echo "尝试创建: $TEST_DIR"
mkdir -p "$TEST_DIR" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 成功创建目录"
    rmdir "$TEST_DIR"
else
    echo "✗ 无法创建目录"
fi
echo ""

# 测试6: 检查conda
echo "测试6: 检查conda环境"
echo "----------------------------------------"
source ~/.bashrc 2>&1 | head -3
conda activate sganet 2>&1 | head -3
echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 测试7: 检查SLURM配置
echo "测试7: SLURM配置"
echo "----------------------------------------"
echo "SLURM版本:"
sbatch --version 2>&1 | head -1 || echo "无法获取版本"
echo ""
echo "SLURM分区:"
sinfo 2>&1 | head -5 || echo "无法获取分区信息"
echo ""

# 测试8: 检查用户权限
echo "测试8: 用户权限"
echo "----------------------------------------"
echo "当前用户: $(whoami)"
echo "用户组: $(groups)"
echo "Home目录权限:"
ls -ld $HOME
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "如果上述某个测试要求sudo密码，那就是问题所在"
