#!/bin/bash
# 修复 GLIBCXX 版本问题
# 这个脚本会使用 conda 环境中的 libstdc++ 替代系统版本

set -e

echo "========================================"
echo "检查 GLIBCXX 版本问题"
echo "========================================"
echo ""

# 检查系统 libstdc++
echo "系统 libstdc++ 位置:"
find /lib64 -name "libstdc++.so.6" 2>/dev/null || echo "未找到"
echo ""

# 检查系统支持的 GLIBCXX 版本
echo "系统 libstdc++ 支持的 GLIBCXX 版本:"
strings /lib64/libstdc++.so.6 2>/dev/null | grep GLIBCXX | tail -10 || echo "无法检查"
echo ""

# 检查 conda 环境中的 libstdc++
echo "Conda 环境 libstdc++ 位置:"
CONDA_PREFIX=${CONDA_PREFIX:-$HOME/.conda/envs/MatMMFuse}
find $CONDA_PREFIX/lib -name "libstdc++.so.6" 2>/dev/null || echo "未找到"
echo ""

# 检查 conda 环境支持的版本
CONDA_LIBSTDCXX=$(find $CONDA_PREFIX/lib -name "libstdc++.so.6" 2>/dev/null | head -1)
if [ -n "$CONDA_LIBSTDCXX" ]; then
    echo "Conda 环境 libstdc++ 支持的 GLIBCXX 版本:"
    strings $CONDA_LIBSTDCXX | grep GLIBCXX | tail -10
    echo ""

    # 检查是否包含需要的版本
    if strings $CONDA_LIBSTDCXX | grep -q "GLIBCXX_3.4.30"; then
        echo "✓ Conda 环境包含 GLIBCXX_3.4.30"
        echo ""
        echo "========================================"
        echo "解决方案: 使用 conda 环境的 libstdc++"
        echo "========================================"
        echo ""
        echo "方法1: 设置 LD_LIBRARY_PATH（推荐）"
        echo ""
        echo "在运行脚本前执行:"
        echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
        echo ""
        echo "或者直接运行:"
        echo "LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH python extract_test_set_from_csv.py ..."
        echo ""
    else
        echo "✗ Conda 环境也不包含 GLIBCXX_3.4.30"
        echo ""
        echo "需要安装更新的 libstdc++"
        echo "conda install -c conda-forge libstdcxx-ng"
    fi
else
    echo "✗ 未找到 conda 环境中的 libstdc++"
    echo ""
    echo "安装 libstdc++:"
    echo "conda install -c conda-forge libstdcxx-ng"
fi

echo ""
echo "========================================"
echo "快速修复命令"
echo "========================================"
echo ""
echo "# 方法1: 临时设置环境变量（推荐）"
echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo ""
echo "# 方法2: 安装更新的 libstdc++"
echo "conda install -c conda-forge libstdcxx-ng"
echo ""
echo "# 方法3: 使用系统 Python（如果 conda 环境有问题）"
echo "module load python/3.10  # 或其他可用版本"
echo ""
