#!/bin/bash
# 自动修复 GLIBCXX 问题并运行测试集提取
# 这个脚本会自动设置正确的库路径

set -e

echo "========================================"
echo "自动修复并运行测试集提取"
echo "========================================"
echo ""

# 获取 conda 环境路径
if [ -z "$CONDA_PREFIX" ]; then
    # 尝试常见的 conda 环境位置
    if [ -d "$HOME/.conda/envs/MatMMFuse" ]; then
        CONDA_PREFIX="$HOME/.conda/envs/MatMMFuse"
        echo "使用 conda 环境: $CONDA_PREFIX"
    elif [ -d "/public/home/ghzhang/.conda/envs/MatMMFuse" ]; then
        CONDA_PREFIX="/public/home/ghzhang/.conda/envs/MatMMFuse"
        echo "使用 conda 环境: $CONDA_PREFIX"
    else
        echo "⚠ 警告: 未找到 conda 环境"
        echo "请先激活 conda 环境:"
        echo "conda activate MatMMFuse"
        exit 1
    fi
else
    echo "使用当前 conda 环境: $CONDA_PREFIX"
fi

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "✓ 已设置 LD_LIBRARY_PATH"
echo ""

# 检查 libstdc++ 是否存在
if [ ! -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    echo "✗ 错误: conda 环境中没有 libstdc++.so.6"
    echo ""
    echo "请安装:"
    echo "conda install -c conda-forge libstdcxx-ng"
    exit 1
fi

# 检查是否支持所需版本
if strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep -q "GLIBCXX_3.4.30"; then
    echo "✓ 找到 GLIBCXX_3.4.30"
else
    echo "⚠ 警告: conda 环境的 libstdc++ 可能不包含 GLIBCXX_3.4.30"
    echo "尝试更新:"
    echo "conda install -c conda-forge libstdcxx-ng"
    echo ""
    echo "继续尝试运行..."
fi

echo ""
echo "========================================"
echo "运行测试集提取"
echo "========================================"
echo ""

# 运行提取脚本
python extract_test_set_from_csv.py \
    --predictions_csv /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set

echo ""
echo "✓ 完成"
