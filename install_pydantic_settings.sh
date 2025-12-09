#!/bin/bash
# 安装 pydantic-settings 包以支持 Pydantic v2

echo "========================================"
echo "安装 pydantic-settings"
echo "========================================"
echo ""

# 检查当前 Pydantic 版本
echo "检查 Pydantic 版本..."
PYDANTIC_VERSION=$(python -c "import pydantic; print(pydantic.__version__)" 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "当前 Pydantic 版本: $PYDANTIC_VERSION"
    echo ""

    # 判断是否为 v2
    MAJOR_VERSION=$(echo $PYDANTIC_VERSION | cut -d. -f1)

    if [ "$MAJOR_VERSION" = "2" ]; then
        echo "检测到 Pydantic v2"
        echo "需要安装 pydantic-settings 包"
        echo ""

        # 检查是否已安装
        python -c "import pydantic_settings" 2>/dev/null
        if [ $? -eq 0 ]; then
            SETTINGS_VERSION=$(python -c "import pydantic_settings; print(pydantic_settings.__version__)" 2>/dev/null)
            echo "✓ pydantic-settings 已安装 (版本: $SETTINGS_VERSION)"
        else
            echo "安装 pydantic-settings..."
            pip install pydantic-settings

            if [ $? -eq 0 ]; then
                echo ""
                echo "✓ pydantic-settings 安装成功"
            else
                echo ""
                echo "✗ 安装失败"
                exit 1
            fi
        fi
    else
        echo "检测到 Pydantic v1"
        echo "不需要 pydantic-settings 包"
    fi
else
    echo "✗ 无法检测 Pydantic 版本"
    echo "请确保已激活正确的 conda 环境"
    exit 1
fi

echo ""
echo "========================================"
echo "验证安装"
echo "========================================"
echo ""

# 测试导入
python -c "
try:
    from utils import BaseSettings
    print('✓ BaseSettings 导入成功')
except Exception as e:
    print(f'✗ 导入失败: {e}')
    exit(1)

try:
    from models.alignn import ALIGNN, ALIGNNConfig
    print('✓ ALIGNN 和 ALIGNNConfig 导入成功')
except Exception as e:
    print(f'✗ 导入失败: {e}')
    exit(1)

print('')
print('所有导入测试通过！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 环境配置完成"
else
    echo ""
    echo "✗ 仍有问题，请检查错误信息"
    exit 1
fi
