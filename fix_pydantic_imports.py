#!/usr/bin/env python
"""
修复 Pydantic v2 兼容性问题

在 Pydantic v2 中，Literal 应该从 typing 模块导入，而不是 pydantic.typing
这个脚本会自动修复所有相关文件。
"""

import os
import sys
import re
from pathlib import Path

def fix_pydantic_imports(file_path):
    """修复单个文件的 pydantic imports"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修复1: from pydantic.typing import Literal
    # 改为: from typing import Literal
    content = re.sub(
        r'from pydantic\.typing import Literal',
        'from typing import Literal',
        content
    )

    # 修复2: from pydantic.typing import Literal, Optional
    # 改为: from typing import Literal, Optional
    content = re.sub(
        r'from pydantic\.typing import (.*)',
        r'from typing import \1',
        content
    )

    # 修复3: 如果已经有 from typing import ...，合并它们
    # 这个稍微复杂，先简单处理

    if content != original_content:
        # 创建备份
        backup_path = str(file_path) + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return True

    return False

def find_and_fix_files(root_dir='.', patterns=['**/*.py']):
    """查找并修复所有相关文件"""
    fixed_files = []

    for pattern in patterns:
        for file_path in Path(root_dir).glob(pattern):
            if file_path.is_file():
                try:
                    if fix_pydantic_imports(file_path):
                        fixed_files.append(file_path)
                except Exception as e:
                    print(f"✗ 错误处理 {file_path}: {e}")

    return fixed_files

if __name__ == "__main__":
    print("="*80)
    print("修复 Pydantic v2 导入兼容性")
    print("="*80)
    print()

    # 查找需要修复的文件
    print("搜索包含 'from pydantic.typing import' 的文件...")
    print()

    # 搜索当前目录和常见子目录
    search_dirs = ['.', 'models', 'crysmmnet-main/src/models']
    all_fixed = []

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"检查目录: {search_dir}")
            fixed = find_and_fix_files(search_dir)
            all_fixed.extend(fixed)

    print()
    print("="*80)
    print("修复结果")
    print("="*80)
    print()

    if all_fixed:
        print(f"✓ 成功修复 {len(all_fixed)} 个文件:")
        for f in all_fixed:
            print(f"  • {f}")
            print(f"    备份: {f}.bak")
        print()
        print("修改内容:")
        print("  from pydantic.typing import Literal  →  from typing import Literal")
    else:
        print("未找到需要修复的文件")

    print()
    print("="*80)
    print("验证修复")
    print("="*80)
    print()

    # 验证是否还有问题
    print("测试导入...")
    try:
        # 尝试导入修复后的模块
        sys.path.insert(0, os.getcwd())
        from models.alignn import ALIGNN, ALIGNNConfig
        print("✓ 成功导入 ALIGNN 和 ALIGNNConfig")
    except ImportError as e:
        print(f"⚠ 仍有导入问题: {e}")
        print()
        print("可能需要:")
        print("1. 降级 pydantic: pip install 'pydantic<2.0'")
        print("2. 或升级代码以完全兼容 pydantic v2")
    except Exception as e:
        print(f"⚠ 其他错误: {e}")

    print()
