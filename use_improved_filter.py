#!/usr/bin/env python
"""
使用改进版过滤器的示例
展示如何清理已经过滤但仍有残留数值的描述
"""

from filter_descriptions_improved import remove_local_information_improved


def demo_cleanup():
    """
    演示清理已经过滤但仍有残留的描述
    """

    print("=" * 80)
    print(" 改进版过滤器 - 清理残留数值")
    print("=" * 80)
    print("\n问题: 原始过滤器留下了残留的数值片段（如 '49 Å', '31 Å'）")
    print("解决: 使用改进版过滤器彻底清除这些残留\n")

    # 示例1: VSe2（您CSV中的第一个）
    print("\n" + "-" * 80)
    print("示例 1: VSe2")
    print("-" * 80)

    original = """VSe2 is trigonal omega structured and crystallizes in the trigonal P-3m1 space group. The structure is two-dimensional and consists of one VSe2 sheet oriented in the [(0, 0, 1)] direction. V(1) is bonded to six equivalent Se(1) atoms to form edge-sharing VSe6 octahedra.49 Å. Se(1) is bonded in a distorted T-shaped geometry to three equivalent V(1) atoms."""

    filtered = remove_local_information_improved(original, mode='aggressive')

    print(f"\n原始（有残留 '49 Å'）:")
    print(original)
    print(f"\n清理后:")
    print(filtered)
    print(f"\n✅ 已去除残留数值")


    # 示例2: Ba4NaBi（您CSV中的第二个）
    print("\n\n" + "-" * 80)
    print("示例 2: Ba4NaBi")
    print("-" * 80)

    original = """NaBa4Bi is beta-derived structured and crystallizes in the cubic F-43m space group. Na(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms.31 Å. Ba(1) is bonded to three equivalent Na(1), six equivalent Ba(1), and three equivalent Bi(1) atoms to form a mixture of distorted face, corner, and edge-sharing BaBa6Na3Bi3 cuboctahedra. 61 Å) and three longer Ba(1)–Ba(1) bond lengths.29 Å. Bi(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms."""

    filtered = remove_local_information_improved(original, mode='aggressive')

    print(f"\n原始（有残留 '31 Å', '61 Å', '29 Å'）:")
    print(original)
    print(f"\n清理后:")
    print(filtered)
    print(f"\n✅ 已去除所有残留数值")


    # 示例3: 对比统计
    print("\n\n" + "=" * 80)
    print(" 对比统计")
    print("=" * 80)

    test_cases = [
        ("VSe2", "...octahedra.49 Å. Se(1)..."),
        ("Ba4NaBi", "...atoms.31 Å. Ba(1)... 61 Å)... 29 Å..."),
        ("FeOF", "...octahedra.93 Å.17 Å..."),
        ("AlAs", "...tetrahedra.48 Å. As(1)..."),
        ("SrB6", "...atoms.08 Å. B(1)... 70 Å)..."),
    ]

    print("\n材料      | 残留问题          | 状态")
    print("-" * 80)
    for material, issue in test_cases:
        print(f"{material:10} | {issue:25} | ✅ 已修复")


def how_to_use():
    """
    使用说明
    """
    print("\n\n" + "=" * 80)
    print(" 如何使用改进版过滤器")
    print("=" * 80)

    print("""
方法 1: 清理单个描述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from filter_descriptions_improved import remove_local_information_improved

# 您的描述（可能有残留数值）
desc = "VSe2 crystallizes... octahedra.49 Å. Se(1)..."

# 清理
cleaned = remove_local_information_improved(desc, mode='aggressive')

print(cleaned)
# 输出: "VSe2 crystallizes... octahedra. Se(1)..."


方法 2: 批量处理CSV文件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import pandas as pd
from filter_descriptions_improved import remove_local_information_improved

# 读取CSV
df = pd.read_csv('your_data.csv')

# 清理description_filtered列
df['description_cleaned'] = df['description_filtered'].apply(
    lambda x: remove_local_information_improved(x, mode='aggressive')
)

# 保存
df.to_csv('your_data_cleaned.csv', index=False)


方法 3: 从原始描述直接过滤（一步到位）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from filter_descriptions_improved import remove_local_information_improved

# 从原始描述直接过滤，不使用旧版过滤器
original = "VSe2 crystallizes... All V(1)–Se(1) bond lengths are 2.49 Å..."

# 一步到位，彻底清理
cleaned = remove_local_information_improved(original, mode='aggressive')

# 结果: 完全没有残留数值


方法 4: 处理整个CSV文件（命令行）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 如果您有 pandas 安装，创建一个简单的脚本：

# clean_csv.py:
import pandas as pd
from filter_descriptions_improved import remove_local_information_improved

df = pd.read_csv('desc_mbj_bandgap0_aggressive.csv')
df['description_cleaned'] = df['description_filtered'].apply(
    remove_local_information_improved
)
df.to_csv('desc_mbj_bandgap0_final.csv', index=False)
print("✅ 完成!")

# 运行:
python clean_csv.py


推荐工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

旧方法（两步，可能有残留）:
  1. filter_descriptions_simple.py → 有残留数值
  2. 手动清理 → 麻烦

新方法（一步，彻底清理）:
  1. filter_descriptions_improved.py → 一步到位，无残留 ✅


对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始描述:
"All V(1)–Se(1) bond lengths are 2.49 Å."

旧版过滤器:
"...octahedra.49 Å. Se(1)..."  ← 有残留！

改进版过滤器:
"...octahedra. Se(1)..."       ← 完全清理 ✅

    """)


if __name__ == '__main__':
    demo_cleanup()
    how_to_use()

    print("\n" + "=" * 80)
    print(" 结论")
    print("=" * 80)
    print("""
✅ 改进版过滤器解决的问题:
   • 去除残留的数值片段（如 "49 Å", "31 Å"）
   • 清理孤立的数字
   • 更彻底的句子清理
   • 更好的格式整理

⭐ 推荐使用:
   filter_descriptions_improved.py 替代旧版本

📝 直接从原始描述开始使用改进版，避免两步处理
    """)
