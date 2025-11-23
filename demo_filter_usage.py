#!/usr/bin/env python
"""
filter_global_information.py 使用示例
展示如何过滤材料描述中的局部信息
"""

import sys

# 方法1: 处理单个描述文本
print("=" * 80)
print("方法 1: 处理单个描述")
print("=" * 80)

from filter_descriptions_simple import remove_local_information, extract_global_summary

# 您的材料描述
original_description = """LiBa4Hf crystallizes in the cubic F-43m space group. The structure consists of four Li clusters inside a Ba4Hf framework. In each Li cluster, Li(1) is bonded in a 12-coordinate geometry to atoms. In the Ba4Hf framework, Ba(1) is bonded in a distorted q6 geometry to six equivalent Ba(1) and three equivalent Hf(1) atoms. There are three shorter (3.60 Å) and three longer (3.66 Å) Ba(1)-Ba(1) bond lengths. All Ba(1)-Hf(1) bond lengths are 4.25 Å. Hf(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms."""

print("\n原始描述:")
print("-" * 80)
print(original_description)
print(f"\n长度: {len(original_description)} 字符")

# 过滤局部信息
filtered = remove_local_information(original_description, mode='aggressive')
print("\n\n过滤后描述 (Aggressive模式):")
print("-" * 80)
print(filtered)
print(f"\n长度: {len(filtered)} 字符 (减少 {100*(1-len(filtered)/len(original_description)):.1f}%)")

# 提取全局摘要
summary = extract_global_summary(original_description)
print("\n\n全局摘要:")
print("-" * 80)
print(summary)
print(f"\n长度: {len(summary)} 字符 (减少 {100*(1-len(summary)/len(original_description)):.1f}%)")


# 方法2: 处理CSV文件 (如果有pandas)
print("\n\n" + "=" * 80)
print("方法 2: 处理CSV文件")
print("=" * 80)

try:
    import pandas as pd
    from filter_global_information import process_descriptions

    print("\n✅ pandas 已安装，可以处理CSV文件")
    print("\n使用方法:")
    print("""
    from filter_global_information import process_descriptions

    # 处理您的数据文件
    df = process_descriptions(
        csv_file='your_data.csv',           # 输入文件
        output_file='your_data_filtered.csv',  # 输出文件
        mode='aggressive',                   # 过滤模式
        include_global_summary=True          # 是否生成全局摘要
    )

    # 查看结果
    print(df[['formula', 'description', 'description_filtered', 'global_summary']].head())
    """)

except ImportError:
    print("\n⚠️  pandas 未安装，无法处理CSV文件")
    print("   但可以使用 filter_descriptions_simple.py 处理单个描述")


# 方法3: 三种过滤模式对比
print("\n\n" + "=" * 80)
print("方法 3: 对比三种过滤模式")
print("=" * 80)

test_desc = "Al(1) is bonded to four equivalent As(1) atoms to form corner-sharing AlAs4 tetrahedra. All Al(1)-As(1) bond lengths are 2.48 Å."

print("\n原始描述:")
print("-" * 80)
print(test_desc)

modes = ['aggressive', 'moderate', 'conservative']
for mode in modes:
    filtered_mode = remove_local_information(test_desc, mode=mode)
    print(f"\n\n{mode.upper()} 模式:")
    print("-" * 80)
    print(filtered_mode)
    print(f"长度: {len(filtered_mode)} 字符")


# 方法4: 批量处理多个描述
print("\n\n" + "=" * 80)
print("方法 4: 批量处理多个描述")
print("=" * 80)

descriptions = [
    "AlAs is Zincblende structured and crystallizes in the cubic F-43m space group. All Al(1)-As(1) bond lengths are 2.48 Å.",
    "NaI is Halite structured and crystallizes in the cubic Fm-3m space group. All Na(1)-I(1) bond lengths are 3.21 Å.",
    "SrB6 is Calcium hexaboride structured. All Sr(1)-B(1) bond lengths are 3.08 Å."
]

print("\n批量处理 3 个描述:\n")

for i, desc in enumerate(descriptions, 1):
    filtered = remove_local_information(desc, mode='aggressive')
    summary = extract_global_summary(desc)

    print(f"{i}. 原始 ({len(desc)} 字符):")
    print(f"   {desc[:100]}...")
    print(f"   过滤后 ({len(filtered)} 字符): {filtered[:100]}...")
    print(f"   摘要 ({len(summary)} 字符): {summary}")
    print()


# 方法5: 在代码中使用
print("\n\n" + "=" * 80)
print("方法 5: 在您的代码中集成")
print("=" * 80)

print("""
# 示例：在数据加载时过滤描述

def load_materials_data(csv_path):
    import pandas as pd
    from filter_descriptions_simple import remove_local_information

    df = pd.read_csv(csv_path)

    # 过滤所有描述
    df['description_filtered'] = df['description'].apply(
        lambda x: remove_local_information(x, mode='aggressive')
    )

    return df

# 示例：在模型训练时使用过滤后的描述
def train_model(structures, descriptions):
    from filter_descriptions_simple import remove_local_information

    # 过滤描述
    filtered_descriptions = [
        remove_local_information(desc, mode='aggressive')
        for desc in descriptions
    ]

    # 使用过滤后的描述训练
    model.train(structures, filtered_descriptions)

# 示例：在注意力分析时使用
def analyze_attention(model, structure, description):
    from filter_descriptions_simple import remove_local_information

    # 过滤描述
    filtered_desc = remove_local_information(description, mode='aggressive')

    # 分析注意力
    output = model(structure, filtered_desc, return_attention=True)
    attention = output['fine_grained_attention_weights']

    return attention
""")


print("\n\n" + "=" * 80)
print("总结")
print("=" * 80)
print("""
快速使用:

1. 处理单个描述:
   from filter_descriptions_simple import remove_local_information
   filtered = remove_local_information(your_description, mode='aggressive')

2. 提取全局摘要:
   from filter_descriptions_simple import extract_global_summary
   summary = extract_global_summary(your_description)

3. 处理CSV文件 (需要pandas):
   from filter_global_information import process_descriptions
   process_descriptions('input.csv', 'output.csv', mode='aggressive')

推荐模式:
- aggressive: 去除所有键长、键角 (推荐用于注意力分析)
- moderate: 保留配位信息，去除具体数值
- conservative: 只隐藏数值，保留句子结构
""")
