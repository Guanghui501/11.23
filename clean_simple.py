#!/usr/bin/env python3
"""
简洁清理工具 - 删除包含特定关键词的句子

删除包含以下关键词的句子:
- Å (埃符号)
- bond lengths / bond length
- shorter
- longer
- tilt angles
"""

import re
import sys
import csv


def clean_description(text):
    """
    删除包含特定关键词的句子

    Parameters:
    -----------
    text : str
        原始描述文本

    Returns:
    --------
    str
        清理后的文本
    """

    if not text or not isinstance(text, str):
        return text

    # 按句号分割成句子
    sentences = text.split('.')

    # 要删除的关键词列表
    keywords = [
        'Å',           # 埃符号
        '?',           # 可能的埃符号编码
        'bond length', # 键长（包含 bond lengths）
        'shorter',     # 更短
        'longer',      # 更长
        'tilt angle',  # 倾斜角（包含 tilt angles）
    ]

    # 保留不包含关键词的句子
    cleaned_sentences = []

    for sentence in sentences:
        # 检查句子是否包含任何关键词
        contains_keyword = False

        for keyword in keywords:
            if keyword in sentence:
                contains_keyword = True
                break

        # 如果不包含关键词，保留这个句子
        if not contains_keyword and sentence.strip():
            cleaned_sentences.append(sentence.strip())

    # 重新组合句子
    result = '. '.join(cleaned_sentences)

    # 确保以句号结尾
    if result and not result.endswith('.'):
        result += '.'

    return result


def process_csv(input_file, output_file, column='Description'):
    """
    处理CSV文件

    Parameters:
    -----------
    input_file : str
        输入CSV文件路径
    output_file : str
        输出CSV文件路径
    column : str
        要处理的列名
    """

    print("=" * 80)
    print(" 简洁清理工具 - 删除包含关键词的句子")
    print("=" * 80)

    # 尝试使用pandas
    try:
        import pandas as pd
        use_pandas = True
        print("✓ 使用 pandas (快速模式)")
    except ImportError:
        use_pandas = False
        print("✓ 使用标准库 (兼容模式)")

    print(f"\n输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"处理列: {column}")

    if use_pandas:
        # 使用pandas处理
        df = pd.read_csv(input_file)

        if column not in df.columns:
            print(f"\n❌ 错误: 列 '{column}' 不存在")
            print(f"可用列: {', '.join(df.columns)}")
            return False

        print(f"\n总行数: {len(df)}")

        # 统计原始长度
        original_lengths = []
        cleaned_lengths = []

        # 处理每一行
        for i, row in df.iterrows():
            original = str(row[column])
            cleaned = clean_description(original)

            df.at[i, column] = cleaned

            original_lengths.append(len(original))
            cleaned_lengths.append(len(cleaned))

        # 显示统计
        avg_original = sum(original_lengths) / len(original_lengths)
        avg_cleaned = sum(cleaned_lengths) / len(cleaned_lengths)
        reduction = (1 - avg_cleaned / avg_original) * 100

        print(f"\n统计信息:")
        print(f"  原始平均长度: {avg_original:.0f} 字符")
        print(f"  清理后平均长度: {avg_cleaned:.0f} 字符")
        print(f"  平均减少: {reduction:.1f}%")

        # 显示前3个示例
        print(f"\n前3个示例:")
        for i in range(min(3, len(df))):
            comp = df.iloc[i].get('Composition', f'行{i}')
            desc = str(df.iloc[i][column])
            print(f"\n{i+1}. {comp}:")
            print(f"   {desc[:120]}{'...' if len(desc) > 120 else ''}")

        # 保存
        df.to_csv(output_file, index=False)

    else:
        # 使用标准库处理
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if column not in fieldnames:
            print(f"\n❌ 错误: 列 '{column}' 不存在")
            print(f"可用列: {', '.join(fieldnames)}")
            return False

        print(f"\n总行数: {len(rows)}")

        # 处理每一行
        original_lengths = []
        cleaned_lengths = []

        for row in rows:
            original = row[column]
            cleaned = clean_description(original)
            row[column] = cleaned

            original_lengths.append(len(original))
            cleaned_lengths.append(len(cleaned))

        # 统计
        avg_original = sum(original_lengths) / len(original_lengths)
        avg_cleaned = sum(cleaned_lengths) / len(cleaned_lengths)
        reduction = (1 - avg_cleaned / avg_original) * 100

        print(f"\n统计信息:")
        print(f"  原始平均长度: {avg_original:.0f} 字符")
        print(f"  清理后平均长度: {avg_cleaned:.0f} 字符")
        print(f"  平均减少: {reduction:.1f}%")

        # 显示示例
        print(f"\n前3个示例:")
        for i in range(min(3, len(rows))):
            comp = rows[i].get('Composition', f'行{i}')
            desc = rows[i][column]
            print(f"\n{i+1}. {comp}:")
            print(f"   {desc[:120]}{'...' if len(desc) > 120 else ''}")

        # 保存
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n✅ 完成!")
    print(f"   输出: {output_file}")
    print(f"   {column} 列已直接替换为清理后的内容")
    print("=" * 80)

    return True


def main():
    """主函数"""

    # 解析命令行参数
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        column = sys.argv[3] if len(sys.argv) > 3 else 'Description'
    else:
        print("\n使用方法:")
        print("  python clean_simple.py <输入文件> <输出文件> [列名]")
        print("\n示例:")
        print("  python clean_simple.py data.csv cleaned.csv")
        print("  python clean_simple.py data.csv cleaned.csv Description")
        print("\n说明:")
        print("  - 删除包含 Å 的句子")
        print("  - 删除包含 bond length/lengths 的句子")
        print("  - 删除包含 shorter/longer 的句子")
        print("  - 删除包含 tilt angles 的句子")
        print("  - 直接替换原始列，不添加新列")
        sys.exit(1)

    # 处理文件
    try:
        import os
        if not os.path.exists(input_file):
            print(f"\n❌ 错误: 找不到文件 {input_file}")
            sys.exit(1)

        success = process_csv(input_file, output_file, column)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
