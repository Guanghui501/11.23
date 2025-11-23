#!/usr/bin/env python3
"""
最终清理工具 - 删除关键词句子 + 可选删除停用词

功能:
1. 删除包含特定关键词的句子 (Å, bond lengths, shorter, longer, tilt angles)
2. 可选: 删除英文停用词 (the, a, an, in, of, etc.)
"""

import re
import sys


# 内置英文停用词列表
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or', 'not'
}


def load_stopwords():
    """
    加载停用词列表
    尝试从 NLTK 加载，失败则使用内置列表
    """
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except:
        # 使用内置停用词
        return ENGLISH_STOPWORDS


def remove_stopwords_from_text(text, stopwords_set):
    """
    从文本中删除停用词

    Parameters:
    -----------
    text : str
        输入文本
    stopwords_set : set
        停用词集合

    Returns:
    --------
    str
        删除停用词后的文本
    """

    if not text:
        return text

    # 分割成单词
    words = text.split()

    # 保留非停用词
    filtered_words = []
    for word in words:
        # 去除标点符号后检查
        word_clean = word.strip('.,;:!?()')
        if word_clean.lower() not in stopwords_set:
            filtered_words.append(word)

    # 重新组合
    result = ' '.join(filtered_words)

    return result


def clean_description(text, remove_stopwords=False, stopwords_set=None):
    """
    清理描述文本

    Parameters:
    -----------
    text : str
        原始描述
    remove_stopwords : bool
        是否删除停用词
    stopwords_set : set
        停用词集合

    Returns:
    --------
    str
        清理后的文本
    """

    if not text or not isinstance(text, str):
        return text

    # 步骤1: 按句号分割成句子
    sentences = text.split('.')

    # 步骤2: 删除包含特定关键词的句子
    keywords = [
        'Å',
        '?',           # 可能的编码问题
        'bond length',
        'shorter',
        'longer',
        'tilt angle',
    ]

    # 保留不包含关键词的句子
    cleaned_sentences = []

    for sentence in sentences:
        # 检查是否包含关键词
        contains_keyword = any(keyword in sentence for keyword in keywords)

        if not contains_keyword and sentence.strip():
            cleaned_sentences.append(sentence.strip())

    # 步骤3: 重组句子
    result = '. '.join(cleaned_sentences)

    # 步骤4: 删除停用词（如果启用）
    if remove_stopwords and stopwords_set:
        result = remove_stopwords_from_text(result, stopwords_set)

    # 步骤5: 确保以句号结尾
    if result and not result.endswith('.'):
        result += '.'

    # 步骤6: 清理多余空格
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+\.', '.', result)

    return result.strip()


def process_csv(input_file, output_file, column='Description', remove_stopwords=False):
    """
    处理CSV文件

    Parameters:
    -----------
    input_file : str
        输入CSV文件
    output_file : str
        输出CSV文件
    column : str
        要处理的列名
    remove_stopwords : bool
        是否删除停用词
    """

    print("=" * 80)
    print(" 最终清理工具")
    print("=" * 80)

    # 加载停用词（如果需要）
    stopwords_set = None
    if remove_stopwords:
        stopwords_set = load_stopwords()
        print(f"✓ 已加载 {len(stopwords_set)} 个停用词")

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
    print(f"删除停用词: {'是' if remove_stopwords else '否'}")

    if use_pandas:
        # Pandas处理
        df = pd.read_csv(input_file)

        if column not in df.columns:
            print(f"\n❌ 错误: 列 '{column}' 不存在")
            print(f"可用列: {', '.join(df.columns)}")
            return False

        print(f"\n总行数: {len(df)}")

        # 统计
        original_lengths = []
        cleaned_lengths = []

        # 处理每行
        for i, row in df.iterrows():
            original = str(row[column])
            cleaned = clean_description(original, remove_stopwords, stopwords_set)

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

        # 显示示例
        print(f"\n前3个示例:")
        for i in range(min(3, len(df))):
            comp = df.iloc[i].get('Composition', f'行{i}')
            desc = str(df.iloc[i][column])
            print(f"\n{i+1}. {comp}:")
            print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")

        # 保存
        df.to_csv(output_file, index=False)

    else:
        # 标准库处理
        import csv

        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if column not in fieldnames:
            print(f"\n❌ 错误: 列 '{column}' 不存在")
            return False

        print(f"\n总行数: {len(rows)}")

        # 处理
        original_lengths = []
        cleaned_lengths = []

        for row in rows:
            original = row[column]
            cleaned = clean_description(original, remove_stopwords, stopwords_set)
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

        # 保存
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n✅ 完成!")
    print(f"   输出: {output_file}")
    print("=" * 80)

    return True


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(
        description='最终清理工具 - 删除关键词句子 + 可选删除停用词',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基本使用 (只删除关键词句子)
  %(prog)s input.csv output.csv

  # 删除关键词句子 + 停用词
  %(prog)s input.csv output.csv --remove-stopwords

  # 指定列名
  %(prog)s input.csv output.csv -c Description

  # 完整示例
  %(prog)s data.csv cleaned.csv -c Description --remove-stopwords

删除的内容:
  1. 包含这些关键词的句子:
     - Å (埃符号)
     - bond length/lengths
     - shorter, longer
     - tilt angles

  2. 停用词 (如果启用 --remove-stopwords):
     - the, a, an, in, of, to, is, are, etc.
     - 共约 40 个常用英文停用词
        '''
    )

    parser.add_argument('input', help='输入CSV文件')
    parser.add_argument('output', help='输出CSV文件')
    parser.add_argument('-c', '--column', default='Description',
                        help='要处理的列名 (默认: Description)')
    parser.add_argument('--remove-stopwords', action='store_true',
                        help='删除英文停用词 (the, a, in, of, etc.)')

    args = parser.parse_args()

    # 处理
    try:
        import os
        if not os.path.exists(args.input):
            print(f"\n❌ 错误: 找不到文件 {args.input}")
            sys.exit(1)

        success = process_csv(
            args.input,
            args.output,
            args.column,
            args.remove_stopwords
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
