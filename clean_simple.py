#!/usr/bin/env python3
"""
简洁清理工具 - 删除包含特定关键词的句子和停用词

删除包含以下关键词的句子:
- Å (埃符号)
- bond lengths / bond length
- shorter
- longer
- tilt angles

可选删除英文停用词（从 stopwords/en/ 文件夹）
"""

import re
import sys
import csv
import os


def load_stopwords(source='nltk'):
    """
    从 stopwords/en/ 文件夹加载停用词

    Parameters:
    -----------
    source : str
        停用词来源，可选值:
        - 'nltk' (默认)
        - 'snowball'
        - 'spacy'
        - 'all' - 加载所有停用词文件
        - 或其他 stopwords/en/ 中的文件名（不含.txt）

    Returns:
    --------
    set
        停用词集合
    """
    # 构建文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_dir = os.path.join(script_dir, 'stopwords', 'en')

    # 如果选择 'all'，加载所有停用词文件
    if source == 'all':
        print("  正在加载所有停用词文件...")
        all_stopwords = set()
        file_count = 0

        if not os.path.exists(stopwords_dir):
            print(f"⚠️  警告: 找不到停用词目录 {stopwords_dir}")
            return set()

        for filename in os.listdir(stopwords_dir):
            if filename.endswith('.txt') and filename != '_none.txt':
                filepath = os.path.join(stopwords_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word:
                                all_stopwords.add(word)
                    file_count += 1
                except Exception as e:
                    print(f"  ⚠️  跳过文件 {filename}: {e}")

        print(f"  ✓ 已从 {file_count} 个文件加载 {len(all_stopwords)} 个唯一停用词")
        return all_stopwords

    # 映射常用名称到实际文件名
    source_mapping = {
        'nltk': 'nltk.txt',
        'snowball': 'snowball_original.txt',
        'spacy': 'spacy.txt',
        'sklearn': 'scikitlearn.txt',
        'smart': 'smart.txt',
    }

    # 获取文件名
    if source in source_mapping:
        filename = source_mapping[source]
    elif source.endswith('.txt'):
        filename = source
    else:
        filename = f"{source}.txt"

    filepath = os.path.join(stopwords_dir, filename)

    # 如果文件不存在，尝试直接使用文件名
    if not os.path.exists(filepath):
        print(f"⚠️  警告: 找不到停用词文件 {filepath}")
        print(f"   使用默认停用词列表")
        # 返回一个基本的停用词集合
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by',
            'for', 'from', 'has', 'had', 'have', 'he', 'her', 'his', 'in',
            'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was',
            'were', 'will', 'with', 'this', 'these', 'those'
        }

    # 读取停用词文件
    stopwords = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:  # 跳过空行
                stopwords.add(word)

    return stopwords


def remove_stopwords(text, stopwords=None):
    """
    从文本中删除停用词

    Parameters:
    -----------
    text : str
        原始文本
    stopwords : set, optional
        停用词集合，默认使用 ENGLISH_STOPWORDS

    Returns:
    --------
    str
        删除停用词后的文本
    """
    if not text or not isinstance(text, str):
        return text

    if stopwords is None:
        stopwords = ENGLISH_STOPWORDS

    # 分词（简单按空格分割）
    words = text.split()

    # 过滤停用词，保留标点符号
    cleaned_words = []
    for word in words:
        # 提取纯单词（去除标点）
        word_lower = re.sub(r'[^\w\s]', '', word).lower()

        # 如果不是停用词或者是空字符串（纯标点），保留原词
        if word_lower not in stopwords or not word_lower:
            cleaned_words.append(word)

    return ' '.join(cleaned_words)


def clean_description(text, stopwords_set=None):
    """
    删除包含特定关键词的句子

    Parameters:
    -----------
    text : str
        原始描述文本
    stopwords_set : set, optional
        停用词集合，如果提供则删除停用词

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
            cleaned_sentence = sentence.strip()

            # 如果提供了停用词集合，删除停用词
            if stopwords_set:
                cleaned_sentence = remove_stopwords(cleaned_sentence, stopwords_set)

            cleaned_sentences.append(cleaned_sentence)

    # 重新组合句子
    result = '. '.join(cleaned_sentences)

    # 确保以句号结尾
    if result and not result.endswith('.'):
        result += '.'

    return result


def process_csv(input_file, output_file, column='Description', stopwords_source=None):
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
    stopwords_source : str, optional
        停用词来源（如 'nltk', 'snowball', 'spacy'）
        如果为 None，不删除停用词
    """

    print("=" * 80)
    print(" 简洁清理工具 - 删除包含关键词的句子")
    if stopwords_source:
        print(f" [启用停用词删除 - 来源: {stopwords_source}]")
    print("=" * 80)

    # 如果启用停用词删除，加载停用词
    stopwords_set = None
    if stopwords_source:
        print(f"加载停用词: {stopwords_source}...")
        stopwords_set = load_stopwords(stopwords_source)
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
    print(f"删除停用词: {'是 (' + stopwords_source + ')' if stopwords_source else '否'}")

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
            cleaned = clean_description(original, stopwords_set)

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
            cleaned = clean_description(original, stopwords_set)
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

    # 检查停用词相关参数
    stopwords_source = None
    args = sys.argv[1:]

    # 解析停用词参数
    for i, arg in enumerate(args):
        if arg.startswith('--stopwords='):
            stopwords_source = arg.split('=', 1)[1]
            args.pop(i)
            break
        elif arg == '--stopwords' or arg == '--remove-stopwords':
            # 默认使用 nltk
            stopwords_source = 'nltk'
            args.pop(i)
            # 检查下一个参数是否是停用词来源
            if i < len(args) and not args[i].endswith('.csv'):
                stopwords_source = args.pop(i)
            break

    # 解析命令行参数
    if len(args) >= 2:
        input_file = args[0]
        output_file = args[1]
        column = args[2] if len(args) > 2 else 'Description'
    else:
        print("\n使用方法:")
        print("  python clean_simple.py <输入文件> <输出文件> [列名] [选项]")
        print("\n选项:")
        print("  --stopwords[=来源]     删除英文停用词")
        print("  --remove-stopwords     删除英文停用词（默认 nltk）")
        print("\n停用词来源:")
        print("  all       加载所有停用词文件（最彻底）⭐")
        print("  nltk      NLTK 停用词列表（默认）")
        print("  snowball  Snowball 停用词列表")
        print("  spacy     spaCy 停用词列表")
        print("  sklearn   scikit-learn 停用词列表")
        print("  smart     SMART 停用词列表")
        print("  或任何 stopwords/en/ 文件夹中的文件名")
        print("\n示例:")
        print("  python clean_simple.py data.csv cleaned.csv")
        print("  python clean_simple.py data.csv cleaned.csv Description")
        print("  python clean_simple.py data.csv cleaned.csv --stopwords")
        print("  python clean_simple.py data.csv cleaned.csv --stopwords=all")
        print("  python clean_simple.py data.csv cleaned.csv --stopwords=nltk")
        print("  python clean_simple.py data.csv cleaned.csv --stopwords=snowball")
        print("  python clean_simple.py data.csv cleaned.csv Description --stopwords=spacy")
        print("\n说明:")
        print("  - 删除包含 Å 的句子")
        print("  - 删除包含 bond length/lengths 的句子")
        print("  - 删除包含 shorter/longer 的句子")
        print("  - 删除包含 tilt angles 的句子")
        print("  - 可选：删除英文停用词（从 stopwords/en/ 文件夹）")
        print("  - 直接替换原始列，不添加新列")
        sys.exit(1)

    # 处理文件
    try:
        if not os.path.exists(input_file):
            print(f"\n❌ 错误: 找不到文件 {input_file}")
            sys.exit(1)

        success = process_csv(input_file, output_file, column, stopwords_source)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
