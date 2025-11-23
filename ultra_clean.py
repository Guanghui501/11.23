#!/usr/bin/env python3
"""
超强版描述清理工具 - 彻底清除所有残留

针对问题:
- Ba4NaBi: ") and three longer" 残留
- SrB6: ") and four longer (1." 残留
"""

import re


def remove_local_information_ultra(description):
    """
    超强版清理 - 多轮迭代直到完全干净
    """

    if not description or not isinstance(description, str):
        return description

    # 第1轮: 去除完整句子
    description = re.sub(
        r'All [A-Za-z0-9()\–\-]+bond lengths? (?:are|is) [^.]*\.',
        '',
        description
    )

    # 第2轮: 去除 "There are/is ... shorter/longer" 句子
    description = re.sub(
        r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.',
        '',
        description
    )

    # 第3轮: 去除角度句子
    description = re.sub(
        r'The [^.]*(?:tilt|bond) angles? [^.]*\.',
        '',
        description
    )

    # 第4轮: 去除任何包含 "bond lengths" 的片段
    description = re.sub(
        r'[^.]*bond lengths?[^.]*\.',
        '',
        description
    )

    # 第5轮: 去除括号及其内容（包含数字+单位）
    description = re.sub(
        r'\([^)]*\d+\.?\d*\s*[?°ÅÅ][^)]*\)',
        '',
        description
    )

    # 第6轮: 去除所有数字+单位组合
    description = re.sub(r'\d+\.\d+\s*[?°ÅÅ]', '', description)
    description = re.sub(r'\d+\s*[?°ÅÅ]', '', description)

    # 第7轮: 去除残留的 "X shorter" 或 "X longer" 片段
    description = re.sub(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:shorter|longer)\b', '', description)

    # 第8轮: 去除孤立的 ") and X longer/shorter" 模式
    description = re.sub(r'\)\s*and\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:shorter|longer)', '', description)

    # 第9轮: 去除 ") and" 后面跟数字的模式
    description = re.sub(r'\)\s*and\s+\d+', '', description)

    # 第10轮: 去除任何以 ")" 开头的孤立片段
    description = re.sub(r'(?<=\.)\s*\)[^.]*?(?=[A-Z])', '', description)

    # 第11轮: 去除以 ")" 开头到下一个句子的片段
    description = re.sub(r'\)\s+(?:and|or)\s+[^.]*?(?=\s+[A-Z]|\.|$)', '', description)

    # 第12轮: 清理多余的空格和标点
    description = re.sub(r'\s+', ' ', description)
    description = re.sub(r'\s+\.', '.', description)
    description = re.sub(r'\.+', '.', description)
    description = re.sub(r'\s+,', ',', description)
    description = re.sub(r'\(\s*\)', '', description)

    # 第13轮: 去除句子开头或中间孤立的 ")"
    description = re.sub(r'(?<=\.\s)\)', '', description)
    description = re.sub(r'\)\s*\.', '.', description)

    # 第14轮: 清理句子之间的格式
    description = re.sub(r'\.\s+', '. ', description)
    description = re.sub(r'^\s*[,.\s)]+', '', description)
    description = re.sub(r'[,.\s)]+$', '.', description)

    # 第15轮: 去除任何残留的数字片段
    description = re.sub(r'\s+\d{1,2}\.\s*(?=[A-Z])', ' ', description)

    return description.strip()


def process_csv_ultra(input_file, output_file, column='Description'):
    """
    处理CSV文件 - 超强清理
    """
    try:
        import pandas as pd
        use_pandas = True
        print("✓ 使用 pandas")
    except ImportError:
        use_pandas = False
        print("✓ 使用标准库")

    if use_pandas:
        df = pd.read_csv(input_file)

        if column not in df.columns:
            print(f"❌ 列 '{column}' 不存在")
            print(f"可用列: {', '.join(df.columns)}")
            return False

        # 超强清理
        df['Description_ultra_cleaned'] = df[column].apply(
            lambda x: remove_local_information_ultra(str(x)) if pd.notna(x) else x
        )

        # 显示修复的案例
        print("\n修复的案例:")
        problematic_indices = []

        for i, row in df.iterrows():
            original = str(row[column])
            cleaned = str(row['Description_ultra_cleaned'])

            # 检查是否有问题片段
            if re.search(r'\)\s*and\s+\w+\s+(?:longer|shorter)', original):
                problematic_indices.append(i)
                print(f"\n{i}. {row.get('Composition', 'Unknown')}:")
                print(f"   前: ...{original[200:280]}...")
                print(f"   后: ...{cleaned[200:280] if len(cleaned) > 200 else cleaned[-80:]}...")

        # 保存
        df.to_csv(output_file, index=False)

        print(f"\n✅ 处理完成!")
        print(f"   处理行数: {len(df)}")
        print(f"   修复问题: {len(problematic_indices)} 处")
        print(f"   输出文件: {output_file}")

        return True

    else:
        import csv

        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames) + ['Description_ultra_cleaned']

        for row in rows:
            row['Description_ultra_cleaned'] = remove_local_information_ultra(row[column])

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✅ 处理完成! 输出: {output_file}")
        return True


if __name__ == '__main__':
    import sys

    print("=" * 80)
    print(" 超强版清理工具 - 彻底去除残留")
    print("=" * 80)

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("\n使用方法:")
        print("  python ultra_clean.py input.csv output.csv")
        print("\n或使用默认值:")
        input_file = input("输入文件 (默认: desc_cleaned.csv): ").strip() or "desc_cleaned.csv"
        output_file = input("输出文件 (默认: desc_ultra_cleaned.csv): ").strip() or "desc_ultra_cleaned.csv"

    print(f"\n输入: {input_file}")
    print(f"输出: {output_file}\n")

    try:
        process_csv_ultra(input_file, output_file)
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
