#!/usr/bin/env python3
"""
最终清理工具 - 直接替换原始描述列

特点：
- 不保留中间版本
- 直接覆盖 Description 列
- 输出干净简洁的CSV
"""

import re
import sys


def ultra_clean(description):
    """超强清理 - 15轮清理"""

    if not description or not isinstance(description, str):
        return description

    # 第1-4轮: 去除完整句子
    description = re.sub(r'All [A-Za-z0-9()\–\-]+bond lengths? (?:are|is) [^.]*\.', '', description)
    description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)
    description = re.sub(r'The [^.]*(?:tilt|bond) angles? [^.]*\.', '', description)
    description = re.sub(r'[^.]*bond lengths?[^.]*\.', '', description)

    # 第5-6轮: 去除数值
    description = re.sub(r'\([^)]*\d+\.?\d*\s*[?°ÅÅ][^)]*\)', '', description)
    description = re.sub(r'\d+\.\d+\s*[?°ÅÅ]', '', description)
    description = re.sub(r'\d+\s*[?°ÅÅ]', '', description)

    # 第7-11轮: 去除残留模式
    description = re.sub(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:shorter|longer)\b', '', description)
    description = re.sub(r'\)\s*and\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:shorter|longer)', '', description)
    description = re.sub(r'\)\s*and\s+\d+', '', description)
    description = re.sub(r'(?<=\.)\s*\)[^.]*?(?=[A-Z])', '', description)
    description = re.sub(r'\)\s+(?:and|or)\s+[^.]*?(?=\s+[A-Z]|\.|$)', '', description)

    # 第12-15轮: 格式整理
    description = re.sub(r'\s+', ' ', description)
    description = re.sub(r'\s+\.', '.', description)
    description = re.sub(r'\.+', '.', description)
    description = re.sub(r'\s+,', ',', description)
    description = re.sub(r'\(\s*\)', '', description)
    description = re.sub(r'(?<=\.\s)\)', '', description)
    description = re.sub(r'\)\s*\.', '.', description)
    description = re.sub(r'\.\s+', '. ', description)
    description = re.sub(r'^\s*[,.\s)]+', '', description)
    description = re.sub(r'[,.\s)]+$', '.', description)
    description = re.sub(r'\s+\d{1,2}\.\s*(?=[A-Z])', ' ', description)

    return description.strip()


def process_csv_replace(input_file, output_file, desc_column='Description'):
    """
    处理CSV - 直接替换Description列
    """

    try:
        import pandas as pd
        use_pandas = True
    except ImportError:
        use_pandas = False

    print(f"\n{'='*80}")
    print(" 最终清理工具 - 直接替换原始描述")
    print(f"{'='*80}\n")

    if use_pandas:
        # Pandas处理
        df = pd.read_csv(input_file)

        if desc_column not in df.columns:
            print(f"❌ 错误: 找不到列 '{desc_column}'")
            print(f"可用列: {', '.join(df.columns)}")
            return False

        print(f"处理列: {desc_column}")
        print(f"行数: {len(df)}")

        # 备份原始数据统计
        original_avg = df[desc_column].str.len().mean()

        # 直接替换
        df[desc_column] = df[desc_column].apply(
            lambda x: ultra_clean(str(x)) if pd.notna(x) else x
        )

        # 新数据统计
        cleaned_avg = df[desc_column].str.len().mean()
        reduction = (1 - cleaned_avg / original_avg) * 100

        print(f"\n统计:")
        print(f"  原始平均: {original_avg:.0f} 字符")
        print(f"  清理后: {cleaned_avg:.0f} 字符")
        print(f"  减少: {reduction:.1f}%")

        # 显示示例
        print(f"\n前3个示例:")
        for i in range(min(3, len(df))):
            comp = df.iloc[i].get('Composition', f'行{i}')
            desc = str(df.iloc[i][desc_column])
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

        if desc_column not in fieldnames:
            print(f"❌ 错误: 找不到列 '{desc_column}'")
            return False

        print(f"处理列: {desc_column}")
        print(f"行数: {len(rows)}")

        # 处理每行
        for row in rows:
            row[desc_column] = ultra_clean(row[desc_column])

        # 保存
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n✅ 完成!")
    print(f"   输出文件: {output_file}")
    print(f"   {desc_column} 列已直接替换为清理后的内容")
    print(f"\n{'='*80}\n")

    return True


if __name__ == '__main__':

    # 解析参数
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        desc_column = sys.argv[3] if len(sys.argv) > 3 else 'Description'
    else:
        print("\n使用方法:")
        print("  python clean_final.py input.csv output.csv [列名]")
        print("\n示例:")
        print("  python clean_final.py data.csv cleaned.csv")
        print("  python clean_final.py data.csv cleaned.csv Description")
        sys.exit(1)

    try:
        success = process_csv_replace(input_file, output_file, desc_column)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
