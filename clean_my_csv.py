#!/usr/bin/env python
"""
清理您的CSV文件 - 去除残留的数值片段
适用于已经用旧版过滤器处理过但仍有残留的CSV
"""

from filter_descriptions_improved import remove_local_information_improved


def clean_csv_no_pandas(input_file, output_file):
    """
    清理CSV文件（不需要pandas）
    """
    import csv

    print(f"正在处理: {input_file}")

    # 读取CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        print("⚠️  CSV文件为空")
        return

    # 处理每一行
    cleaned_count = 0
    for i, row in enumerate(rows):
        if 'description_filtered' in row:
            original = row['description_filtered']
            cleaned = remove_local_information_improved(original, mode='aggressive')

            # 添加清理后的列
            row['description_cleaned'] = cleaned

            if original != cleaned:
                cleaned_count += 1

                if i < 3:  # 显示前3个示例
                    print(f"\n示例 {i+1} ({row.get('formula', 'Unknown')}):")
                    print(f"  前: {original[:80]}...")
                    print(f"  后: {cleaned[:80]}...")

    # 添加新列到fieldnames
    if 'description_cleaned' not in fieldnames:
        fieldnames = list(fieldnames) + ['description_cleaned']

    # 写入新CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ 完成!")
    print(f"   处理了 {len(rows)} 行")
    print(f"   清理了 {cleaned_count} 行的残留数值")
    print(f"   输出: {output_file}")


def clean_csv_with_pandas(input_file, output_file):
    """
    清理CSV文件（使用pandas，更快）
    """
    import pandas as pd

    print(f"正在处理: {input_file}")

    # 读取CSV
    df = pd.read_csv(input_file)

    print(f"读取了 {len(df)} 行")

    # 清理description_filtered列
    if 'description_filtered' in df.columns:
        df['description_cleaned'] = df['description_filtered'].apply(
            lambda x: remove_local_information_improved(str(x), mode='aggressive')
            if pd.notna(x) else x
        )

        # 显示前3个示例
        print("\n前3个示例:")
        for i in range(min(3, len(df))):
            formula = df.iloc[i].get('formula', 'Unknown')
            original = str(df.iloc[i]['description_filtered'])[:60]
            cleaned = str(df.iloc[i]['description_cleaned'])[:60]
            print(f"\n{i+1}. {formula}:")
            print(f"   前: {original}...")
            print(f"   后: {cleaned}...")

    else:
        print("⚠️  未找到 'description_filtered' 列")
        return

    # 保存
    df.to_csv(output_file, index=False)

    print(f"\n✅ 完成!")
    print(f"   输出: {output_file}")


if __name__ == '__main__':
    import sys
    import os

    print("=" * 80)
    print(" CSV清理工具 - 去除残留数值")
    print("=" * 80)

    # 检查是否有pandas
    try:
        import pandas as pd
        use_pandas = True
        print("✅ 检测到 pandas，使用快速模式\n")
    except ImportError:
        use_pandas = False
        print("⚠️  未安装 pandas，使用标准模式\n")

    # 处理命令行参数或使用默认值
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # 默认文件名
        input_file = 'desc_mbj_bandgap0_aggressive.csv'
        output_file = 'desc_mbj_bandgap0_cleaned.csv'

        print(f"使用默认文件名:")
        print(f"  输入: {input_file}")
        print(f"  输出: {output_file}")
        print(f"\n或指定文件: python clean_my_csv.py input.csv output.csv\n")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 找不到文件 '{input_file}'")
        print(f"\n请将您的CSV文件命名为 '{input_file}'")
        print(f"或使用: python clean_my_csv.py 您的文件.csv 输出文件.csv")
        sys.exit(1)

    # 清理文件
    try:
        if use_pandas:
            clean_csv_with_pandas(input_file, output_file)
        else:
            clean_csv_no_pandas(input_file, output_file)

        print("\n" + "=" * 80)
        print(" 清理完成！")
        print("=" * 80)
        print(f"\n您现在可以使用清理后的文件: {output_file}")
        print("\n该文件中的 'description_cleaned' 列已去除所有残留数值")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
