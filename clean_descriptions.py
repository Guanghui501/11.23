#!/usr/bin/env python3
"""
材料描述清理工具 - 命令行版本

功能：
- 从CSV文件中读取材料描述
- 去除局部信息（键长、键角等数值）
- 保留全局信息（空间群、晶系、配位几何等）
- 输出清理后的CSV文件

使用方法：
    python clean_descriptions.py -i input.csv -o output.csv
    python clean_descriptions.py --input data.csv --output cleaned.csv --mode aggressive
    python clean_descriptions.py -i input.csv -o output.csv --column description --verbose
"""

import argparse
import sys
import os
import re


def remove_local_information(description, mode='aggressive'):
    """
    去除描述中的局部信息

    Parameters:
    -----------
    description : str
        原始材料描述
    mode : str
        过滤模式：'aggressive', 'moderate', 'conservative'

    Returns:
    --------
    str
        过滤后的描述
    """

    if not description or not isinstance(description, str):
        return description

    if mode == 'aggressive':
        # 第1步: 去除完整的键长句子
        description = re.sub(
            r'All [A-Za-z0-9()\–\-]+bond lengths? (?:are|is) [^.]*\.',
            '',
            description
        )

        # 第2步: 去除包含 shorter/longer 的句子
        description = re.sub(
            r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.',
            '',
            description
        )

        # 第3步: 去除键角句子
        description = re.sub(
            r'The [^.]*(?:tilt|bond) angles? [^.]*\.',
            '',
            description
        )

        # 第4步: 去除任何包含 "bond lengths" 的句子片段
        description = re.sub(
            r'[^.]*bond lengths?[^.]*\.',
            '',
            description
        )

        # 第5步: 去除括号中的数值
        description = re.sub(
            r'\([^)]*\d+\.\d+\s*[ÅÅ?°][^)]*\)',
            '',
            description
        )

        # 第6步: 去除所有数值+单位
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', '', description)
        description = re.sub(r'\d+\s*[ÅÅ?°]', '', description)

        # 第7步: 清理孤立数字
        description = re.sub(r'\s+\d{1,3}\s+(?=[A-Z])', ' ', description)

        # 第8步: 格式整理
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'\s+\.', '.', description)
        description = re.sub(r'\.+', '.', description)
        description = re.sub(r'\s+,', ',', description)
        description = re.sub(r'\(\s*\)', '', description)
        description = re.sub(r'\.\s+', '. ', description)
        description = re.sub(r'^\s*[,.\s]+', '', description)

    elif mode == 'moderate':
        description = re.sub(r'[^.]*bond lengths? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'[^.]*(?:tilt |bond )?angles? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'\s+\.', '.', description)

    elif mode == 'conservative':
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', 'X', description)
        description = re.sub(r'\d+\s*[ÅÅ?°]', 'X', description)

    return description.strip()


def process_csv(input_file, output_file, column='description',
                output_column=None, mode='aggressive', verbose=False):
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
    output_column : str
        输出列名（如果为None，则覆盖原列）
    mode : str
        过滤模式
    verbose : bool
        是否显示详细信息
    """

    # 尝试使用pandas（更快）
    try:
        import pandas as pd
        use_pandas = True
        if verbose:
            print("✓ 使用 pandas 处理（快速模式）")
    except ImportError:
        use_pandas = False
        if verbose:
            print("✓ 使用标准库处理（兼容模式）")

    if use_pandas:
        # pandas 处理
        df = pd.read_csv(input_file)

        if column not in df.columns:
            print(f"❌ 错误: 列 '{column}' 不存在")
            print(f"   可用列: {', '.join(df.columns)}")
            return False

        # 确定输出列名
        out_col = output_column if output_column else f"{column}_cleaned"

        # 处理
        if verbose:
            print(f"\n处理中...")
            print(f"  输入列: {column}")
            print(f"  输出列: {out_col}")
            print(f"  模式: {mode}")

        df[out_col] = df[column].apply(
            lambda x: remove_local_information(str(x), mode=mode)
            if pd.notna(x) else x
        )

        # 显示统计
        if verbose:
            original_lengths = df[column].str.len().mean()
            cleaned_lengths = df[out_col].str.len().mean()
            reduction = (1 - cleaned_lengths / original_lengths) * 100

            print(f"\n统计信息:")
            print(f"  处理行数: {len(df)}")
            print(f"  原始平均长度: {original_lengths:.0f} 字符")
            print(f"  清理后平均长度: {cleaned_lengths:.0f} 字符")
            print(f"  平均减少: {reduction:.1f}%")

            # 显示前3个示例
            print(f"\n前3个示例:")
            for i in range(min(3, len(df))):
                if 'formula' in df.columns:
                    print(f"\n  {i+1}. {df.iloc[i]['formula']}:")
                else:
                    print(f"\n  {i+1}.")
                orig = str(df.iloc[i][column])
                clean = str(df.iloc[i][out_col])
                print(f"     原始: {orig[:80]}{'...' if len(orig) > 80 else ''}")
                print(f"     清理: {clean[:80]}{'...' if len(clean) > 80 else ''}")

        # 保存
        df.to_csv(output_file, index=False)

    else:
        # 标准库处理
        import csv

        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if not rows:
            print("❌ 错误: CSV文件为空")
            return False

        if column not in fieldnames:
            print(f"❌ 错误: 列 '{column}' 不存在")
            print(f"   可用列: {', '.join(fieldnames)}")
            return False

        # 确定输出列名
        out_col = output_column if output_column else f"{column}_cleaned"

        if verbose:
            print(f"\n处理中...")
            print(f"  输入列: {column}")
            print(f"  输出列: {out_col}")
            print(f"  模式: {mode}")

        # 处理每一行
        for row in rows:
            original = row[column]
            cleaned = remove_local_information(original, mode=mode)
            row[out_col] = cleaned

        # 添加新列到fieldnames
        if out_col not in fieldnames:
            fieldnames = list(fieldnames) + [out_col]

        # 显示统计
        if verbose:
            original_lengths = sum(len(r[column]) for r in rows) / len(rows)
            cleaned_lengths = sum(len(r[out_col]) for r in rows) / len(rows)
            reduction = (1 - cleaned_lengths / original_lengths) * 100

            print(f"\n统计信息:")
            print(f"  处理行数: {len(rows)}")
            print(f"  原始平均长度: {original_lengths:.0f} 字符")
            print(f"  清理后平均长度: {cleaned_lengths:.0f} 字符")
            print(f"  平均减少: {reduction:.1f}%")

            # 显示前3个示例
            print(f"\n前3个示例:")
            for i in range(min(3, len(rows))):
                if 'formula' in rows[i]:
                    print(f"\n  {i+1}. {rows[i]['formula']}:")
                else:
                    print(f"\n  {i+1}.")
                orig = rows[i][column]
                clean = rows[i][out_col]
                print(f"     原始: {orig[:80]}{'...' if len(orig) > 80 else ''}")
                print(f"     清理: {clean[:80]}{'...' if len(clean) > 80 else ''}")

        # 保存
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return True


def main():
    """主函数"""

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='材料描述清理工具 - 去除局部信息，保留全局特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基本使用
  %(prog)s -i input.csv -o output.csv

  # 指定列名和模式
  %(prog)s -i data.csv -o cleaned.csv -c description -m aggressive

  # 详细输出
  %(prog)s -i input.csv -o output.csv -v

  # 指定输出列名
  %(prog)s -i input.csv -o output.csv --output-column description_filtered

模式说明:
  aggressive   : 去除所有键长、键角（推荐用于注意力分析）
  moderate     : 保留配位信息，去除具体数值
  conservative : 只隐藏数值，保留句子结构
        '''
    )

    # 必需参数
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入CSV文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出CSV文件路径'
    )

    # 可选参数
    parser.add_argument(
        '-c', '--column',
        default='description',
        help='要处理的列名 (默认: description)'
    )

    parser.add_argument(
        '--output-column',
        default=None,
        help='输出列名 (默认: {输入列名}_cleaned)'
    )

    parser.add_argument(
        '-m', '--mode',
        choices=['aggressive', 'moderate', 'conservative'],
        default='aggressive',
        help='过滤模式 (默认: aggressive)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0'
    )

    # 解析参数
    args = parser.parse_args()

    # 显示标题
    if args.verbose:
        print("=" * 80)
        print(" 材料描述清理工具 v2.0")
        print("=" * 80)

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        return 1

    # 检查输出文件
    if os.path.exists(args.output):
        if args.verbose:
            print(f"⚠️  警告: 输出文件已存在，将被覆盖: {args.output}")

    # 处理文件
    if args.verbose:
        print(f"\n输入文件: {args.input}")
        print(f"输出文件: {args.output}")

    try:
        success = process_csv(
            input_file=args.input,
            output_file=args.output,
            column=args.column,
            output_column=args.output_column,
            mode=args.mode,
            verbose=args.verbose
        )

        if success:
            print(f"\n✅ 成功! 清理后的文件已保存到: {args.output}")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
