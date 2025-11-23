"""
改进版过滤脚本 - 更彻底地去除局部信息
"""

import re


def remove_local_information_improved(description, mode='aggressive'):
    """
    改进版过滤函数，更彻底地去除键长、键角等局部信息

    Parameters:
    -----------
    description : str
        原始材料描述
    mode : str
        'aggressive': 去除所有键长、键角、具体数值

    Returns:
    --------
    filtered_desc : str
        过滤后的描述
    """

    if mode == 'aggressive':
        # 步骤1: 去除包含键长的完整句子
        # 匹配 "All X–Y bond lengths are ..." 或 "X–Y bond length is ..."
        description = re.sub(
            r'All [A-Za-z0-9()\–\-]+bond lengths? (?:are|is) [^.]*\.',
            '',
            description
        )

        # 步骤2: 去除包含 "shorter" 和 "longer" 的句子
        # 匹配 "There are X shorter (Y Å) and Z longer (W Å) bond lengths."
        description = re.sub(
            r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.',
            '',
            description
        )

        # 步骤3: 去除包含键角的句子
        # 匹配 "The ... tilt angles range from X–Y°" 或 "tilt angles are X°"
        description = re.sub(
            r'The [^.]*(?:tilt|bond) angles? [^.]*\.',
            '',
            description
        )

        # 步骤4: 去除包含 "bond lengths" 的任何句子片段
        description = re.sub(
            r'[^.]*bond lengths?[^.]*\.',
            '',
            description
        )

        # 步骤5: 去除所有带单位的数值（Å, ?, °）
        # 先去除整个短语，如 "(3.60 Å)"
        description = re.sub(
            r'\([^)]*\d+\.\d+\s*[ÅÅ?°][^)]*\)',
            '',
            description
        )

        # 然后去除剩余的数值
        description = re.sub(
            r'\d+\.\d+\s*[ÅÅ?°]',
            '',
            description
        )

        # 去除单独的数字+单位（如 "3 Å"）
        description = re.sub(
            r'\d+\s*[ÅÅ?°]',
            '',
            description
        )

        # 步骤6: 去除残留的数字片段（如 "49 Å", "31 Å"）
        # 匹配孤立的数字后跟单位
        description = re.sub(
            r'\s+\d+\s+[ÅÅ?°]',
            '',
            description
        )

        # 步骤7: 清理格式
        # 去除多余的空格
        description = re.sub(r'\s+', ' ', description)

        # 去除空格+句号
        description = re.sub(r'\s+\.', '.', description)

        # 去除多个句号
        description = re.sub(r'\.+', '.', description)

        # 去除空格+逗号
        description = re.sub(r'\s+,', ',', description)

        # 去除空括号
        description = re.sub(r'\(\s*\)', '', description)

        # 去除句子之间多余的空格
        description = re.sub(r'\.\s+', '. ', description)

        # 去除开头的多余字符
        description = re.sub(r'^\s*[,.\s]+', '', description)

        # 去除残留的孤立数字（如 "49", "31"）后面跟着句号或空格
        description = re.sub(r'\s+\d{1,3}\s+(?=[A-Z])', ' ', description)

        # 去除以数字开头的句子片段
        description = re.sub(r'(?<=\.)\s*\d+[^.]*?(?=[A-Z][a-z])', '', description)

    description = description.strip()

    return description


def batch_filter_csv(input_csv, output_csv):
    """
    批量过滤CSV文件（改进版）
    """
    import csv

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 过滤每一行
    for row in rows:
        if 'description' in row:
            original = row['description']
            filtered = remove_local_information_improved(original, mode='aggressive')

            # 更新过滤后的描述
            if 'description_filtered' in row:
                row['description_filtered'] = filtered
            else:
                row['description'] = filtered

    # 写入新文件
    if rows:
        fieldnames = rows[0].keys()
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ 已处理 {len(rows)} 条记录")
    print(f"   输入: {input_csv}")
    print(f"   输出: {output_csv}")


def test_improved_filter():
    """
    测试改进版过滤器
    """
    test_cases = [
        {
            'name': 'VSe2',
            'desc': 'VSe2 is trigonal omega structured and crystallizes in the trigonal P-3m1 space group. The structure is two-dimensional and consists of one VSe2 sheet oriented in the [(0, 0, 1)] direction. V(1) is bonded to six equivalent Se(1) atoms to form edge-sharing VSe6 octahedra.49 Å. Se(1) is bonded in a distorted T-shaped geometry to three equivalent V(1) atoms.'
        },
        {
            'name': 'Ba4NaBi',
            'desc': 'NaBa4Bi is beta-derived structured and crystallizes in the cubic F-43m space group. Na(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms.31 Å. Ba(1) is bonded to three equivalent Na(1), six equivalent Ba(1), and three equivalent Bi(1) atoms to form a mixture of distorted face, corner, and edge-sharing BaBa6Na3Bi3 cuboctahedra. 61 Å) and three longer Ba(1)–Ba(1) bond lengths.29 Å. Bi(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms.'
        },
        {
            'name': 'AlAs',
            'desc': 'AlAs is Zincblende, Sphalerite structured and crystallizes in the cubic F-43m space group. Al(1) is bonded to four equivalent As(1) atoms to form corner-sharing AlAs4 tetrahedra.48 Å. As(1) is bonded to four equivalent Al(1) atoms to form corner-sharing AsAl4 tetrahedra.'
        }
    ]

    print("=" * 80)
    print(" 改进版过滤器测试")
    print("=" * 80)

    for case in test_cases:
        print(f"\n【{case['name']}】")
        print("-" * 80)

        original = case['desc']
        filtered = remove_local_information_improved(original, mode='aggressive')

        print(f"原始 ({len(original)} 字符):")
        print(f"  {original}")
        print(f"\n过滤后 ({len(filtered)} 字符, 减少 {100*(1-len(filtered)/len(original)):.1f}%):")
        print(f"  {filtered}")

        # 检查是否还有残留的数值
        remaining_numbers = re.findall(r'\d+\s*[ÅÅ?°]', filtered)
        if remaining_numbers:
            print(f"\n⚠️  残留数值: {remaining_numbers}")
        else:
            print(f"\n✅ 已清理所有数值")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # 测试模式
            test_improved_filter()
        elif sys.argv[1] == 'csv' and len(sys.argv) > 3:
            # CSV处理模式
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            batch_filter_csv(input_file, output_file)
        else:
            print("使用方法:")
            print("  python filter_descriptions_improved.py test")
            print("  python filter_descriptions_improved.py csv input.csv output.csv")
    else:
        # 默认运行测试
        test_improved_filter()

        print("\n\n" + "=" * 80)
        print(" 使用方法")
        print("=" * 80)
        print("""
1. 测试过滤效果:
   python filter_descriptions_improved.py test

2. 处理CSV文件:
   python filter_descriptions_improved.py csv desc_mbj_bandgap0_aggressive.csv desc_mbj_bandgap0_cleaned.csv

3. 在Python中使用:
   from filter_descriptions_improved import remove_local_information_improved
   filtered = remove_local_information_improved(description, mode='aggressive')
        """)
