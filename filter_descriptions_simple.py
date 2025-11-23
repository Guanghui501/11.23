"""
简化版：过滤材料描述中的局部信息（不需要pandas）
"""

import re
import csv


def remove_local_information(description, mode='aggressive'):
    """
    从描述中去除局部信息

    Parameters:
    -----------
    description : str
        原始材料描述
    mode : str
        'aggressive': 去除所有键长、键角、具体数值
        'moderate': 保留配位数，去除键长键角
        'conservative': 只去除键长键角数值，保留其他
    """

    if mode == 'aggressive':
        # 1. 去除键长信息
        description = re.sub(r'[^.]*bond lengths? (?:are|is|range)[^.]*\.', '', description)

        # 2. 去除键角信息
        description = re.sub(r'[^.]*(?:tilt |bond )?angles? (?:are|is|range)[^.]*\.', '', description)

        # 3. 去除包含 "shorter" 和 "longer" 的句子
        description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)

        # 4. 去除具体的数值+单位
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', '[X]', description)

        # 5. 清理 [X]
        description = re.sub(r'\([^)]*\[X\][^)]*\)', '', description)
        description = re.sub(r'\[X\]', '', description)

    elif mode == 'moderate':
        description = re.sub(r'[^.]*bond lengths? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'[^.]*(?:tilt |bond )?angles? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)

    elif mode == 'conservative':
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', 'X', description)
        description = re.sub(r'\d+\s*[ÅÅ?°]', 'X', description)

    # 清理多余的空格和标点
    description = re.sub(r'\s+', ' ', description)
    description = re.sub(r'\s+\.', '.', description)
    description = re.sub(r'\.+', '.', description)
    description = re.sub(r'\s+,', ',', description)
    description = re.sub(r'\(\s*\)', '', description)
    description = description.strip()

    return description


def extract_global_summary(description):
    """
    提取纯全局摘要
    """
    formula = description.split(' is ')[0] if ' is ' in description else description.split()[0]

    # 提取结构类型
    structure_match = re.search(r'is ([A-Z][a-z\s,]+) structured', description)
    structure = structure_match.group(1).strip() if structure_match else None

    # 提取空间群
    space_group_match = re.search(r'space group ([A-Z0-9\-/]+)', description)
    space_group = space_group_match.group(1) if space_group_match else None

    # 提取晶系
    crystal_systems = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal',
                       'trigonal', 'monoclinic', 'triclinic']
    crystal_system = None
    for system in crystal_systems:
        if system in description.lower():
            crystal_system = system
            break

    # 构建摘要
    parts = [formula]
    if structure:
        parts.append(f"has {structure} structure")
    if crystal_system:
        parts.append(f"crystallizes in {crystal_system} system")
    if space_group:
        parts.append(f"space group {space_group}")

    return ' '.join(parts) + '.'


def demo_examples():
    """
    演示示例
    """
    examples = [
        {
            'name': 'Ba4LiHf',
            'desc': 'LiBa4Hf crystallizes in the cubic F-43m space group. The structure consists of four Li clusters inside a Ba4Hf framework. In each Li cluster, Li(1) is bonded in a 12-coordinate geometry to atoms. In the Ba4Hf framework, Ba(1) is bonded in a distorted q6 geometry to six equivalent Ba(1) and three equivalent Hf(1) atoms. There are three shorter (3.60 ?) and three longer (3.66 ?) Ba(1)?Ba(1) bond lengths. All Ba(1)?Hf(1) bond lengths are 4.25 ?. Hf(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms.'
        },
        {
            'name': 'AlAs',
            'desc': 'AlAs is Zincblende, Sphalerite structured and crystallizes in the cubic F-43m space group. Al(1) is bonded to four equivalent As(1) atoms to form corner-sharing AlAs4 tetrahedra. All Al(1)?As(1) bond lengths are 2.48 ?. As(1) is bonded to four equivalent Al(1) atoms to form corner-sharing AsAl4 tetrahedra.'
        },
        {
            'name': 'NaI',
            'desc': 'NaI is Halite, Rock Salt structured and crystallizes in the cubic Fm-3m space group. Na(1) is bonded to six equivalent I(1) atoms to form a mixture of corner and edge-sharing NaI6 octahedra. The corner-sharing octahedra are not tilted. All Na(1)?I(1) bond lengths are 3.21 ?. I(1) is bonded to six equivalent Na(1) atoms to form a mixture of corner and edge-sharing INa6 octahedra. The corner-sharing octahedra are not tilted.'
        }
    ]

    for ex in examples:
        print("\n" + "=" * 80)
        print(f"示例: {ex['name']}")
        print("=" * 80)

        original = ex['desc']
        filtered = remove_local_information(original, mode='aggressive')
        summary = extract_global_summary(original)

        print(f"\n原始长度: {len(original)} 字符")
        print("-" * 80)
        print(original)

        print(f"\n\n过滤后长度: {len(filtered)} 字符 (减少 {100*(1-len(filtered)/len(original)):.1f}%)")
        print("-" * 80)
        print(filtered)

        print(f"\n\n全局摘要: {len(summary)} 字符 (减少 {100*(1-len(summary)/len(original)):.1f}%)")
        print("-" * 80)
        print(summary)
        print()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" 材料描述全局信息过滤工具")
    print("=" * 80)

    demo_examples()

    print("\n" + "=" * 80)
    print(" 信息层级说明")
    print("=" * 80)
    print("""
全局信息 (保留✅):
  • 晶体结构类型: "Halite", "Zincblende"
  • 空间群: "F-43m", "Fm-3m"
  • 晶系: "cubic", "trigonal"
  • 维度: "one-dimensional"

半全局信息 (保留✅):
  • 配位几何: "octahedral", "12-coordinate"
  • 成键拓扑: "corner-sharing", "edge-sharing"
  • 原子连接: "bonded to six atoms"

局部信息 (去除❌):
  • 键长数值: "2.48 Å", "3.60 Å"
  • 键角数值: "40-54°"
  • 具体距离: "All Ba(1)-Hf(1) bond lengths are 4.25 Å"
    """)

    print("\n" + "=" * 80)
    print(" 使用方法")
    print("=" * 80)
    print("""
在Python中使用:

    from filter_descriptions_simple import remove_local_information

    original_desc = "Your description here..."

    # Aggressive模式 (推荐)
    filtered = remove_local_information(original_desc, mode='aggressive')

    # Moderate模式
    filtered = remove_local_information(original_desc, mode='moderate')

    # Conservative模式
    filtered = remove_local_information(original_desc, mode='conservative')
    """)
