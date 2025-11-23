"""
过滤材料描述，只保留全局和半全局信息，去除局部细节

全局信息: 晶体结构类型、空间群、晶系、衍生结构
半全局信息: 配位几何、成键方式、结构连接性
局部信息: 具体键长、键角数值 ← 需要去除
"""

import re
import pandas as pd


def classify_information_types():
    """
    定义信息层级分类
    """
    return {
        'global': [
            '晶体结构类型 (e.g., "Halite", "Zincblende")',
            '空间群 (e.g., "Fm-3m", "F-43m")',
            '晶系 (e.g., "cubic", "orthorhombic")',
            '衍生结构 (e.g., "Laves-derived", "beta Vanadium nitride-derived")'
        ],
        'semi_global': [
            '配位几何 (e.g., "octahedral", "tetrahedral", "12-coordinate")',
            '成键拓扑 (e.g., "corner-sharing", "edge-sharing", "face-sharing")',
            '原子连接性 (e.g., "bonded to X atoms")',
            '结构维度 (e.g., "one-dimensional", "zero-dimensional")'
        ],
        'local': [
            '键长数值 (e.g., "2.48 Å", "3.61 Å")',
            '键角数值 (e.g., "40-54°")',
            '具体原子标签 (e.g., "Fe(1)", "Ba(1)")',
            '精确配位数 (具体的"three", "four", "six"等)'
        ]
    }


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

    Returns:
    --------
    filtered_desc : str
        过滤后的描述
    """

    if mode == 'aggressive':
        # 去除所有包含数值的句子
        # 1. 去除键长信息 (e.g., "All Ba(1)-Hf(1) bond lengths are 4.25 Å.")
        description = re.sub(r'[^.]*bond lengths? (?:are|is|range)[^.]*\.', '', description)

        # 2. 去除键角信息 (e.g., "tilt angles range from 40-54°")
        description = re.sub(r'[^.]*(?:tilt |bond )?angles? (?:are|is|range)[^.]*\.', '', description)

        # 3. 去除包含 "shorter" 和 "longer" 的句子
        description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)

        # 4. 去除具体的数值+单位 (如 "2.48 Å", "3.61 ?")
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', '[removed]', description)

        # 5. 去除包含 [removed] 的整个短语
        description = re.sub(r'\([^)]*\[removed\][^)]*\)', '', description)
        description = re.sub(r'\[removed\]', '', description)

    elif mode == 'moderate':
        # 只去除键长键角，保留配位描述
        description = re.sub(r'[^.]*bond lengths? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'[^.]*(?:tilt |bond )?angles? (?:are|is|range)[^.]*\.', '', description)
        description = re.sub(r'There (?:is|are) [^.]*(?:shorter|longer)[^.]*\.', '', description)

    elif mode == 'conservative':
        # 只去除数值本身，保留句子结构
        description = re.sub(r'\d+\.\d+\s*[ÅÅ?°]', 'X', description)
        description = re.sub(r'\d+\s*[ÅÅ?°]', 'X', description)

    # 清理多余的空格和标点
    description = re.sub(r'\s+', ' ', description)  # 多个空格 → 单个空格
    description = re.sub(r'\s+\.', '.', description)  # 空格+句号 → 句号
    description = re.sub(r'\.+', '.', description)  # 多个句号 → 单个句号
    description = re.sub(r'\s+,', ',', description)  # 空格+逗号 → 逗号
    description = re.sub(r'\(\s*\)', '', description)  # 空括号
    description = description.strip()

    return description


def extract_global_keywords(description):
    """
    提取全局关键词
    """
    keywords = {
        'structure_type': None,
        'space_group': None,
        'crystal_system': None,
        'derived_from': None
    }

    # 提取结构类型
    structure_patterns = [
        r'is ([A-Z][a-z\s,]+) structured',
        r'is ([A-Z][a-z\s,]+)-derived structured',
        r'is ([A-Z][a-z\s,]+)-like'
    ]
    for pattern in structure_patterns:
        match = re.search(pattern, description)
        if match:
            keywords['structure_type'] = match.group(1).strip()
            break

    # 提取空间群
    space_group_match = re.search(r'space group ([A-Z0-9\-/]+)', description)
    if space_group_match:
        keywords['space_group'] = space_group_match.group(1)

    # 提取晶系
    crystal_systems = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal',
                       'trigonal', 'monoclinic', 'triclinic']
    for system in crystal_systems:
        if system in description.lower():
            keywords['crystal_system'] = system
            break

    # 提取衍生信息
    derived_match = re.search(r'([A-Za-z\s]+)-derived', description)
    if derived_match:
        keywords['derived_from'] = derived_match.group(1).strip()

    return keywords


def create_global_summary(description, keywords):
    """
    创建纯全局摘要
    """
    summary_parts = []

    formula = description.split(' is ')[0] if ' is ' in description else description.split()[0]
    summary_parts.append(formula)

    if keywords['structure_type']:
        summary_parts.append(f"has {keywords['structure_type']} structure")

    if keywords['crystal_system']:
        summary_parts.append(f"crystallizes in {keywords['crystal_system']} system")

    if keywords['space_group']:
        summary_parts.append(f"space group {keywords['space_group']}")

    return ' '.join(summary_parts) + '.'


def process_descriptions(csv_file, output_file, mode='aggressive',
                        include_global_summary=True):
    """
    处理整个CSV文件

    Parameters:
    -----------
    csv_file : str
        输入CSV文件路径
    output_file : str
        输出CSV文件路径
    mode : str
        过滤模式 ('aggressive', 'moderate', 'conservative')
    include_global_summary : bool
        是否添加纯全局摘要列
    """

    # 读取CSV
    df = pd.read_csv(csv_file, header=None,
                     names=['id', 'formula', 'bandgap', 'description', 'source'])

    # 处理每一行
    filtered_descriptions = []
    global_summaries = []

    for idx, row in df.iterrows():
        original_desc = row['description']

        # 过滤局部信息
        filtered_desc = remove_local_information(original_desc, mode=mode)
        filtered_descriptions.append(filtered_desc)

        # 提取全局关键词并创建摘要
        if include_global_summary:
            keywords = extract_global_keywords(original_desc)
            global_summary = create_global_summary(original_desc, keywords)
            global_summaries.append(global_summary)

    # 添加新列
    df['description_filtered'] = filtered_descriptions

    if include_global_summary:
        df['global_summary'] = global_summaries

    # 保存
    df.to_csv(output_file, index=False)

    print(f"✅ 处理完成!")
    print(f"   输入: {csv_file}")
    print(f"   输出: {output_file}")
    print(f"   模式: {mode}")
    print(f"   总行数: {len(df)}")

    return df


def compare_descriptions(original, filtered, global_summary=None):
    """
    对比显示原始描述和过滤后的描述
    """
    print("=" * 80)
    print("原始描述 (包含局部信息):")
    print("-" * 80)
    print(original)
    print("\n" + "=" * 80)
    print("过滤后描述 (只保留全局/半全局信息):")
    print("-" * 80)
    print(filtered)

    if global_summary:
        print("\n" + "=" * 80)
        print("纯全局摘要:")
        print("-" * 80)
        print(global_summary)

    print("=" * 80)


# ============ 示例用法 ============

if __name__ == '__main__':

    # 示例1: 测试单个描述
    print("\n" + "=" * 80)
    print("示例 1: Ba4LiHf 描述过滤")
    print("=" * 80 + "\n")

    original = """LiBa4Hf crystallizes in the cubic F-43m space group. The structure consists of four Li clusters inside a Ba4Hf framework. In each Li cluster, Li(1) is bonded in a 12-coordinate geometry to atoms. In the Ba4Hf framework, Ba(1) is bonded in a distorted q6 geometry to six equivalent Ba(1) and three equivalent Hf(1) atoms. There are three shorter (3.60 Å) and three longer (3.66 Å) Ba(1)-Ba(1) bond lengths. All Ba(1)-Hf(1) bond lengths are 4.25 Å. Hf(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms."""

    filtered_aggressive = remove_local_information(original, mode='aggressive')
    keywords = extract_global_keywords(original)
    global_summary = create_global_summary(original, keywords)

    compare_descriptions(original, filtered_aggressive, global_summary)


    # 示例2: 处理整个CSV文件
    print("\n" + "=" * 80)
    print("示例 2: 处理 CSV 文件")
    print("=" * 80 + "\n")

    # 假设您的文件名为 desc_mbj_bandgap0.csv
    input_file = 'desc_mbj_bandgap0.csv'

    # 三种模式的输出
    modes = ['aggressive', 'moderate', 'conservative']

    for mode in modes:
        output_file = f'desc_mbj_bandgap0_{mode}.csv'

        try:
            df = process_descriptions(
                input_file,
                output_file,
                mode=mode,
                include_global_summary=True
            )

            print(f"\n✅ {mode.upper()} 模式处理完成")
            print(f"   示例对比 (第1行):")
            print(f"   原始长度: {len(df.iloc[0]['description'])} 字符")
            print(f"   过滤长度: {len(df.iloc[0]['description_filtered'])} 字符")
            print(f"   压缩率: {(1 - len(df.iloc[0]['description_filtered']) / len(df.iloc[0]['description'])) * 100:.1f}%")

        except FileNotFoundError:
            print(f"⚠️  文件未找到: {input_file}")
            print(f"   请确保文件在当前目录下")


    # 示例3: 信息层级说明
    print("\n" + "=" * 80)
    print("信息层级分类说明")
    print("=" * 80 + "\n")

    info_types = classify_information_types()

    for level, items in info_types.items():
        print(f"📊 {level.upper()} 信息:")
        for item in items:
            print(f"   • {item}")
        print()
