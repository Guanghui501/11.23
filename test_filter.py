#!/usr/bin/env python
"""
快速测试过滤器
使用方法: python test_filter.py
"""

from filter_descriptions_simple import remove_local_information, extract_global_summary

def test_with_your_description():
    """
    交互式测试
    """
    print("=" * 80)
    print(" 材料描述过滤器 - 交互式测试")
    print("=" * 80)
    print("\n请输入您的材料描述（输入 'demo' 使用示例，输入 'quit' 退出）:")
    print()

    while True:
        print("-" * 80)
        user_input = input("描述 >>> ").strip()

        if user_input.lower() == 'quit':
            print("\n再见！")
            break

        if user_input.lower() == 'demo':
            description = """LiBa4Hf crystallizes in the cubic F-43m space group. The structure consists of four Li clusters inside a Ba4Hf framework. In each Li cluster, Li(1) is bonded in a 12-coordinate geometry to atoms. In the Ba4Hf framework, Ba(1) is bonded in a distorted q6 geometry to six equivalent Ba(1) and three equivalent Hf(1) atoms. There are three shorter (3.60 Å) and three longer (3.66 Å) Ba(1)-Ba(1) bond lengths. All Ba(1)-Hf(1) bond lengths are 4.25 Å. Hf(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms."""
            print(f"\n使用示例描述:\n{description[:100]}...\n")
        elif user_input:
            description = user_input
        else:
            continue

        # 原始描述
        print(f"\n📄 原始描述 ({len(description)} 字符):")
        print("-" * 80)
        print(description)

        # Aggressive过滤
        aggressive = remove_local_information(description, mode='aggressive')
        print(f"\n🔥 Aggressive 模式 ({len(aggressive)} 字符, 减少 {100*(1-len(aggressive)/len(description)):.1f}%):")
        print("-" * 80)
        print(aggressive)

        # Moderate过滤
        moderate = remove_local_information(description, mode='moderate')
        print(f"\n⚡ Moderate 模式 ({len(moderate)} 字符, 减少 {100*(1-len(moderate)/len(description)):.1f}%):")
        print("-" * 80)
        print(moderate)

        # 全局摘要
        summary = extract_global_summary(description)
        print(f"\n📋 全局摘要 ({len(summary)} 字符, 减少 {100*(1-len(summary)/len(description)):.1f}%):")
        print("-" * 80)
        print(summary)

        print("\n" + "=" * 80)
        print()


def quick_test():
    """
    快速测试几个示例
    """
    print("\n" + "=" * 80)
    print(" 快速测试 - 3个示例")
    print("=" * 80)

    examples = [
        ("LiBa4Hf", "LiBa4Hf crystallizes in the cubic F-43m space group. All Ba(1)-Hf(1) bond lengths are 4.25 Å."),
        ("AlAs", "AlAs is Zincblende structured and crystallizes in the cubic F-43m space group. All Al(1)-As(1) bond lengths are 2.48 Å."),
        ("NaI", "NaI is Halite structured and crystallizes in the cubic Fm-3m space group. All Na(1)-I(1) bond lengths are 3.21 Å.")
    ]

    for name, desc in examples:
        filtered = remove_local_information(desc, mode='aggressive')
        summary = extract_global_summary(desc)

        print(f"\n【{name}】")
        print(f"原始 ({len(desc)} 字符):")
        print(f"  {desc}")
        print(f"\n过滤 ({len(filtered)} 字符, -{100*(1-len(filtered)/len(desc)):.0f}%):")
        print(f"  {filtered}")
        print(f"\n摘要 ({len(summary)} 字符, -{100*(1-len(summary)/len(desc)):.0f}%):")
        print(f"  {summary}")
        print("-" * 80)


if __name__ == '__main__':
    import sys

    print("\n" + "=" * 80)
    print(" filter_global_information.py 测试工具")
    print("=" * 80)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            # 快速测试模式
            quick_test()
        elif sys.argv[1] == 'interactive':
            # 交互模式
            test_with_your_description()
        else:
            # 直接处理命令行参数
            description = ' '.join(sys.argv[1:])
            print(f"\n处理您的描述:\n{description}\n")
            filtered = remove_local_information(description, mode='aggressive')
            print(f"\n过滤结果:\n{filtered}\n")
    else:
        # 默认：快速测试 + 交互选项
        quick_test()

        print("\n\n" + "=" * 80)
        print(" 使用方法")
        print("=" * 80)
        print("""
1. 快速测试:
   python test_filter.py quick

2. 交互模式:
   python test_filter.py interactive

3. 直接处理:
   python test_filter.py "Your description here..."

4. 在代码中使用:
   from filter_descriptions_simple import remove_local_information
   filtered = remove_local_information(desc, mode='aggressive')
        """)

        print("\n继续交互测试? (y/n): ", end='')
        choice = input().strip().lower()
        if choice == 'y':
            test_with_your_description()
