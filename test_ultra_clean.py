#!/usr/bin/env python3
"""
测试超强清理 - 针对残留问题
"""

from ultra_clean import remove_local_information_ultra


def test_problematic_cases():
    """
    测试问题案例
    """

    print("=" * 80)
    print(" 超强清理测试 - 修复残留问题")
    print("=" * 80)

    # 问题1: Ba4NaBi
    print("\n问题 1: Ba4NaBi")
    print("-" * 80)

    ba_original = """NaBa4Bi is beta-derived structured and crystallizes in the cubic F-43m space group. Na(1) is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms. Ba(1) is bonded to three equivalent Na(1), six equivalent Ba(1), and three equivalent Bi(1) atoms to form a mixture of distorted face, corner, and edge-sharing BaBa6Na3Bi3 cuboctahedra. ) and three longer is bonded in a 12-coordinate geometry to twelve equivalent Ba(1) atoms."""

    ba_cleaned = remove_local_information_ultra(ba_original)

    print("原始（有残留）:")
    print(ba_original)
    print("\n清理后:")
    print(ba_cleaned)

    if ") and three longer" in ba_cleaned:
        print("\n❌ 仍有残留")
    else:
        print("\n✅ 残留已清除")


    # 问题2: SrB6
    print("\n\n问题 2: SrB6")
    print("-" * 80)

    sr_original = """SrB6 is Calcium hexaboride structured and crystallizes in the cubic Pm-3m space group. Sr(1) is bonded in a 24-coordinate geometry to twenty-four equivalent B(1) atoms. B(1) is bonded in a 9-coordinate geometry to four equivalent Sr(1) and five equivalent B(1) atoms. ) and four longer (1."""

    sr_cleaned = remove_local_information_ultra(sr_original)

    print("原始（有残留）:")
    print(sr_original)
    print("\n清理后:")
    print(sr_cleaned)

    if ") and four longer" in sr_cleaned or "(1." in sr_cleaned:
        print("\n❌ 仍有残留")
    else:
        print("\n✅ 残留已清除")


    # 完整测试
    print("\n\n" + "=" * 80)
    print(" 完整测试结果")
    print("=" * 80)

    test_cases = [
        ("Ba4NaBi", ba_original, ba_cleaned),
        ("SrB6", sr_original, sr_cleaned)
    ]

    all_clean = True
    for name, orig, clean in test_cases:
        # 检查常见残留模式
        residuals = [
            r'\)\s*and\s+\w+\s+(?:longer|shorter)',
            r'\(\d+\.',
            r'\)\s*and\s+\d+'
        ]

        has_residual = any(re.search(pattern, clean) for pattern in residuals)

        status = "❌ 仍有残留" if has_residual else "✅ 完全清理"
        print(f"{name:10} | {status}")

        if has_residual:
            all_clean = False
            import re
            for pattern in residuals:
                match = re.search(pattern, clean)
                if match:
                    print(f"           | 残留: '{match.group()}'")

    print("-" * 80)
    if all_clean:
        print("✅ 所有案例完全清理")
    else:
        print("⚠️  仍有部分残留，需要进一步调整")


if __name__ == '__main__':
    import re
    test_problematic_cases()
