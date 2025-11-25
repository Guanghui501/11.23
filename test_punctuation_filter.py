#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试标点符号过滤功能
验证所有标点符号都不会被分配注意力
"""

import sys
sys.path.append('.')

from interpretability_enhanced import EnhancedMaterialsInterpretability

def test_punctuation_filtering():
    """测试各种标点符号是否被正确过滤"""

    # 创建实例
    analyzer = EnhancedMaterialsInterpretability()

    # 英文标点符号
    english_punctuations = [
        '.', ',', '!', '?', ';', ':',
        '(', ')', '[', ']', '{', '}', '<', '>',
        '"', "'", '`',
        '-', '–', '—', '_',
        '/', '\\', '|',
        '+', '=', '*', '&', '%', '$', '#', '@', '^', '~',
    ]

    # 中文标点符号
    chinese_punctuations = [
        '，', '。', '！', '？', '；', '：',
        '（', '）', '【', '】', '《', '》', '〈', '〉',
        '"', '"', ''', ''', '、', '·',
        '……', '～',
    ]

    # 组合标点符号字符串
    combined_punctuations = [
        '...', '!!!', '???', '---', '___',
        '(*)', '[1]', '{x}', '<>',
        '....',  # 多个相同标点
        '.,;',   # 多个不同标点
        '（1）', '【注】',  # 中文标点组合
    ]

    # BERT 特殊 tokens
    special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']

    # WordPiece 词缀
    wordpiece_fragments = ['##s', '##ed', '##ing', '##ly', '##er', '##est']

    # 不应该被过滤的词（正常词汇和元素符号）
    valid_words = [
        'oxygen', 'MgO', 'cubic', 'structure', 'coordination',
        'O', 'N', 'C', 'H', 'Mg', 'Ca', 'Fe',  # 元素符号（大写）
        'α', 'β', 'γ',  # 希腊字母
        'F-43m', 'I4/mmm',  # 空间群（包含标点但不是纯标点）
        '12-coordinate',  # 含连字符的化学术语
        'see-saw-like',  # 含连字符的复合词
    ]

    print("=" * 80)
    print("测试标点符号过滤功能")
    print("=" * 80)

    # 测试英文标点符号
    print("\n【测试 1】英文标点符号（应该被过滤）:")
    all_passed = True
    for punct in english_punctuations:
        is_stop = analyzer.is_stopword(punct)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        if not is_stop:
            all_passed = False
        print(f"  '{punct:3s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    # 测试中文标点符号
    print("\n【测试 2】中文标点符号（应该被过滤）:")
    all_passed = True
    for punct in chinese_punctuations:
        is_stop = analyzer.is_stopword(punct)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        if not is_stop:
            all_passed = False
        print(f"  '{punct:3s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    # 测试组合标点符号
    print("\n【测试 3】组合标点符号字符串（应该被过滤）:")
    all_passed = True
    for punct in combined_punctuations:
        is_stop = analyzer.is_stopword(punct)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        if not is_stop:
            all_passed = False
        print(f"  '{punct:10s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    # 测试特殊 tokens
    print("\n【测试 4】BERT 特殊 tokens（应该被过滤）:")
    all_passed = True
    for token in special_tokens:
        is_stop = analyzer.is_stopword(token)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        if not is_stop:
            all_passed = False
        print(f"  '{token:8s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    # 测试 WordPiece 词缀
    print("\n【测试 5】WordPiece 词缀（应该被过滤）:")
    all_passed = True
    for frag in wordpiece_fragments:
        is_stop = analyzer.is_stopword(frag)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        if not is_stop:
            all_passed = False
        print(f"  '{frag:8s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    # 测试正常词汇（不应该被过滤）
    print("\n【测试 6】正常词汇和元素符号（不应该被过滤）:")
    all_passed = True
    for word in valid_words:
        is_stop = analyzer.is_stopword(word)
        status = "✓ PASS" if not is_stop else "✗ FAIL"
        if is_stop:
            all_passed = False
        print(f"  '{word:20s}' -> is_stopword = {is_stop:5} {status}")
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '有失败 ✗'}")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == '__main__':
    test_punctuation_filtering()
