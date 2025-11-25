#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试标点符号过滤功能（简化版，不需要torch）
验证所有标点符号都不会被分配注意力
"""


def is_stopword(word, stopwords_set):
    """检查是否为停用词（包括所有标点符号）"""
    word_lower = word.lower().strip()

    # 空字符串
    if not word_lower:
        return True

    # 检查是否在停用词列表中
    if word_lower in stopwords_set or word in stopwords_set:
        return True

    # 检查是否为纯标点符号字符串（例如："...", "---", "!!!"）
    # 定义所有标点符号字符（包括中英文）
    punctuation_chars = set('.,:;!?()[]{}"\'-—–_/\\|+*&%$#@^~`<>，。：；！？（）【】《》""''、·……～')
    if all(c in punctuation_chars for c in word):
        return True

    # 检查是否为WordPiece碎片（以##开头且长度小于5）
    if word.startswith('##') and len(word) < 5:
        return True

    # 检查是否为单个字符（除了大写字母如元素符号O/N/C，或希腊字母α/β/γ）
    if len(word_lower) == 1:
        # 保留ASCII大写字母（元素符号）
        if word.isupper():
            return False
        # 保留希腊字母 (U+03B1-U+03C9 小写, U+0391-U+03A9 大写)
        char_code = ord(word_lower)
        if 0x03B1 <= char_code <= 0x03C9 or 0x0391 <= char_code <= 0x03A9:
            return False
        # 其他单字符都过滤
        return True

    return False


def load_stopwords_set():
    """创建停用词集合（模拟真实的停用词加载）"""
    stopwords = set()

    # BERT 特殊 tokens
    special_tokens = {
        # BERT 特殊 tokens
        '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
        # 英文标点符号（所有常见标点）
        '.', ',', '!', '?', ';', ':',
        '(', ')', '[', ']', '{', '}', '<', '>',
        '"', "'", '`',
        '-', '–', '—', '_',
        '/', '\\', '|',
        '+', '=', '*', '&', '%', '$', '#', '@', '^', '~',
        # 中文标点符号
        '，', '。', '！', '？', '；', '：',
        '（', '）', '【', '】', '《', '》', '〈', '〉',
        '"', '"', ''', ''', '、', '·',
        '……', '—', '～',
        # WordPiece 词缀
        '##s', '##ed', '##ing', '##ly', '##er', '##est', '##tion', '##ment',
    }
    stopwords.update(special_tokens)

    # 添加一些常见的英文停用词（可选）
    common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    stopwords.update(common_stopwords)

    return stopwords


def test_punctuation_filtering():
    """测试各种标点符号是否被正确过滤"""

    # 创建停用词集合
    stopwords_set = load_stopwords_set()

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

    # 组合标点符号字符串（纯标点符号）
    combined_punctuations = [
        '...', '!!!', '???', '---', '___',
        '(*)', '<>',
        '....',  # 多个相同标点
        '.,;',   # 多个不同标点
        '（）', '【】',  # 纯中文标点
    ]

    # 包含字母/数字的标点组合（不应该被过滤）
    punctuation_with_content = [
        '[1]', '{x}', '（1）', '【注】',
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

    total_tests = 0
    failed_tests = 0

    # 测试英文标点符号
    print("\n【测试 1】英文标点符号（应该被过滤）:")
    for punct in english_punctuations:
        is_stop = is_stopword(punct, stopwords_set)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        total_tests += 1
        if not is_stop:
            failed_tests += 1
        print(f"  '{punct:3s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(english_punctuations) - failed_tests}/{len(english_punctuations)}")

    # 测试中文标点符号
    print("\n【测试 2】中文标点符号（应该被过滤）:")
    test_start = failed_tests
    for punct in chinese_punctuations:
        is_stop = is_stopword(punct, stopwords_set)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        total_tests += 1
        if not is_stop:
            failed_tests += 1
        print(f"  '{punct:3s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(chinese_punctuations) - (failed_tests - test_start)}/{len(chinese_punctuations)}")

    # 测试组合标点符号
    print("\n【测试 3】组合标点符号字符串/纯标点（应该被过滤）:")
    test_start = failed_tests
    for punct in combined_punctuations:
        is_stop = is_stopword(punct, stopwords_set)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        total_tests += 1
        if not is_stop:
            failed_tests += 1
        print(f"  '{punct:10s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(combined_punctuations) - (failed_tests - test_start)}/{len(combined_punctuations)}")

    # 测试包含字母/数字的标点组合
    print("\n【测试 3b】包含字母/数字的标点组合（不应该被过滤）:")
    test_start = failed_tests
    for punct in punctuation_with_content:
        is_stop = is_stopword(punct, stopwords_set)
        status = "✓ PASS" if not is_stop else "✗ FAIL"
        total_tests += 1
        if is_stop:
            failed_tests += 1
        print(f"  '{punct:10s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(punctuation_with_content) - (failed_tests - test_start)}/{len(punctuation_with_content)}")

    # 测试特殊 tokens
    print("\n【测试 4】BERT 特殊 tokens（应该被过滤）:")
    test_start = failed_tests
    for token in special_tokens:
        is_stop = is_stopword(token, stopwords_set)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        total_tests += 1
        if not is_stop:
            failed_tests += 1
        print(f"  '{token:8s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(special_tokens) - (failed_tests - test_start)}/{len(special_tokens)}")

    # 测试 WordPiece 词缀
    print("\n【测试 5】WordPiece 词缀（应该被过滤）:")
    test_start = failed_tests
    for frag in wordpiece_fragments:
        is_stop = is_stopword(frag, stopwords_set)
        status = "✓ PASS" if is_stop else "✗ FAIL"
        total_tests += 1
        if not is_stop:
            failed_tests += 1
        print(f"  '{frag:8s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(wordpiece_fragments) - (failed_tests - test_start)}/{len(wordpiece_fragments)}")

    # 测试正常词汇（不应该被过滤）
    print("\n【测试 6】正常词汇、元素符号和希腊字母（不应该被过滤）:")
    test_start = failed_tests
    for word in valid_words:
        is_stop = is_stopword(word, stopwords_set)
        status = "✓ PASS" if not is_stop else "✗ FAIL"
        total_tests += 1
        if is_stop:
            failed_tests += 1
        print(f"  '{word:20s}' -> is_stopword = {is_stop:5} {status}")
    print(f"  通过: {len(valid_words) - (failed_tests - test_start)}/{len(valid_words)}")

    print("\n" + "=" * 80)
    print(f"测试完成: {total_tests - failed_tests}/{total_tests} 通过")
    if failed_tests == 0:
        print("✓ 所有测试通过！标点符号过滤功能正常工作。")
    else:
        print(f"✗ 有 {failed_tests} 个测试失败。")
    print("=" * 80)

    return failed_tests == 0


if __name__ == '__main__':
    success = test_punctuation_filtering()
    exit(0 if success else 1)
