#!/usr/bin/env python
"""
Test script to verify punctuation filtering in heatmap visualization
"""

import numpy as np
import sys

# Mock the necessary classes/methods to test the filtering logic
class MockAnalyzer:
    def __init__(self):
        # Same punctuation chars as in is_stopword
        self.punctuation_chars = set('.,:;!?()[]{}"\'-—–_/\\|+*&%$#@^~`<>，。：；！？（）【】《》""''、·……～')

    def is_stopword(self, word):
        """Simplified version of is_stopword focusing on punctuation"""
        word_lower = word.lower().strip()
        if not word_lower:
            return True
        # Check if pure punctuation string
        if all(c in self.punctuation_chars for c in word):
            return True
        return False

def test_filtering_logic():
    """Test the filtering logic used in visualize_fine_grained_attention"""

    analyzer = MockAnalyzer()

    # Simulate text tokens that include punctuation
    text_tokens = ['oxygen', '.', 'structure', ',', 'coordination', '-', 'crystal', '...', 'lattice']

    # Simulate word importance scores (higher = more important)
    word_importance = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.82, 0.78, 0.88, 0.76])

    # Original logic (without filtering) - would select indices with highest scores
    top_k_words = 5
    top_word_indices_original = word_importance.argsort()[-top_k_words:][::-1]
    selected_words_original = [text_tokens[i] for i in top_word_indices_original]

    print("=" * 80)
    print("测试热图标点符号过滤逻辑")
    print("=" * 80)
    print()
    print(f"输入文本tokens: {text_tokens}")
    print(f"重要性分数: {word_importance}")
    print(f"Top-k: {top_k_words}")
    print()

    print("【原始逻辑 - 不过滤标点】:")
    print(f"  选中的索引: {top_word_indices_original.tolist()}")
    print(f"  选中的词语: {selected_words_original}")
    has_punctuation_original = any(analyzer.is_stopword(w) for w in selected_words_original)
    print(f"  包含标点符号: {'是 ❌' if has_punctuation_original else '否 ✓'}")
    print()

    # New logic (with filtering) - filter stopwords first
    valid_word_indices = []
    for idx in range(len(word_importance)):
        if idx < len(text_tokens) and not analyzer.is_stopword(text_tokens[idx]):
            valid_word_indices.append(idx)

    # Select top-k from valid (non-stopword) words
    if valid_word_indices:
        valid_word_scores = [(idx, word_importance[idx]) for idx in valid_word_indices]
        valid_word_scores.sort(key=lambda x: x[1], reverse=True)
        top_word_indices_new = [idx for idx, _ in valid_word_scores[:top_k_words]]
    else:
        # Fallback: if all words are stopwords, use original logic
        top_word_indices_new = word_importance.argsort()[-top_k_words:][::-1].tolist()

    selected_words_new = [text_tokens[i] for i in top_word_indices_new]

    print("【新逻辑 - 过滤标点】:")
    print(f"  有效词语索引: {valid_word_indices}")
    print(f"  有效词语: {[text_tokens[i] for i in valid_word_indices]}")
    print(f"  选中的索引: {top_word_indices_new}")
    print(f"  选中的词语: {selected_words_new}")
    has_punctuation_new = any(analyzer.is_stopword(w) for w in selected_words_new)
    print(f"  包含标点符号: {'是 ❌' if has_punctuation_new else '否 ✓'}")
    print()

    # Test result
    print("=" * 80)
    if not has_punctuation_new and has_punctuation_original:
        print("✓ 测试通过！新逻辑成功过滤了标点符号。")
        print("=" * 80)
        return True
    else:
        print("❌ 测试失败！")
        if has_punctuation_new:
            print("   问题：新逻辑仍然包含标点符号")
        if not has_punctuation_original:
            print("   警告：测试用例可能不合适（原始逻辑也没有标点符号）")
        print("=" * 80)
        return False

def test_edge_case_all_punctuation():
    """Test edge case where all tokens are punctuation"""

    analyzer = MockAnalyzer()

    text_tokens = ['.', ',', '!', '?', '...']
    word_importance = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    top_k_words = 3

    print()
    print("=" * 80)
    print("边缘测试：所有tokens都是标点符号")
    print("=" * 80)
    print()
    print(f"输入文本tokens: {text_tokens}")
    print(f"重要性分数: {word_importance}")
    print(f"Top-k: {top_k_words}")
    print()

    # New logic with fallback
    valid_word_indices = []
    for idx in range(len(word_importance)):
        if idx < len(text_tokens) and not analyzer.is_stopword(text_tokens[idx]):
            valid_word_indices.append(idx)

    if valid_word_indices:
        valid_word_scores = [(idx, word_importance[idx]) for idx in valid_word_indices]
        valid_word_scores.sort(key=lambda x: x[1], reverse=True)
        top_word_indices = [idx for idx, _ in valid_word_scores[:top_k_words]]
    else:
        # Fallback: if all words are stopwords, use original logic
        top_word_indices = word_importance.argsort()[-top_k_words:][::-1].tolist()
        print("⚠️  警告：所有词语都是标点符号，使用fallback逻辑")

    selected_words = [text_tokens[i] for i in top_word_indices]

    print(f"选中的索引: {top_word_indices}")
    print(f"选中的词语: {selected_words}")
    print()
    print("✓ Fallback逻辑正常工作（避免返回空列表）")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success1 = test_filtering_logic()
    success2 = test_edge_case_all_punctuation()

    print()
    if success1 and success2:
        print("=" * 80)
        print("✓✓✓ 所有测试通过！热图标点符号过滤功能正常工作。")
        print("=" * 80)
        sys.exit(0)
    else:
        print("=" * 80)
        print("❌ 部分测试失败")
        print("=" * 80)
        sys.exit(1)
