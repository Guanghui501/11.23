#!/usr/bin/env python
"""
Debug script to check how Ca(1) and Ca(1)O6 tokens are processed
"""

import numpy as np

# Copied from interpretability_enhanced.py
def _merge_tokens_and_weights(tokens, weights):
    """Merge WordPiece tokens and their corresponding attention weights."""

    # Space group starting letters (Bravais lattice symbols)
    space_group_starters = {'P', 'I', 'F', 'R', 'C', 'A', 'B'}
    # Characters that are part of space group notation
    space_group_chars = {'-', '/', 'm', 'n', 'c', 'a', 'b', 'd', 'e'}
    # Common atom symbols (1-2 letters)
    atom_symbols = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                   'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                   'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                   'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                   'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                   'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                   'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'}

    merged_tokens = []
    token_mapping = []  # Each element is a list of original indices
    current_token = ""
    current_indices = []
    in_space_group = False
    in_coordinate = False
    in_parentheses = False
    after_closing_paren = False
    in_hyphenated_word = False

    for i, token in enumerate(tokens):
        if token.startswith("##"):
            current_token += token[2:]
            current_indices.append(i)
        elif token == '-' and current_token:
            if current_token[0].upper() in space_group_starters and len(current_token) <= 4:
                current_token += token
                current_indices.append(i)
                in_space_group = True
            elif current_token.isdigit():
                current_token += token
                current_indices.append(i)
                in_coordinate = True
            else:
                token_without_hyphens = current_token.replace('##', '').replace('-', '')
                is_compound_word = (token_without_hyphens.isalpha() and
                                   len(current_token) >= 2 and
                                   not (current_token in atom_symbols or current_token.capitalize() in atom_symbols))
                if is_compound_word:
                    current_token += token
                    current_indices.append(i)
                    in_hyphenated_word = True
                else:
                    if current_token:
                        merged_tokens.append(current_token)
                        token_mapping.append(current_indices)
                        in_space_group = False
                        in_coordinate = False
                        after_closing_paren = False
                        in_hyphenated_word = False
                    current_token = token
                    current_indices = [i]
        elif token == '/' and current_token:
            if current_token[0].upper() in space_group_starters:
                current_token += token
                current_indices.append(i)
                in_space_group = True
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    after_closing_paren = False
                    in_hyphenated_word = False
                current_token = token
                current_indices = [i]
        elif in_coordinate and token.isalpha():
            current_token += token
            current_indices.append(i)
            in_coordinate = False
        elif in_space_group:
            if token.isdigit() or (token.lower() in space_group_chars):
                current_token += token
                current_indices.append(i)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                in_space_group = False
                after_closing_paren = False
                in_hyphenated_word = False
                current_token = token
                current_indices = [i]
        elif token == '(' and current_token:
            current_token += token
            current_indices.append(i)
            in_parentheses = True
        elif token == ')' and current_token and in_parentheses:
            current_token += token
            current_indices.append(i)
            in_parentheses = False
            after_closing_paren = True
        elif in_parentheses and (token.isdigit() or token.isalpha()):
            current_token += token
            current_indices.append(i)
        elif after_closing_paren and (token in atom_symbols or token.capitalize() in atom_symbols or token.isdigit()):
            current_token += token
            current_indices.append(i)
            if token.isdigit():
                after_closing_paren = False
        elif token.isdigit() and current_token and not current_token[-1].isdigit():
            is_atom = current_token in atom_symbols or current_token.capitalize() in atom_symbols
            is_short_space_group = (len(current_token) <= 2 and current_token[0].upper() in space_group_starters)
            if is_atom or is_short_space_group:
                current_token += token
                current_indices.append(i)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    in_space_group = False
                    after_closing_paren = False
                    in_hyphenated_word = False
                current_token = token
                current_indices = [i]
        elif token == '.' and current_token:
            is_numeric = current_token.replace('-', '').replace('/', '').isdigit()
            ends_with_paren = current_token.endswith(')')
            is_space_group = len(current_token) <= 6 and any(c in current_token for c in ['-', '/'])

            if is_numeric and not is_space_group and not ends_with_paren:
                current_token += token
                current_indices.append(i)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    in_space_group = False
                    in_coordinate = False
                    in_parentheses = False
                    after_closing_paren = False
                    in_hyphenated_word = False
                current_token = token
                current_indices = [i]
        elif in_hyphenated_word and token.replace('##', '').isalpha():
            current_token += token
            current_indices.append(i)
            in_hyphenated_word = False
        else:
            if current_token and not in_hyphenated_word:
                merged_tokens.append(current_token)
                token_mapping.append(current_indices)
                in_space_group = False
                in_coordinate = False
                in_parentheses = False
                after_closing_paren = False
                current_token = token
                current_indices = [i]
            elif current_token and in_hyphenated_word:
                merged_tokens.append(current_token)
                token_mapping.append(current_indices)
                in_space_group = False
                in_coordinate = False
                in_parentheses = False
                after_closing_paren = False
                in_hyphenated_word = False
                current_token = token
                current_indices = [i]
            else:
                current_token = token
                current_indices = [i]
            if token.upper() in space_group_starters:
                in_space_group = True
            else:
                in_space_group = False

    # Don't forget the last token
    if current_token:
        merged_tokens.append(current_token)
        token_mapping.append(current_indices)

    # Merge weights by taking maximum over grouped indices
    if weights is not None and len(weights.shape) >= 1:
        if len(weights.shape) == 1:
            merged_weights = np.array([weights[indices].max() for indices in token_mapping])
        elif len(weights.shape) == 2:
            if weights.shape[-1] == len(tokens):
                merged_weights = np.array([[weights[i, indices].max() for indices in token_mapping]
                                           for i in range(weights.shape[0])])
            else:
                merged_weights = np.array([[weights[indices, i].max() for indices in token_mapping]
                                           for i in range(weights.shape[1])]).T
        else:
            merged_weights = weights
    else:
        merged_weights = weights

    return merged_tokens, merged_weights, token_mapping


def test_ca_token_scenario():
    """Test the scenario where Ca(1) and Ca(1)O6 both appear"""

    print("="*80)
    print("测试场景：文本中同时包含 Ca(1) 和 Ca(1)O6")
    print("="*80)
    print()

    # Simulate a realistic tokenization from BERT
    # Text: "Ca(1) atoms bonded to Ca(1)O6 octahedra"
    tokens = ['ca', '(', '1', ')', 'atoms', 'bonded', 'to', 'ca', '(', '1', ')', 'o', '6', 'octahedra']

    # Simulate attention weights (higher = more important)
    # Let's say Ca(1)O6 has higher importance than standalone Ca(1)
    weights = np.array([
        0.3, 0.3, 0.3, 0.3,  # ca(1) - indices 0-3, importance ~0.3
        0.1, 0.1, 0.1,        # atoms bonded to - indices 4-6
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # ca(1)o6 - indices 7-12, importance ~0.5
        0.2                   # octahedra - index 13
    ])

    print(f"原始tokens ({len(tokens)}):")
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        print(f"  [{i:2d}] {tok:12s} -> weight: {w:.2f}")
    print()

    # Merge tokens
    merged_tokens, merged_weights, token_mapping = _merge_tokens_and_weights(tokens, weights)

    print(f"合并后tokens ({len(merged_tokens)}):")
    for i, (tok, indices) in enumerate(zip(merged_tokens, token_mapping)):
        orig_weights = [weights[j] for j in indices]
        merged_w = merged_weights[i]
        print(f"  [{i:2d}] {tok:12s} <- indices {indices} (原始权重: {orig_weights}, 合并后: {merged_w:.2f})")
    print()

    # Check if Ca(1) and Ca(1)O6 are separate
    ca1_indices = [i for i, tok in enumerate(merged_tokens) if tok == 'ca(1)']
    ca1o6_indices = [i for i, tok in enumerate(merged_tokens) if tok == 'ca(1)o6']

    print("分析：")
    print(f"  'ca(1)' 出现次数: {len(ca1_indices)}")
    if ca1_indices:
        for idx in ca1_indices:
            print(f"    位置 {idx}: 权重 = {merged_weights[idx]:.2f}, 映射自原始indices {token_mapping[idx]}")

    print(f"  'ca(1)o6' 出现次数: {len(ca1o6_indices)}")
    if ca1o6_indices:
        for idx in ca1o6_indices:
            print(f"    位置 {idx}: 权重 = {merged_weights[idx]:.2f}, 映射自原始indices {token_mapping[idx]}")
    print()

    # Verify they are independent
    if len(ca1_indices) == 1 and len(ca1o6_indices) == 1:
        ca1_mapped = set(token_mapping[ca1_indices[0]])
        ca1o6_mapped = set(token_mapping[ca1o6_indices[0]])
        overlap = ca1_mapped & ca1o6_mapped

        if overlap:
            print(f"❌ 错误：'ca(1)' 和 'ca(1)o6' 的token indices有重叠！{overlap}")
            print(f"   这意味着权重可能被错误地共享或重复计算")
        else:
            print(f"✅ 正确：'ca(1)' 和 'ca(1)o6' 的token indices完全独立，没有重叠")
            print(f"   ca(1) 映射: {ca1_mapped}")
            print(f"   ca(1)o6 映射: {ca1o6_mapped}")
    print()
    print("="*80)


def test_multiple_ca1_occurrences():
    """Test when Ca(1) appears multiple times in text"""

    print()
    print("="*80)
    print("测试场景：Ca(1) 多次出现")
    print("="*80)
    print()

    # Text: "Ca(1) atoms form Ca(1) bonded Ca(1)O6"
    tokens = ['ca', '(', '1', ')', 'atoms', 'form', 'ca', '(', '1', ')', 'bonded', 'ca', '(', '1', ')', 'o', '6']

    # Assign different weights to each occurrence
    weights = np.array([
        0.2, 0.2, 0.2, 0.2,  # First ca(1) - indices 0-3
        0.1, 0.1,            # atoms form - indices 4-5
        0.4, 0.4, 0.4, 0.4,  # Second ca(1) - indices 6-9
        0.15,                # bonded - index 10
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # ca(1)o6 - indices 11-16
    ])

    print(f"原始tokens ({len(tokens)}):")
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        print(f"  [{i:2d}] {tok:12s} -> weight: {w:.2f}")
    print()

    merged_tokens, merged_weights, token_mapping = _merge_tokens_and_weights(tokens, weights)

    print(f"合并后tokens ({len(merged_tokens)}):")
    for i, (tok, indices) in enumerate(zip(merged_tokens, token_mapping)):
        orig_weights = [weights[j] for j in indices]
        merged_w = merged_weights[i]
        print(f"  [{i:2d}] {tok:12s} <- indices {indices} (max of {orig_weights}, merged: {merged_w:.2f})")
    print()

    # Count occurrences
    ca1_count = merged_tokens.count('ca(1)')
    ca1o6_count = merged_tokens.count('ca(1)o6')

    print("分析：")
    print(f"  'ca(1)' 出现 {ca1_count} 次（期望: 2次）")
    print(f"  'ca(1)o6' 出现 {ca1o6_count} 次（期望: 1次）")

    if ca1_count == 2 and ca1o6_count == 1:
        print(f"  ✅ 正确：所有出现都被独立识别和合并")
    else:
        print(f"  ❌ 错误：token合并逻辑可能有问题")

    print()
    print("="*80)


if __name__ == "__main__":
    test_ca_token_scenario()
    test_multiple_ca1_occurrences()

    print()
    print("结论：")
    print("如果以上测试都通过，说明token合并逻辑是正确的。")
    print("如果热图中 'ca(1)' 的重要性很高，可能的原因是：")
    print("  1. 文本中 'ca(1)' 确实出现多次，且都有较高的attention权重")
    print("  2. 原始文本中只有 'ca(1)' 而没有 'ca(1)o6'")
    print("  3. 需要检查原始文本内容来确定具体情况")
