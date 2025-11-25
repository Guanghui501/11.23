#!/usr/bin/env python3
"""
Test script to verify token merging improvements for:
1. Hyphenated compound words (see-saw-like)
2. Chemical formulas with site numbers (Ca(1)O6)

This version extracts only the _merge_tokens_and_weights function to avoid torch dependency.
"""

import sys
import numpy as np


# Extracted from interpretability_enhanced.py
def _merge_tokens_and_weights(tokens, weights):
    """Merge WordPiece tokens and their corresponding attention weights.

    Handles space groups (F-43m, P63/mmc), N-coordinate (12-coordinate),
    atom labels with site numbers (Ba(1)), and element symbols with digits (Ba4),
    hyphenated compound words (see-saw-like), and chemical formulas (Ca(1)O6).

    Args:
        tokens: list of token strings
        weights: numpy array of shape [..., seq_len]

    Returns:
        merged_tokens: list of merged token strings
        merged_weights: numpy array with merged weights
        token_mapping: list mapping merged index to original indices
    """
    import numpy as np

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
    in_space_group = False  # Track if we're in a space group symbol
    in_coordinate = False  # Track if we're in a N-coordinate pattern
    in_parentheses = False  # Track if we're inside parentheses like Ba(1)
    after_closing_paren = False  # Track if we just closed a parenthesis (for chemical formulas like Ca(1)O6)
    in_hyphenated_word = False  # Track if we're in a hyphenated compound word like "see-saw-like"

    for i, token in enumerate(tokens):
        if token.startswith("##"):
            # Continue previous token (WordPiece continuation)
            current_token += token[2:]
            current_indices.append(i)
        elif token == '-' and current_token:
            # Check if this is part of space group or N-coordinate pattern
            # Use upper() for case-insensitive space group detection (BERT lowercases)
            if current_token[0].upper() in space_group_starters and len(current_token) <= 4:
                # Space group pattern (F-43m, P-1, f-43m)
                current_token += token
                current_indices.append(i)
                in_space_group = True
            elif current_token.isdigit():
                # N-coordinate pattern (12-coordinate)
                current_token += token
                current_indices.append(i)
                in_coordinate = True
            else:
                # Check if this is part of a hyphenated compound word (like "see-saw-like")
                # Merge hyphen if current token is alphabetic (or already hyphenated) and has reasonable length
                # This keeps compound words together while avoiding merging with formulas
                token_without_hyphens = current_token.replace('##', '').replace('-', '')
                is_compound_word = (token_without_hyphens.isalpha() and
                                   len(current_token) >= 2 and
                                   not (current_token in atom_symbols or current_token.capitalize() in atom_symbols))
                if is_compound_word:
                    # This looks like a hyphenated compound word
                    current_token += token
                    current_indices.append(i)
                    in_hyphenated_word = True  # Continue merging the next word
                else:
                    # Don't merge "-" with atom symbols or other patterns
                    if current_token:
                        merged_tokens.append(current_token)
                        token_mapping.append(current_indices)
                        in_space_group = False
                        in_coordinate = False
                        after_closing_paren = False
                    current_token = token
                    current_indices = [i]
        elif token == '/' and current_token:
            # Merge "/" for space groups (I4/mmm, P63/mmc)
            # Use upper() for case-insensitive detection
            if current_token[0].upper() in space_group_starters:
                current_token += token
                current_indices.append(i)
                in_space_group = True
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    after_closing_paren = False
                current_token = token
                current_indices = [i]
        elif in_coordinate and token.isalpha():
            # Continue N-coordinate pattern (12-coordinate)
            current_token += token
            current_indices.append(i)
            in_coordinate = False  # End after the word
        elif in_space_group:
            # Check if this token should be part of the space group
            # Continue only for digits and specific space group characters
            if token.isdigit() or (token.lower() in space_group_chars):
                # Continue space group: merge numbers, m/n/c/a/b/d letters
                current_token += token
                current_indices.append(i)
            else:
                # End space group - don't merge this token
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                in_space_group = False
                after_closing_paren = False
                current_token = token
                current_indices = [i]
        elif token == '(' and current_token:
            # Merge opening parentheses (e.g., for atom labels like Li(1))
            current_token += token
            current_indices.append(i)
            in_parentheses = True
        elif token == ')' and current_token and in_parentheses:
            # Merge closing parentheses
            current_token += token
            current_indices.append(i)
            in_parentheses = False
            # Set flag to continue merging if followed by element symbols/digits (like Ca(1)O6)
            after_closing_paren = True
        elif in_parentheses and (token.isdigit() or token.isalpha()):
            # Merge content inside parentheses (digits or letters)
            current_token += token
            current_indices.append(i)
        elif after_closing_paren and (token in atom_symbols or token.capitalize() in atom_symbols or token.isdigit()):
            # Continue merging after closing parenthesis for chemical formulas (Ca(1)O6)
            # Merge element symbols or digits immediately following ")"
            current_token += token
            current_indices.append(i)
            # Keep the flag set if it's an element symbol (to continue with digits like O6)
            # Reset the flag if it's a digit (we're done with this formula unit)
            if token.isdigit():
                after_closing_paren = False
        elif token.isdigit() and current_token and not current_token[-1].isdigit():
            # Only merge numbers with atom symbols or short space group starters
            # NOT with regular words like "bonded"
            # Check both original case and capitalized for atom symbols
            # For space group starters, only merge if token is short (<=2 chars, like P, I, F, Pm)
            is_atom = current_token in atom_symbols or current_token.capitalize() in atom_symbols
            is_short_space_group = (len(current_token) <= 2 and current_token[0].upper() in space_group_starters)
            if is_atom or is_short_space_group:
                current_token += token
                current_indices.append(i)
            else:
                # Save previous and start new with the digit
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    in_space_group = False
                    after_closing_paren = False
                current_token = token
                current_indices = [i]
        elif token == '.' and current_token:
            # Only merge decimal points if followed by digits (e.g., 3.14)
            # Don't merge period after space groups or between different entities
            # Check if the current token looks like it should accept a decimal
            # (e.g., a number, not a space group or atom label)
            is_numeric = current_token.replace('-', '').replace('/', '').isdigit()
            # Don't merge if current_token ends with ')' (like "Ca(1).")
            ends_with_paren = current_token.endswith(')')
            # Don't merge if current token is a space group (short and has special chars)
            is_space_group = len(current_token) <= 6 and any(c in current_token for c in ['-', '/'])

            if is_numeric and not is_space_group and not ends_with_paren:
                # This looks like a decimal number
                current_token += token
                current_indices.append(i)
            else:
                # Save previous token and start new with period
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    in_space_group = False
                    in_coordinate = False
                    in_parentheses = False
                    after_closing_paren = False
                current_token = token
                current_indices = [i]
        elif in_hyphenated_word and token.replace('##', '').isalpha():
            # Continue merging hyphenated compound word (like "saw" in "see-saw-like")
            current_token += token
            current_indices.append(i)
            # Reset flag after merging the word part
            # The next hyphen will be checked again to see if it should continue the compound
            in_hyphenated_word = False
        else:
            # Save previous token if exists (unless we're continuing a hyphenated word)
            if current_token and not in_hyphenated_word:
                merged_tokens.append(current_token)
                token_mapping.append(current_indices)
                in_space_group = False
                in_coordinate = False
                in_parentheses = False
                after_closing_paren = False
                # Start new token
                current_token = token
                current_indices = [i]
            elif current_token and in_hyphenated_word:
                # Reset hyphenated word flag if next token is not alphabetic or hyphen
                # This handles the end of the hyphenated word
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
            # Check if starting a space group (case-insensitive)
            if token.upper() in space_group_starters:
                in_space_group = True
            else:
                in_space_group = False

    # Don't forget the last token
    if current_token:
        merged_tokens.append(current_token)
        token_mapping.append(current_indices)

    # Merge weights by taking maximum over grouped indices
    # Using max preserves the strongest attention signal in merged tokens
    if weights is not None and len(weights.shape) >= 1:
        # Handle different weight shapes
        if len(weights.shape) == 1:
            # [seq_len]
            merged_weights = np.array([weights[indices].max() for indices in token_mapping])
        elif len(weights.shape) == 2:
            # [num_atoms, seq_len] or [seq_len, num_atoms]
            if weights.shape[-1] == len(tokens):
                # Last dim is seq_len
                merged_weights = np.array([[weights[i, indices].max() for indices in token_mapping]
                                           for i in range(weights.shape[0])])
            else:
                # First dim is seq_len
                merged_weights = np.array([[weights[indices, i].max() for indices in token_mapping]
                                           for i in range(weights.shape[1])]).T
        else:
            merged_weights = weights  # Don't merge for complex shapes
    else:
        merged_weights = weights

    return merged_tokens, merged_weights, token_mapping


def test_hyphenated_words():
    """Test that hyphenated compound words stay together"""
    print("\n" + "="*80)
    print("Test 1: Hyphenated Compound Words")
    print("="*80)

    # Test case: "see-saw-like"
    # BERT tokenizer might split this as: ['see', '-', 'saw', '-', 'like']
    test_cases = [
        {
            'name': 'see-saw-like',
            'tokens': ['see', '-', 'saw', '-', 'like'],
            'expected': 'see-saw-like'
        },
        {
            'name': 'edge-sharing',
            'tokens': ['edge', '-', 'sharing'],
            'expected': 'edge-sharing'
        },
        {
            'name': 'corner-sharing',
            'tokens': ['corner', '-', 'sharing'],
            'expected': 'corner-sharing'
        }
    ]

    for test in test_cases:
        tokens = test['tokens']
        expected = test['expected']
        weights = np.ones(len(tokens))  # Dummy weights

        merged_tokens, _, _ = _merge_tokens_and_weights(tokens, weights)

        # Check if the expected merged token appears in the result
        if expected in merged_tokens:
            print(f"✅ PASS: {test['name']}")
            print(f"   Input:  {tokens}")
            print(f"   Output: {merged_tokens}")
        else:
            print(f"❌ FAIL: {test['name']}")
            print(f"   Input:    {tokens}")
            print(f"   Output:   {merged_tokens}")
            print(f"   Expected: '{expected}' in output")


def test_chemical_formulas():
    """Test that chemical formulas with site numbers stay together"""
    print("\n" + "="*80)
    print("Test 2: Chemical Formulas with Site Numbers")
    print("="*80)

    # Test case: "Ca(1)O6" should stay as one token
    # BERT tokenizer might split this as: ['ca', '(', '1', ')', 'o', '6']
    test_cases = [
        {
            'name': 'Ca(1)O6',
            'tokens': ['ca', '(', '1', ')', 'o', '6'],
            'expected': 'ca(1)o6'
        },
        {
            'name': 'Ti(1)O6',
            'tokens': ['ti', '(', '1', ')', 'o', '6'],
            'expected': 'ti(1)o6'
        },
        {
            'name': 'Ba(1)O12',
            'tokens': ['ba', '(', '1', ')', 'o', '12'],
            'expected': 'ba(1)o12'
        }
    ]

    for test in test_cases:
        tokens = test['tokens']
        expected = test['expected']
        weights = np.ones(len(tokens))  # Dummy weights

        merged_tokens, _, _ = _merge_tokens_and_weights(tokens, weights)

        # Check if the expected merged token appears in the result
        if expected in merged_tokens:
            print(f"✅ PASS: {test['name']}")
            print(f"   Input:  {tokens}")
            print(f"   Output: {merged_tokens}")
        else:
            print(f"❌ FAIL: {test['name']}")
            print(f"   Input:    {tokens}")
            print(f"   Output:   {merged_tokens}")
            print(f"   Expected: '{expected}' in output")


def test_distinction():
    """Test that Ca(1) and Ca(1)O6 are properly distinguished in the same sentence"""
    print("\n" + "="*80)
    print("Test 3: Distinguish Ca(1) vs Ca(1)O6")
    print("="*80)

    # Simulate a sentence with both Ca(1) and Ca(1)O6
    # "Ca(1) bonded equivalent O(1) atoms form distorted Ca(1)O6 pentagonal pyramids"
    tokens = [
        'ca', '(', '1', ')',  # Ca(1)
        'bonded',
        'equivalent',
        'o', '(', '1', ')',  # O(1)
        'atoms',
        'form',
        'distorted',
        'ca', '(', '1', ')', 'o', '6',  # Ca(1)O6
        'pentagonal',
        'pyramids'
    ]

    weights = np.ones(len(tokens))

    merged_tokens, _, _ = _merge_tokens_and_weights(tokens, weights)

    print(f"Input tokens:")
    print(f"  {tokens}")
    print(f"\nMerged tokens:")
    print(f"  {merged_tokens}")

    # Check that both Ca(1) and Ca(1)O6 appear correctly
    # Count occurrences
    ca1_count = merged_tokens.count('ca(1)')
    ca1o6_count = merged_tokens.count('ca(1)o6')

    if ca1_count == 1 and ca1o6_count == 1:
        print(f"\n✅ PASS: Both 'ca(1)' and 'ca(1)o6' are correctly distinguished")
        print(f"   Found 1x 'ca(1)' and 1x 'ca(1)o6' (expected)")
    else:
        print(f"\n❌ FAIL: Distinction failed")
        print(f"   Found {ca1_count}x 'ca(1)' (expected 1)")
        print(f"   Found {ca1o6_count}x 'ca(1)o6' (expected 1)")


def test_real_example():
    """Test with the real example from the user"""
    print("\n" + "="*80)
    print("Test 4: Real Example from User")
    print("="*80)

    # Simulate the text with key phrases
    # "distorted see-saw-like geometry equivalent Ca(1) equivalent Ti(1) atoms"
    tokens = [
        'distorted',
        'see', '-', 'saw', '-', 'like',  # see-saw-like
        'geometry',
        'equivalent',
        'ca', '(', '1', ')',  # Ca(1)
        'equivalent',
        'ti', '(', '1', ')',  # Ti(1)
        'atoms'
    ]

    weights = np.ones(len(tokens))

    merged_tokens, _, _ = _merge_tokens_and_weights(tokens, weights)

    print(f"Input tokens:")
    print(f"  {tokens}")
    print(f"\nMerged tokens:")
    print(f"  {merged_tokens}")

    # Check that see-saw-like is merged
    has_see_saw_like = 'see-saw-like' in merged_tokens
    has_ca1 = 'ca(1)' in merged_tokens
    has_ti1 = 'ti(1)' in merged_tokens

    if has_see_saw_like and has_ca1 and has_ti1:
        print(f"\n✅ PASS: All patterns correctly merged")
    else:
        print(f"\n❌ FAIL: Some patterns not merged correctly")
        print(f"   Has 'see-saw-like': {has_see_saw_like}")
        print(f"   Has 'ca(1)': {has_ca1}")
        print(f"   Has 'ti(1)': {has_ti1}")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Token Merging Test Suite")
    print("="*80)

    try:
        test_hyphenated_words()
        test_chemical_formulas()
        test_distinction()
        test_real_example()

        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
