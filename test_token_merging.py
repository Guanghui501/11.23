#!/usr/bin/env python3
"""
Test script to verify token merging improvements for:
1. Hyphenated compound words (see-saw-like)
2. Chemical formulas with site numbers (Ca(1)O6)
"""

import sys
import numpy as np

# Import the merge function from interpretability_enhanced
from interpretability_enhanced import EnhancedInterpretabilityAnalyzer


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

        merged_tokens, _, _ = EnhancedInterpretabilityAnalyzer._merge_tokens_and_weights(
            tokens, weights
        )

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

        merged_tokens, _, _ = EnhancedInterpretabilityAnalyzer._merge_tokens_and_weights(
            tokens, weights
        )

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

    merged_tokens, _, _ = EnhancedInterpretabilityAnalyzer._merge_tokens_and_weights(
        tokens, weights
    )

    print(f"Input tokens:")
    print(f"  {tokens}")
    print(f"\nMerged tokens:")
    print(f"  {merged_tokens}")

    # Check that both Ca(1) and Ca(1)O6 appear correctly
    has_ca1 = 'ca(1)' in merged_tokens
    has_ca1o6 = 'ca(1)o6' in merged_tokens

    if has_ca1 and has_ca1o6:
        print(f"\n✅ PASS: Both 'ca(1)' and 'ca(1)o6' are correctly distinguished")
    else:
        print(f"\n❌ FAIL: Distinction failed")
        print(f"   Has 'ca(1)': {has_ca1}")
        print(f"   Has 'ca(1)o6': {has_ca1o6}")


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

    merged_tokens, _, _ = EnhancedInterpretabilityAnalyzer._merge_tokens_and_weights(
        tokens, weights
    )

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
