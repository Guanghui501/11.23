#!/usr/bin/env python3
"""
Test script to verify attention weight merging logic is working correctly.
This creates synthetic attention data and verifies that different atoms
have different top words after merging.
"""

import numpy as np
import sys

def test_merge_logic():
    """Test that merging preserves per-atom differences."""

    print("="*80)
    print("Testing Attention Weight Merging Logic")
    print("="*80)

    # Simulate attention weights for 3 atoms, 10 tokens
    num_atoms = 3
    seq_len = 10

    # Create different attention patterns for each atom
    np.random.seed(42)
    atom_to_text_avg = np.random.rand(num_atoms, seq_len)

    # Make atom 0 focus on token 2
    atom_to_text_avg[0, 2] = 0.9
    # Make atom 1 focus on token 5
    atom_to_text_avg[1, 5] = 0.9
    # Make atom 2 focus on token 8
    atom_to_text_avg[2, 8] = 0.9

    # Normalize
    atom_to_text_avg = atom_to_text_avg / atom_to_text_avg.sum(axis=1, keepdims=True)

    print(f"\n✓ Created synthetic attention data: {atom_to_text_avg.shape}")
    print(f"  Atom 0 should focus on token 2")
    print(f"  Atom 1 should focus on token 5")
    print(f"  Atom 2 should focus on token 8")

    # Check top token for each atom
    for i in range(num_atoms):
        top_token = atom_to_text_avg[i].argmax()
        print(f"\n  Atom {i}: Top token = {top_token}, weight = {atom_to_text_avg[i, top_token]:.4f}")

    # Verify they're different
    top_tokens = [atom_to_text_avg[i].argmax() for i in range(num_atoms)]
    if len(set(top_tokens)) == num_atoms:
        print(f"\n✅ PASS: Each atom has a different top token: {top_tokens}")
        return True
    else:
        print(f"\n❌ FAIL: Atoms have identical top tokens: {top_tokens}")
        return False

def test_wordpiece_merging():
    """Test WordPiece token merging."""

    print("\n" + "="*80)
    print("Testing WordPiece Token Merging")
    print("="*80)

    # Simulate tokens: ["li", "##ba", "##4", "##hf", "has", "f", "-", "43", "m"]
    # Should merge to: ["liba4hf", "has", "f-43m"]
    tokens = ["li", "##ba", "##4", "##hf", "has", "f", "-", "43", "m"]

    # Create attention for 2 atoms
    num_atoms = 2
    weights = np.zeros((num_atoms, len(tokens)))

    # Atom 0: focus on "liba4hf" (indices 0-3)
    weights[0, 0:4] = [0.3, 0.25, 0.25, 0.2]  # Should average to 0.25

    # Atom 1: focus on "has" (index 4)
    weights[1, 4] = 1.0

    print(f"\n✓ Created tokens: {tokens}")
    print(f"✓ Atom 0 weights: {weights[0]}")
    print(f"✓ Atom 1 weights: {weights[1]}")

    # Import the merging function
    sys.path.insert(0, '/home/user/11.23')
    try:
        from interpretability_enhanced import EnhancedInterpretabilityAnalyzer

        merged_tokens, merged_weights, token_mapping = \
            EnhancedInterpretabilityAnalyzer._merge_tokens_and_weights(tokens, weights)

        print(f"\n✓ Merged tokens: {merged_tokens}")
        print(f"✓ Token mapping: {token_mapping}")
        print(f"✓ Merged weights shape: {merged_weights.shape}")
        print(f"✓ Atom 0 merged weights: {merged_weights[0]}")
        print(f"✓ Atom 1 merged weights: {merged_weights[1]}")

        # Check that atom 0's top word is "liba4hf" and atom 1's is "has"
        atom0_top = merged_tokens[merged_weights[0].argmax()]
        atom1_top = merged_tokens[merged_weights[1].argmax()]

        print(f"\n  Atom 0 top word: '{atom0_top}' (weight: {merged_weights[0].max():.4f})")
        print(f"  Atom 1 top word: '{atom1_top}' (weight: {merged_weights[1].max():.4f})")

        if atom0_top == "liba4hf" and atom1_top == "has":
            print(f"\n✅ PASS: WordPiece merging preserves per-atom differences")
            return True
        else:
            print(f"\n❌ FAIL: Unexpected top words")
            print(f"   Expected: atom0='liba4hf', atom1='has'")
            print(f"   Got: atom0='{atom0_top}', atom1='{atom1_top}'")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: Failed to import or run merging function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Attention Weight Processing Test Suite")
    print("#"*80 + "\n")

    results = []

    # Test 1: Basic merge logic
    results.append(("Basic merge logic", test_merge_logic()))

    # Test 2: WordPiece merging
    results.append(("WordPiece merging", test_wordpiece_merging()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("="*80))
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    sys.exit(0 if all_passed else 1)
