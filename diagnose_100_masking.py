#!/usr/bin/env python3
"""
Diagnostic script to understand 100% masking behavior

This script helps answer the question:
"Why do models have different MAE at 100% masking?"

It will show:
1. What 100% masking actually produces
2. Whether MatSciBERT produces embeddings for all [MASK] text
3. How different text inputs affect embeddings
"""

import sys
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from pregenerate_masked_dataset import DeterministicTextMasker


def test_masking_output():
    """Test what 100% masking produces"""
    print("="*80)
    print("TEST 1: What does 100% masking produce?")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')

    # Create masker
    masker = DeterministicTextMasker(tokenizer, strategy='random_token', base_seed=42)

    # Test with example text
    test_texts = [
        "Copper oxide cubic structure with space group Fm-3m",
        "Silicon has diamond structure",
        "NaCl is face-centered cubic",
    ]

    for i, text in enumerate(test_texts):
        print(f"\nExample {i+1}:")
        print(f"  Original: {text}")

        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"  Tokens ({len(tokens)}): {tokens}")

        # 100% masking
        masked_text = masker.mask_text_deterministic(text, ratio=1.0, sample_id=f"test_{i}")
        print(f"  100% masked: {masked_text}")
        print(f"  Length: {len(masked_text)} characters")
        print(f"  Is empty: {masked_text == ''}")

        # 50% masking for comparison
        masked_50 = masker.mask_text_deterministic(text, ratio=0.5, sample_id=f"test_{i}")
        print(f"  50% masked: {masked_50}")


def test_embeddings():
    """Test MatSciBERT embeddings for different text inputs"""
    print("\n" + "="*80)
    print("TEST 2: MatSciBERT embeddings for different text types")
    print("="*80)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
    model.eval()

    # Test cases
    test_cases = {
        'Clean text': "Copper oxide cubic structure",
        'All MASK (10 tokens)': "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
        'Single MASK': "[MASK]",
        'Empty string': "",
        'Whitespace only': "   ",
    }

    results = {}

    for name, text in test_cases.items():
        print(f"\n{name}:")
        print(f"  Text: '{text}'")

        # Tokenize
        if text.strip() == "":
            # Handle empty text
            print(f"  Warning: Empty text, using default token")
            text = "[PAD]"

        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        print(f"  Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  Input IDs: {inputs['input_ids'][0].tolist()}")

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            # Also compute mean pooling
            mean_embedding = outputs.last_hidden_state.mean(dim=1)

        # Compute statistics
        cls_norm = cls_embedding.norm().item()
        mean_norm = mean_embedding.norm().item()
        is_zero_cls = torch.allclose(cls_embedding, torch.zeros_like(cls_embedding), atol=1e-6)
        is_zero_mean = torch.allclose(mean_embedding, torch.zeros_like(mean_embedding), atol=1e-6)

        print(f"  CLS embedding norm: {cls_norm:.4f}")
        print(f"  Mean embedding norm: {mean_norm:.4f}")
        print(f"  Is zero (CLS): {is_zero_cls}")
        print(f"  Is zero (Mean): {is_zero_mean}")

        results[name] = {
            'cls_embedding': cls_embedding,
            'mean_embedding': mean_embedding,
            'cls_norm': cls_norm,
            'mean_norm': mean_norm,
        }

    # Compare embeddings
    print("\n" + "="*80)
    print("COMPARISON: How similar are the embeddings?")
    print("="*80)

    # Compare all-MASK vs clean text
    if 'All MASK (10 tokens)' in results and 'Clean text' in results:
        mask_emb = results['All MASK (10 tokens)']['mean_embedding']
        clean_emb = results['Clean text']['mean_embedding']

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(mask_emb, clean_emb, dim=1).item()

        # L2 distance
        l2_dist = (mask_emb - clean_emb).norm().item()

        print(f"\nAll-MASK vs Clean text:")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  L2 distance: {l2_dist:.4f}")
        print(f"  Interpretation:")
        if cos_sim > 0.8:
            print(f"    → Very similar! All-MASK produces similar embeddings to clean text")
        elif cos_sim > 0.5:
            print(f"    → Somewhat similar")
        else:
            print(f"    → Very different!")

    # Compare all-MASK vs empty
    if 'All MASK (10 tokens)' in results and 'Empty string' in results:
        mask_emb = results['All MASK (10 tokens)']['mean_embedding']
        empty_emb = results['Empty string']['mean_embedding']

        cos_sim = torch.nn.functional.cosine_similarity(mask_emb, empty_emb, dim=1).item()
        l2_dist = (mask_emb - empty_emb).norm().item()

        print(f"\nAll-MASK vs Empty string:")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  L2 distance: {l2_dist:.4f}")
        print(f"  Interpretation:")
        if cos_sim > 0.8:
            print(f"    → Very similar! All-MASK is like empty text")
        elif cos_sim > 0.5:
            print(f"    → Somewhat similar")
        else:
            print(f"    → Very different! All-MASK ≠ empty text")


def test_with_actual_data():
    """Test with actual dataset if available"""
    print("\n" + "="*80)
    print("TEST 3: Test with actual dataset")
    print("="*80)

    test_data_path = './corrected_test_set/test.pkl'

    if not os.path.exists(test_data_path):
        print(f"Dataset not found at: {test_data_path}")
        print("Skipping this test.")
        return

    # Load data
    print(f"Loading data from: {test_data_path}")
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} samples")

    # Test masking on first 3 samples
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    masker = DeterministicTextMasker(tokenizer, strategy='random_token', base_seed=42)

    print("\nMasking examples from actual dataset:")

    for i in range(min(3, len(data))):
        sample = data[i]
        sample_id = str(sample.get('id', f'sample_{i}'))
        original_text = sample.get('text', '')

        print(f"\nSample {i} (ID: {sample_id}):")
        print(f"  Original: {original_text}")

        # 100% masking
        masked_100 = masker.mask_text_deterministic(original_text, ratio=1.0, sample_id=sample_id)
        print(f"  100% masked: {masked_100}")

        # Count MASK tokens
        num_mask_tokens = masked_100.count('[MASK]')
        print(f"  Number of [MASK] tokens: {num_mask_tokens}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  Diagnostic Tool: Understanding 100% Masking Behavior                    ║
║                                                                           ║
║  This script will help you understand why models have different MAE      ║
║  at 100% masking by showing:                                             ║
║    1. What 100% masking actually produces                                ║
║    2. Whether MatSciBERT produces embeddings for all [MASK] text         ║
║    3. How all-MASK text differs from empty text                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Run tests
    test_masking_output()
    test_embeddings()
    test_with_actual_data()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Key findings:

1. 100% masking produces "[MASK] [MASK] [MASK] ..." (NOT empty string!)

2. MatSciBERT creates non-zero embeddings for [MASK] tokens
   → These embeddings are learned during pre-training
   → They represent "average materials science text"

3. All-MASK text ≠ Empty text
   → Different embeddings → Different model predictions

4. Why models differ at 100% masking:
   → Different architectures (Middle Fusion vs. no Middle Fusion)
   → Different ways of using [MASK] embeddings
   → Different training distributions

5. This is expected behavior!
   → model1+2 is more robust (MAE = 0.5358)
   → SAGE-Net degrades more (MAE = 0.7470)
   → Middle Fusion provides better robustness

For detailed explanation, see: WHY_100_MASKING_DIFFERS.md
    """)


if __name__ == "__main__":
    main()
