#!/usr/bin/env python3
"""
Generate datasets with text DELETION (not masking)

Key difference from masking:
- Masking: Replace tokens with [MASK] → "The [MASK] is [MASK]"
- Deletion: Remove tokens completely → "The is" or ""

This allows testing true graph-only performance when text is 100% deleted.
"""

import os
import sys
import pickle
import random
import re
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeterministicTextDeleter:
    """
    Deterministically delete text tokens (not replace with [MASK])

    Ensures the same sample_id always gets the same deletion pattern.
    """

    def __init__(self, tokenizer, strategy='random_token', base_seed=42):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            strategy: Deletion strategy
                - 'random_token': Randomly delete individual tokens
                - 'random_word': Randomly delete complete words
                - 'random_chunk': Delete continuous chunks
                - 'sentence': Delete complete sentences
                - 'keep_keywords': Keep only chemical elements and structure terms
            base_seed: Base random seed
        """
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.base_seed = base_seed

        # Patterns for keyword detection
        self.element_pattern = re.compile(r'\b([A-Z][a-z]?)\b')
        self.structure_keywords = [
            'cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'monoclinic',
            'triclinic', 'rhombohedral', 'bcc', 'fcc', 'hcp', 'diamond',
            'structure', 'lattice', 'crystal', 'symmetry', 'space', 'group'
        ]

    def delete_text_deterministic(self, text: str, deletion_ratio: float, sample_id: str) -> str:
        """
        Deterministically delete text based on sample_id

        Args:
            text: Original text
            deletion_ratio: Fraction of text to delete (0.0 to 1.0)
            sample_id: Unique sample identifier

        Returns:
            Text with deleted tokens (may be empty string at ratio=1.0)
        """
        if deletion_ratio == 0.0:
            return text

        if deletion_ratio >= 1.0:
            # 100% deletion → empty string
            return ""

        # Generate deterministic seed from sample_id
        seed = self.base_seed + hash(sample_id) % 1000000

        # Save current random state
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        try:
            # Set deterministic seed
            random.seed(seed)
            np.random.seed(seed % (2**32))

            # Apply deletion strategy
            if self.strategy == 'random_token':
                deleted = self._delete_random_tokens(text, deletion_ratio)
            elif self.strategy == 'random_word':
                deleted = self._delete_random_words(text, deletion_ratio)
            elif self.strategy == 'random_chunk':
                deleted = self._delete_random_chunks(text, deletion_ratio)
            elif self.strategy == 'sentence':
                deleted = self._delete_sentences(text, deletion_ratio)
            elif self.strategy == 'keep_keywords':
                deleted = self._keep_only_keywords(text, deletion_ratio)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        finally:
            # Restore original random state
            random.setstate(random_state)
            np.random.set_state(np_random_state)

        return deleted

    def _delete_random_tokens(self, text: str, ratio: float) -> str:
        """Delete random individual tokens"""
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            return text

        num_to_delete = int(len(tokens) * ratio)
        if num_to_delete == 0:
            return text

        if num_to_delete >= len(tokens):
            return ""

        # Select tokens to DELETE
        delete_indices = set(random.sample(range(len(tokens)), num_to_delete))

        # Keep only non-deleted tokens
        kept_tokens = [token for i, token in enumerate(tokens) if i not in delete_indices]

        if len(kept_tokens) == 0:
            return ""

        deleted_text = self.tokenizer.convert_tokens_to_string(kept_tokens)
        return deleted_text.strip()

    def _delete_random_words(self, text: str, ratio: float) -> str:
        """Delete random complete words"""
        words = text.split()
        if len(words) == 0:
            return text

        num_to_delete = int(len(words) * ratio)
        if num_to_delete == 0:
            return text

        if num_to_delete >= len(words):
            return ""

        # Select words to DELETE
        delete_indices = set(random.sample(range(len(words)), num_to_delete))

        # Keep only non-deleted words
        kept_words = [word for i, word in enumerate(words) if i not in delete_indices]

        if len(kept_words) == 0:
            return ""

        return ' '.join(kept_words).strip()

    def _delete_random_chunks(self, text: str, ratio: float) -> str:
        """Delete continuous text chunks"""
        words = text.split()
        if len(words) == 0:
            return text

        num_to_delete = int(len(words) * ratio)
        if num_to_delete == 0:
            return text

        if num_to_delete >= len(words):
            return ""

        # Randomly select starting position for deletion
        start_idx = random.randint(0, len(words) - num_to_delete)

        # Keep words before and after the deleted chunk
        kept_words = words[:start_idx] + words[start_idx + num_to_delete:]

        if len(kept_words) == 0:
            return ""

        return ' '.join(kept_words).strip()

    def _delete_sentences(self, text: str, ratio: float) -> str:
        """Delete complete sentences"""
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return text

        num_to_delete = max(1, int(len(sentences) * ratio))

        if num_to_delete >= len(sentences):
            return ""

        # Select sentences to DELETE
        delete_indices = set(random.sample(range(len(sentences)), min(num_to_delete, len(sentences))))

        # Keep non-deleted sentences
        kept_sentences = [s for i, s in enumerate(sentences) if i not in delete_indices]

        if len(kept_sentences) == 0:
            return ""

        return '. '.join(kept_sentences).strip()

    def _keep_only_keywords(self, text: str, ratio: float) -> str:
        """Keep only chemical elements and structure keywords, delete others"""
        words = text.split()
        if len(words) == 0:
            return text

        # Find all keyword positions
        keyword_indices = []
        for i, word in enumerate(words):
            if self.element_pattern.search(word) or \
               any(kw in word.lower() for kw in self.structure_keywords):
                keyword_indices.append(i)

        if len(keyword_indices) == 0:
            # No keywords, delete everything based on ratio
            num_to_keep = max(0, int(len(words) * (1.0 - ratio)))
            if num_to_keep == 0:
                return ""
            keep_indices = set(random.sample(range(len(words)), num_to_keep))
            kept_words = [word for i, word in enumerate(words) if i in keep_indices]
            return ' '.join(kept_words).strip() if kept_words else ""

        # Determine how many keywords to keep
        num_keywords_to_keep = max(1, int(len(keyword_indices) * (1.0 - ratio)))

        if num_keywords_to_keep == 0:
            return ""

        keep_indices = set(random.sample(keyword_indices, num_keywords_to_keep))

        # Keep only selected keywords
        kept_words = [word for i, word in enumerate(words) if i in keep_indices]

        return ' '.join(kept_words).strip() if kept_words else ""


def generate_deletion_datasets(
    input_data_path: str,
    output_dir: str,
    strategies: List[str],
    ratios: List[float],
    tokenizer_name: str = 'm3rg-iitd/matscibert',
    seed: int = 42
):
    """
    Generate all deletion strategy and deletion ratio datasets

    Args:
        input_data_path: Input data path (pickle file)
        output_dir: Output directory
        strategies: List of deletion strategies
        ratios: List of deletion ratios
        tokenizer_name: Tokenizer name
        seed: Random seed
    """
    # Set global seed
    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load data
    print(f"Loading data from: {input_data_path}")
    with open(input_data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} samples")

    # Generate datasets for each strategy and ratio
    total_combinations = len(strategies) * len(ratios)
    pbar = tqdm(total=total_combinations, desc="Generating deletion datasets")

    # Store metadata
    metadata = {
        'method': 'deletion',  # NOT masking
        'strategies': strategies,
        'ratios': ratios,
        'num_samples': len(data),
        'seed': seed,
        'tokenizer': tokenizer_name
    }

    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy}")
        print('='*80)

        # Create deleter
        deleter = DeterministicTextDeleter(tokenizer, strategy=strategy, base_seed=seed)

        for ratio in ratios:
            # Create deletion dataset
            deleted_data = []

            # Statistics
            num_empty = 0
            total_length_original = 0
            total_length_deleted = 0

            for sample in data:
                # Get sample ID
                sample_id = str(sample.get('id', ''))

                # Get original text
                original_text = sample.get('text', '')

                # Deterministic deletion
                deleted_text = deleter.delete_text_deterministic(original_text, ratio, sample_id)

                # Create new sample (keep all fields, only replace text)
                deleted_sample = sample.copy()
                deleted_sample['text'] = deleted_text
                deleted_sample['original_text'] = original_text
                deleted_sample['deletion_ratio'] = ratio
                deleted_sample['deletion_strategy'] = strategy

                deleted_data.append(deleted_sample)

                # Update statistics
                if deleted_text == "":
                    num_empty += 1
                total_length_original += len(original_text)
                total_length_deleted += len(deleted_text)

            # Save dataset
            output_file = os.path.join(output_dir, f"{strategy}_{ratio:.1f}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(deleted_data, f)

            avg_original = total_length_original / len(data) if len(data) > 0 else 0
            avg_deleted = total_length_deleted / len(data) if len(data) > 0 else 0
            actual_deletion_rate = 1.0 - (avg_deleted / avg_original) if avg_original > 0 else 1.0

            print(f"  Ratio {ratio:.1f}: Saved {len(deleted_data)} samples to {output_file}")
            print(f"    Empty texts: {num_empty}/{len(data)} ({100*num_empty/len(data):.1f}%)")
            print(f"    Avg original length: {avg_original:.1f} chars")
            print(f"    Avg deleted length:  {avg_deleted:.1f} chars")
            print(f"    Actual deletion rate: {100*actual_deletion_rate:.1f}%")

            pbar.update(1)

    pbar.close()

    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n{'='*80}")
    print("Generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total datasets: {total_combinations}")
    print(f"Metadata saved to: {metadata_file}")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate datasets with deterministic text DELETION (not masking)'
    )
    parser.add_argument(
        '--input_data',
        type=str,
        required=True,
        help='Input data path (pickle file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./deletion_datasets',
        help='Output directory (default: ./deletion_datasets)'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['random_token', 'random_word', 'keep_keywords'],
        help='Deletion strategies (default: random_token random_word keep_keywords)'
    )
    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='Deletion ratios (default: 0.0 0.2 0.4 0.6 0.8 1.0)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='m3rg-iitd/matscibert',
        help='Tokenizer name (default: m3rg-iitd/matscibert)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  Text Deletion Dataset Generator (NOT Masking!)                         ║
║                                                                           ║
║  Key difference:                                                         ║
║    - Masking: Replace tokens with [MASK] → "[MASK] [MASK] [MASK]"      ║
║    - Deletion: Remove tokens completely → "" (empty string)             ║
║                                                                           ║
║  This allows testing TRUE graph-only performance at 100% deletion!      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    generate_deletion_datasets(
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        strategies=args.strategies,
        ratios=args.ratios,
        tokenizer_name=args.tokenizer,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
