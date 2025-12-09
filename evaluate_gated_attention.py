#!/usr/bin/env python3
"""
Evaluate ALIGNN with Gated Cross-Attention on text masking robustness.

This script compares:
1. Original ALIGNN (baseline)
2. ALIGNN with Gated Cross-Attention (improved)

Focus on addressing the 100% masking collapse issue:
- Baseline: MAE = 1.93 at 100% masking
- Expected with Gated Attention: MAE ≈ 0.80
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import the new gated attention model
from models.alignn_with_gated_attention import (
    ALIGNNWithGatedAttention,
    create_gated_alignn
)

# Import text masking utilities
from evaluate_text_masking import (
    TextMasker,
    compute_metrics
)


class GatedAttentionDataset(Dataset):
    """Dataset for evaluating gated attention with text masking."""

    def __init__(self, data_path: str, masker: TextMasker = None, masking_ratio: float = 0.0):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.masker = masker
        self.masking_ratio = masking_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply text masking if masker is provided
        if self.masker is not None and self.masking_ratio > 0:
            original_text = sample.get('text', '')
            masked_text = self.masker.mask_text(original_text, self.masking_ratio)
            sample = {**sample, 'text': masked_text}

        return sample


def collate_fn_gated(batch):
    """Collate function for batching graph and text data."""
    import dgl

    graphs = []
    line_graphs = []
    text_list = []
    targets = []

    for sample in batch:
        # Graph data
        g = sample.get('graph')
        lg = sample.get('line_graph')

        if isinstance(g, tuple):
            g = g[0]
        if isinstance(lg, tuple):
            lg = lg[0]

        graphs.append(g)
        line_graphs.append(lg)

        # Text data
        text_list.append(sample.get('text', ''))

        # Target
        target = sample.get('target', 0.0)
        if isinstance(target, (list, np.ndarray)):
            target = target[0]
        targets.append(float(target))

    # Batch graphs
    batched_graph = dgl.batch(graphs)
    batched_line_graph = dgl.batch(line_graphs)

    # Targets to tensor
    targets_tensor = torch.FloatTensor(targets)

    return batched_graph, batched_line_graph, text_list, targets_tensor


def encode_text(text_list: List[str], text_encoder, tokenizer, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text using MatSciBERT.

    Returns:
        text_features: [batch, seq_len, hidden_dim]
        text_mask: [batch, seq_len] - True for padding tokens
    """
    # Tokenize
    encoded = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Encode with MatSciBERT
    with torch.no_grad():
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state  # [batch, seq_len, 768]

    # Create padding mask (True for padding)
    text_mask = (attention_mask == 0)

    return text_features, text_mask


def evaluate_with_gated_attention(
    model: ALIGNNWithGatedAttention,
    text_encoder,
    tokenizer,
    data_loader: DataLoader,
    device: torch.device,
    return_diagnostics: bool = False
) -> Dict:
    """
    Evaluate model with gated attention.

    Returns:
        results: Dict with metrics and optional diagnostics
    """
    model.eval()
    text_encoder.eval()

    all_predictions = []
    all_targets = []
    all_diagnostics = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Evaluating")):
            g, lg, text_list, targets = batch_data

            g = g.to(device)
            lg = lg.to(device)
            targets = targets.to(device)

            # Encode text
            text_features, text_mask = encode_text(text_list, text_encoder, tokenizer, device)

            # Forward pass
            batch_input = [g, lg, text_features, text_mask]

            if return_diagnostics:
                predictions, diagnostics = model(batch_input, return_attention=True)
                all_diagnostics.append(diagnostics)
            else:
                predictions = model(batch_input, return_attention=False)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    if return_diagnostics:
        # Aggregate diagnostics
        avg_text_quality = np.mean([d['quality_mean'] for d in all_diagnostics])
        avg_text_influence = np.mean([d['text_influence'] for d in all_diagnostics])

        metrics['diagnostics'] = {
            'avg_text_quality': float(avg_text_quality),
            'avg_text_influence': float(avg_text_influence),
        }

    return metrics


def run_masking_evaluation(
    model_path: str,
    text_encoder_path: str,
    test_data_path: str,
    output_dir: str,
    masking_strategies: List[str],
    masking_ratios: List[float],
    device: str = 'cuda',
    batch_size: int = 32,
):
    """
    Run comprehensive masking evaluation with gated attention.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load text encoder (MatSciBERT)
    print("Loading text encoder...")
    from transformers import AutoModel, AutoTokenizer

    text_encoder = AutoModel.from_pretrained(text_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Load model with gated attention
    print("Loading model with gated attention...")
    model = create_gated_alignn(
        checkpoint_path=model_path,
        hidden_dim=256,
        text_hidden_dim=768,
        use_gated_attention=True,
        gated_attention_layers=1,
        attention_heads=8,
        output_dim=1
    )
    model = model.to(device)
    model.eval()

    # Results storage
    all_results = {
        strategy: {
            'masking_ratios': [],
            'mae': [],
            'rmse': [],
            'r2': [],
            'text_quality': [],
            'text_influence': []
        }
        for strategy in masking_strategies
    }

    # Evaluate each strategy
    for strategy in masking_strategies:
        print(f"\n{'='*80}")
        print(f"Evaluating strategy: {strategy}")
        print('='*80)

        for ratio in masking_ratios:
            print(f"\n  Masking ratio: {ratio*100:.0f}%")

            # Create masker
            masker = TextMasker(tokenizer, strategy=strategy)

            # Create dataset and dataloader
            dataset = GatedAttentionDataset(
                test_data_path,
                masker=masker,
                masking_ratio=ratio
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_gated,
                num_workers=4
            )

            # Evaluate
            results = evaluate_with_gated_attention(
                model, text_encoder, tokenizer, dataloader, device,
                return_diagnostics=True
            )

            # Store results
            all_results[strategy]['masking_ratios'].append(ratio * 100)
            all_results[strategy]['mae'].append(results['mae'])
            all_results[strategy]['rmse'].append(results['rmse'])
            all_results[strategy]['r2'].append(results['r2'])
            all_results[strategy]['text_quality'].append(
                results['diagnostics']['avg_text_quality']
            )
            all_results[strategy]['text_influence'].append(
                results['diagnostics']['avg_text_influence']
            )

            print(f"    MAE: {results['mae']:.4f}")
            print(f"    RMSE: {results['rmse']:.4f}")
            print(f"    R²: {results['r2']:.4f}")
            print(f"    Text Quality: {results['diagnostics']['avg_text_quality']:.4f}")
            print(f"    Text Influence: {results['diagnostics']['avg_text_influence']:.4f}")

    # Save results
    output_file = os.path.join(output_dir, 'gated_attention_masking_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ALIGNN with Gated Cross-Attention on text masking robustness'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--text_encoder',
        type=str,
        default='m3rg-iitd/matscibert',
        help='MatSciBERT model path or name'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test data (pickle file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gated_attention_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--masking_strategies',
        nargs='+',
        default=['random_token', 'sentence', 'keep_keywords'],
        choices=['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords'],
        help='Masking strategies to evaluate'
    )
    parser.add_argument(
        '--masking_ratios',
        nargs='+',
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help='Masking ratios to test'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Run evaluation
    results = run_masking_evaluation(
        model_path=args.model_path,
        text_encoder_path=args.text_encoder,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        masking_strategies=args.masking_strategies,
        masking_ratios=args.masking_ratios,
        device=args.device,
        batch_size=args.batch_size,
    )

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
