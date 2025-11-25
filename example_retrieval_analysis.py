"""
Example script for advanced retrieval analysis.

This script demonstrates how to use the RetrievalEvaluator class programmatically
for custom analysis beyond the basic command-line interface.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from evaluate_retrieval import RetrievalEvaluator, load_model_and_config
from config import TrainingConfig
from data import get_train_val_loaders


def example_1_basic_evaluation():
    """Example 1: Basic programmatic evaluation."""
    print("\n" + "="*80)
    print("Example 1: Basic Programmatic Evaluation")
    print("="*80)

    # Load model
    model_path = "outputs/best_val_model.pt"
    model, config = load_model_and_config(model_path, device='cuda')

    # Load data
    _, val_loader, test_loader, _ = get_train_val_loaders(
        dataset=config.dataset,
        target=config.target,
        batch_size=32,
        line_graph=True,
    )

    # Initialize evaluator
    evaluator = RetrievalEvaluator(model, device='cuda')

    # Extract features
    features = evaluator.extract_features(test_loader, modality='both')

    # Compute similarity
    similarity = evaluator.compute_similarity_matrix(
        features['graph_features'],
        features['text_features']
    )

    # Compute metrics
    metrics = evaluator.compute_retrieval_metrics(similarity)

    # Print results
    print(f"\nR@1:  {metrics['Avg_R@1']:.2f}%")
    print(f"R@5:  {metrics['Avg_R@5']:.2f}%")
    print(f"R@10: {metrics['Avg_R@10']:.2f}%")
    print(f"MRR:  {metrics['Avg_MRR']:.2f}%")


def example_2_error_analysis():
    """Example 2: Analyze retrieval failures."""
    print("\n" + "="*80)
    print("Example 2: Error Analysis")
    print("="*80)

    # Load retrieval results
    results_path = "outputs/retrieval_results/graph_to_text_retrieval.json"
    with open(results_path) as f:
        results = json.load(f)

    # Find failed cases
    failed_cases = [r for r in results if r['correct_rank'] > 1]
    success_cases = [r for r in results if r['correct_rank'] == 1]

    print(f"\nTotal samples: {len(results)}")
    print(f"Success (R@1): {len(success_cases)} ({len(success_cases)/len(results)*100:.1f}%)")
    print(f"Failed (R>1):  {len(failed_cases)} ({len(failed_cases)/len(results)*100:.1f}%)")

    # Analyze failure ranks
    if failed_cases:
        ranks = [r['correct_rank'] for r in failed_cases]
        print(f"\nFailure statistics:")
        print(f"  Mean rank: {np.mean(ranks):.2f}")
        print(f"  Median rank: {np.median(ranks):.0f}")
        print(f"  Max rank: {max(ranks)}")

        # Show worst cases
        failed_cases.sort(key=lambda x: x['correct_rank'], reverse=True)
        print(f"\nTop 5 worst failures:")
        for i, case in enumerate(failed_cases[:5], 1):
            print(f"\n  {i}. ID: {case['graph_id']}")
            print(f"     Correct rank: {case['correct_rank']}")
            print(f"     Target value: {case['graph_target']:.4f}")
            print(f"     Top-3 retrieved IDs: {case['top_k_ids'][:3]}")
            print(f"     Top-3 similarities: {[f'{s:.3f}' for s in case['top_k_similarities'][:3]]}")


def example_3_target_stratified_analysis():
    """Example 3: Analyze retrieval performance by target value ranges."""
    print("\n" + "="*80)
    print("Example 3: Target-Stratified Analysis")
    print("="*80)

    # Load retrieval results
    results_path = "outputs/retrieval_results/graph_to_text_retrieval.json"
    with open(results_path) as f:
        results = json.load(f)

    # Get target values
    targets = [r['graph_target'] for r in results]
    ranks = [r['correct_rank'] for r in results]

    # Define target bins
    percentiles = [0, 25, 50, 75, 100]
    bins = np.percentile(targets, percentiles)

    print(f"\nTarget value range: [{min(targets):.4f}, {max(targets):.4f}]")
    print(f"\nPerformance by target value quartile:")
    print(f"{'Quartile':<12} {'Range':<25} {'R@1':<10} {'MRR':<10}")
    print("-" * 60)

    for i in range(len(bins)-1):
        # Find samples in this bin
        mask = (np.array(targets) >= bins[i]) & (np.array(targets) < bins[i+1])
        if i == len(bins)-2:  # Last bin includes upper bound
            mask = (np.array(targets) >= bins[i]) & (np.array(targets) <= bins[i+1])

        bin_ranks = np.array(ranks)[mask]

        if len(bin_ranks) > 0:
            r_at_1 = np.mean(bin_ranks == 1) * 100
            mrr = np.mean(1.0 / bin_ranks) * 100
            range_str = f"[{bins[i]:.4f}, {bins[i+1]:.4f}]"
            quartile = f"Q{i+1} ({len(bin_ranks)})"

            print(f"{quartile:<12} {range_str:<25} {r_at_1:>6.2f}%   {mrr:>6.2f}%")


def example_4_similarity_distribution_analysis():
    """Example 4: Analyze similarity score distributions."""
    print("\n" + "="*80)
    print("Example 4: Similarity Distribution Analysis")
    print("="*80)

    # Load retrieval results
    results_path = "outputs/retrieval_results/graph_to_text_retrieval.json"
    with open(results_path) as f:
        results = json.load(f)

    # Extract positive and negative similarities
    positive_sims = []  # Similarity with correct text
    negative_sims = []  # Similarity with top-1 incorrect text

    for r in results:
        # Positive: similarity with correct match (at correct_rank position)
        correct_rank_idx = r['correct_rank'] - 1
        positive_sim = r['top_k_similarities'][correct_rank_idx]
        positive_sims.append(positive_sim)

        # Negative: similarity with top-1 retrieved (if wrong)
        if r['correct_rank'] > 1:
            negative_sims.append(r['top_k_similarities'][0])

    # Statistics
    print(f"\nPositive pairs (correct matches):")
    print(f"  Mean similarity: {np.mean(positive_sims):.4f} ± {np.std(positive_sims):.4f}")
    print(f"  Min/Max: [{min(positive_sims):.4f}, {max(positive_sims):.4f}]")

    if negative_sims:
        print(f"\nNegative pairs (incorrect top-1 for failed cases):")
        print(f"  Mean similarity: {np.mean(negative_sims):.4f} ± {np.std(negative_sims):.4f}")
        print(f"  Min/Max: [{min(negative_sims):.4f}, {max(negative_sims):.4f}]")

        # Separation
        separation = np.mean(positive_sims) - np.mean(negative_sims)
        print(f"\nSeparation (pos - neg): {separation:.4f}")
        print(f"  (Larger separation indicates better discriminative power)")

        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.hist(positive_sims, bins=50, alpha=0.7, label='Positive pairs', color='green')
        plt.hist(negative_sims, bins=50, alpha=0.7, label='Negative pairs (failed cases)', color='red')
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Similarity Distribution: Positive vs Negative Pairs', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/similarity_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved similarity distribution plot to: outputs/similarity_distribution.png")
        plt.close()


def example_5_custom_metrics():
    """Example 5: Compute custom retrieval metrics."""
    print("\n" + "="*80)
    print("Example 5: Custom Retrieval Metrics")
    print("="*80)

    # Load retrieval results
    results_path = "outputs/retrieval_results/graph_to_text_retrieval.json"
    with open(results_path) as f:
        results = json.load(f)

    ranks = np.array([r['correct_rank'] for r in results])

    # Standard metrics
    r_at_1 = np.mean(ranks == 1) * 100
    r_at_5 = np.mean(ranks <= 5) * 100
    r_at_10 = np.mean(ranks <= 10) * 100
    mrr = np.mean(1.0 / ranks) * 100

    # Custom metrics
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)

    # Mean Average Precision (MAP)
    # For retrieval, AP for each query is 1/rank (since there's only 1 relevant item)
    map_score = mrr  # Equivalent to MRR for single relevant item per query

    # Success rate at different thresholds
    success_at_3 = np.mean(ranks <= 3) * 100
    success_at_20 = np.mean(ranks <= 20) * 100

    print(f"\nStandard Metrics:")
    print(f"  R@1:  {r_at_1:.2f}%")
    print(f"  R@5:  {r_at_5:.2f}%")
    print(f"  R@10: {r_at_10:.2f}%")
    print(f"  MRR:  {mrr:.2f}%")

    print(f"\nCustom Metrics:")
    print(f"  Median Rank: {median_rank:.1f}")
    print(f"  Mean Rank:   {mean_rank:.2f}")
    print(f"  MAP:         {map_score:.2f}%")
    print(f"  R@3:         {success_at_3:.2f}%")
    print(f"  R@20:        {success_at_20:.2f}%")

    # Rank distribution percentiles
    print(f"\nRank Distribution Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        percentile_rank = np.percentile(ranks, p)
        print(f"  {p}th percentile: {percentile_rank:.1f}")


def example_6_cross_dataset_comparison():
    """Example 6: Compare retrieval performance across different datasets or splits."""
    print("\n" + "="*80)
    print("Example 6: Cross-Dataset/Split Comparison")
    print("="*80)

    # Load metrics from different evaluations
    eval_dirs = {
        'Validation': 'outputs/retrieval_results_val',
        'Test': 'outputs/retrieval_results_test',
    }

    comparison_data = {}

    for name, dir_path in eval_dirs.items():
        metrics_path = Path(dir_path) / 'retrieval_metrics_test.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            comparison_data[name] = metrics
        else:
            print(f"⚠️  Metrics not found: {metrics_path}")

    if len(comparison_data) >= 2:
        # Print comparison table
        print(f"\n{'Metric':<15} " + " ".join([f"{name:>12}" for name in comparison_data.keys()]))
        print("-" * (15 + 13 * len(comparison_data)))

        metric_keys = ['Avg_R@1', 'Avg_R@5', 'Avg_R@10', 'Avg_MRR']
        for metric in metric_keys:
            values = [comparison_data[name].get(metric, 0) for name in comparison_data.keys()]
            value_strs = [f"{v:>11.2f}%" for v in values]
            print(f"{metric:<15} " + " ".join(value_strs))

        # Visualize comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metric_keys))
        width = 0.35
        colors = ['steelblue', 'coral', 'mediumseagreen']

        for i, (name, metrics) in enumerate(comparison_data.items()):
            values = [metrics.get(m, 0) for m in metric_keys]
            ax.bar(x + i*width, values, width, label=name, color=colors[i % len(colors)])

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Retrieval Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(comparison_data) - 1) / 2)
        ax.set_xticklabels(metric_keys)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/retrieval_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved comparison plot to: outputs/retrieval_comparison.png")
        plt.close()


def main():
    """Run all examples."""
    examples = [
        ("Basic Evaluation", example_1_basic_evaluation),
        ("Error Analysis", example_2_error_analysis),
        ("Target-Stratified Analysis", example_3_target_stratified_analysis),
        ("Similarity Distribution", example_4_similarity_distribution_analysis),
        ("Custom Metrics", example_5_custom_metrics),
        ("Cross-Dataset Comparison", example_6_cross_dataset_comparison),
    ]

    print("\n" + "="*80)
    print("Advanced Retrieval Analysis Examples")
    print("="*80)
    print("\nThis script demonstrates various ways to analyze retrieval results.")
    print("Uncomment the examples you want to run in the main() function.\n")

    # Run examples (comment out the ones you don't want to run)
    for name, func in examples:
        try:
            func()
        except FileNotFoundError as e:
            print(f"\n⚠️  Skipping {name}: {e}")
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Run specific examples by uncommenting:

    # example_1_basic_evaluation()
    # example_2_error_analysis()
    # example_3_target_stratified_analysis()
    # example_4_similarity_distribution_analysis()
    # example_5_custom_metrics()
    # example_6_cross_dataset_comparison()

    # Or run all:
    main()
