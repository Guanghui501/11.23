#!/usr/bin/env python3
"""
Compare MASKING vs DELETION results

Shows the difference between:
- Masking: Replace with [MASK] tokens (still has embeddings)
- Deletion: Remove completely (empty string, true graph-only)
"""

import argparse
import json
import glob
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")


def load_results(pattern: str) -> Dict:
    """Load all result files matching pattern"""
    files = glob.glob(pattern)
    results = {}

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        # Extract strategy and ratio from filename
        # e.g., results_masking_model1_random_token_0.5.json
        basename = os.path.basename(file)
        parts = basename.replace('.json', '').split('_')

        # Find strategy and ratio
        strategy = None
        ratio = None
        for i, part in enumerate(parts):
            if part in ['random', 'sentence', 'keep']:
                if part == 'random' and i + 1 < len(parts):
                    strategy = f"{part}_{parts[i+1]}"
                elif part == 'keep' and i + 1 < len(parts):
                    strategy = f"{part}_{parts[i+1]}"
                else:
                    strategy = part
            try:
                ratio = float(part)
            except:
                pass

        if strategy and ratio is not None:
            key = (strategy, ratio)
            results[key] = data

    return results


def plot_masking_vs_deletion_comparison(
    masking_model1: Dict,
    masking_model2: Dict,
    deletion_model1: Dict,
    deletion_model2: Dict,
    model1_name: str,
    model2_name: str,
    output_dir: str
):
    """Create comprehensive comparison plots"""

    os.makedirs(output_dir, exist_ok=True)

    # Get all strategies
    strategies = set()
    for results in [masking_model1, masking_model2, deletion_model1, deletion_model2]:
        for (strategy, _) in results.keys():
            strategies.add(strategy)

    strategies = sorted(strategies)

    # For each strategy, create comparison plot
    for strategy in strategies:
        print(f"\nPlotting comparison for strategy: {strategy}")

        # Collect data for this strategy
        ratios = []
        masking_m1_mae = []
        masking_m2_mae = []
        deletion_m1_mae = []
        deletion_m2_mae = []

        for (strat, ratio) in sorted(masking_model1.keys()):
            if strat == strategy:
                ratios.append(ratio * 100)  # Convert to percentage
                masking_m1_mae.append(masking_model1[(strat, ratio)].get('mae', np.nan))

                masking_m2_mae.append(masking_model2.get((strat, ratio), {}).get('mae', np.nan))
                deletion_m1_mae.append(deletion_model1.get((strat, ratio), {}).get('mae', np.nan))
                deletion_m2_mae.append(deletion_model2.get((strat, ratio), {}).get('mae', np.nan))

        if not ratios:
            continue

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Masking vs Deletion Comparison - {strategy}', fontsize=16, fontweight='bold')

        # Plot 1: Model 1 comparison
        ax1 = axes[0, 0]
        ax1.plot(ratios, masking_m1_mae, 'o-', color='#E63946', linewidth=2.5, markersize=8,
                label=f'{model1_name} (MASKING)')
        ax1.plot(ratios, deletion_m1_mae, 's-', color='#06A77D', linewidth=2.5, markersize=8,
                label=f'{model1_name} (DELETION)')
        ax1.set_xlabel('Deletion/Masking Ratio (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model1_name}: Masking vs Deletion', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Annotate 100% difference
        if len(ratios) > 0 and ratios[-1] == 100:
            diff = deletion_m1_mae[-1] - masking_m1_mae[-1]
            improvement = (masking_m1_mae[-1] - deletion_m1_mae[-1]) / masking_m1_mae[-1] * 100
            color = 'green' if diff < 0 else 'red'
            ax1.annotate(f'Δ = {diff:+.4f}\n({improvement:+.1f}%)',
                        xy=(100, deletion_m1_mae[-1]),
                        xytext=(85, deletion_m1_mae[-1] + 0.05),
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color=color))

        # Plot 2: Model 2 comparison
        ax2 = axes[0, 1]
        ax2.plot(ratios, masking_m2_mae, 'o-', color='#E63946', linewidth=2.5, markersize=8,
                label=f'{model2_name} (MASKING)')
        ax2.plot(ratios, deletion_m2_mae, 's-', color='#06A77D', linewidth=2.5, markersize=8,
                label=f'{model2_name} (DELETION)')
        ax2.set_xlabel('Deletion/Masking Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax2.set_title(f'{model2_name}: Masking vs Deletion', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Annotate 100% difference
        if len(ratios) > 0 and ratios[-1] == 100:
            diff = deletion_m2_mae[-1] - masking_m2_mae[-1]
            improvement = (masking_m2_mae[-1] - deletion_m2_mae[-1]) / masking_m2_mae[-1] * 100
            color = 'green' if diff < 0 else 'red'
            ax2.annotate(f'Δ = {diff:+.4f}\n({improvement:+.1f}%)',
                        xy=(100, deletion_m2_mae[-1]),
                        xytext=(85, deletion_m2_mae[-1] + 0.05),
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color=color))

        # Plot 3: 100% comparison bar chart
        ax3 = axes[1, 0]
        if len(ratios) > 0 and ratios[-1] == 100:
            categories = ['MASKING\n(has [MASK] embeddings)', 'DELETION\n(empty string)']
            x = np.arange(len(categories))
            width = 0.35

            m1_values = [masking_m1_mae[-1], deletion_m1_mae[-1]]
            m2_values = [masking_m2_mae[-1], deletion_m2_mae[-1]]

            bars1 = ax3.bar(x - width/2, m1_values, width, label=model1_name, color='#2E86AB')
            bars2 = ax3.bar(x + width/2, m2_values, width, label=model2_name, color='#A23B72')

            ax3.set_ylabel('MAE at 100%', fontsize=12, fontweight='bold')
            ax3.set_title('100% Masking vs Deletion Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories, fontsize=10)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Plot 4: Improvement from Deletion
        ax4 = axes[1, 1]
        m1_improvement = [(masking_m1_mae[i] - deletion_m1_mae[i]) / masking_m1_mae[i] * 100
                          for i in range(len(ratios))]
        m2_improvement = [(masking_m2_mae[i] - deletion_m2_mae[i]) / masking_m2_mae[i] * 100
                          for i in range(len(ratios))]

        ax4.plot(ratios, m1_improvement, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
                label=model1_name)
        ax4.plot(ratios, m2_improvement, 's-', color='#A23B72', linewidth=2.5, markersize=8,
                label=model2_name)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Deletion/Masking Ratio (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Improvement from Deletion (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Deletion vs Masking Improvement', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # Add annotations
        ax4.text(0.05, 0.95, 'Positive = Deletion better\nNegative = Masking better',
                transform=ax4.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'comparison_{strategy}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_file}")

    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY: Masking vs Deletion at 100%")
    print("="*80)

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        print("-" * 80)

        # Get 100% results
        key = (strategy, 1.0)

        masking_m1 = masking_model1.get(key, {}).get('mae', np.nan)
        masking_m2 = masking_model2.get(key, {}).get('mae', np.nan)
        deletion_m1 = deletion_model1.get(key, {}).get('mae', np.nan)
        deletion_m2 = deletion_model2.get(key, {}).get('mae', np.nan)

        print(f"{model1_name}:")
        print(f"  MASKING (text='[MASK] [MASK] ...'):  MAE = {masking_m1:.4f}")
        print(f"  DELETION (text=''):                  MAE = {deletion_m1:.4f}")
        if not np.isnan(masking_m1) and not np.isnan(deletion_m1):
            improvement = (masking_m1 - deletion_m1) / masking_m1 * 100
            better = "DELETION" if improvement > 0 else "MASKING"
            print(f"  → {better} is better by {abs(improvement):.2f}%")

        print(f"\n{model2_name}:")
        print(f"  MASKING (text='[MASK] [MASK] ...'):  MAE = {masking_m2:.4f}")
        print(f"  DELETION (text=''):                  MAE = {deletion_m2:.4f}")
        if not np.isnan(masking_m2) and not np.isnan(deletion_m2):
            improvement = (masking_m2 - deletion_m2) / masking_m2 * 100
            better = "DELETION" if improvement > 0 else "MASKING"
            print(f"  → {better} is better by {abs(improvement):.2f}%")

    print("\n" + "="*80)
    print("Key Insight:")
    print("If DELETION is better at 100%, it means [MASK] embeddings are HARMFUL!")
    print("This proves that fixed Middle Fusion mixes bad [MASK] embeddings with graph features.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare masking vs deletion results')
    parser.add_argument('--masking_model1', type=str, required=True,
                       help='Masking results for model 1 (glob pattern)')
    parser.add_argument('--masking_model2', type=str, required=True,
                       help='Masking results for model 2 (glob pattern)')
    parser.add_argument('--deletion_model1', type=str, required=True,
                       help='Deletion results for model 1 (glob pattern)')
    parser.add_argument('--deletion_model2', type=str, required=True,
                       help='Deletion results for model 2 (glob pattern)')
    parser.add_argument('--model1_name', type=str, default='Model 1',
                       help='Name for model 1')
    parser.add_argument('--model2_name', type=str, default='Model 2',
                       help='Name for model 2')
    parser.add_argument('--output_dir', type=str, default='./masking_vs_deletion_comparison',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("Loading results...")
    masking_m1 = load_results(args.masking_model1)
    masking_m2 = load_results(args.masking_model2)
    deletion_m1 = load_results(args.deletion_model1)
    deletion_m2 = load_results(args.deletion_model2)

    print(f"Loaded {len(masking_m1)} masking results for {args.model1_name}")
    print(f"Loaded {len(masking_m2)} masking results for {args.model2_name}")
    print(f"Loaded {len(deletion_m1)} deletion results for {args.model1_name}")
    print(f"Loaded {len(deletion_m2)} deletion results for {args.model2_name}")

    plot_masking_vs_deletion_comparison(
        masking_m1, masking_m2,
        deletion_m1, deletion_m2,
        args.model1_name, args.model2_name,
        args.output_dir
    )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
