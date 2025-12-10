#!/usr/bin/env python3
"""
诊断Gated Cross-Attention性能问题

分析：
1. 质量得分分布
2. 融合权重分布
3. 有效权重分布
4. 与原始跨模态注意力对比

问题：Gated Attention (0.2672) vs SGANet (0.2554)，性能差4.6%
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

from models.alignn import ALIGNN
from data import get_train_val_loaders


def diagnose_gating_behavior(model, data_loader, device='cpu', max_batches=50):
    """分析门控机制的行为"""
    model.eval()

    quality_scores = []
    fusion_weights = []
    effective_weights = []
    text_norms = []
    predictions = []
    targets = []

    print("🔍 采集门控统计数据...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing")):
            if batch_idx >= max_batches:
                break

            # Unpack batch
            if len(batch) == 4:
                g, lg, text, target = batch
                model_input = (g.to(device), lg.to(device), text)
            else:
                g, text, target = batch
                model_input = (g.to(device), text)

            # Forward with diagnostics
            output = model(model_input, return_attention=True)

            # Extract predictions
            if isinstance(output, tuple):
                pred, attention_info = output
            else:
                pred = output
                attention_info = {}

            predictions.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

            # Extract gating diagnostics if available
            if 'quality_diagnostics' in attention_info:
                diag = attention_info['quality_diagnostics']
                quality_scores.append(diag['quality_score'].cpu().numpy())
                fusion_weights.append(diag['fusion_weight'].cpu().numpy())
                effective_weights.append(diag['effective_weight'].cpu().numpy())

    # Concatenate results
    quality_scores = np.concatenate(quality_scores, axis=0) if quality_scores else None
    fusion_weights = np.concatenate(fusion_weights, axis=0) if fusion_weights else None
    effective_weights = np.concatenate(effective_weights, axis=0) if effective_weights else None
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute MAE
    mae = np.mean(np.abs(predictions - targets))

    return {
        'quality_scores': quality_scores,
        'fusion_weights': fusion_weights,
        'effective_weights': effective_weights,
        'predictions': predictions,
        'targets': targets,
        'mae': mae
    }


def plot_diagnostics(results, save_dir):
    """生成诊断可视化"""
    os.makedirs(save_dir, exist_ok=True)

    quality = results['quality_scores']
    fusion = results['fusion_weights']
    effective = results['effective_weights']

    if quality is None:
        print("⚠️ 未找到门控诊断信息，模型可能未使用Gated Cross-Attention")
        return

    # Flatten arrays
    quality = quality.flatten()
    fusion = fusion.flatten()
    effective = effective.flatten()

    # Create comprehensive diagnostic plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Quality Score Distribution
    ax1 = axes[0, 0]
    ax1.hist(quality, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(quality.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {quality.mean():.3f}')
    ax1.axvline(quality.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {quality.median():.3f}')
    ax1.set_xlabel('Quality Score', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Text Quality Score Distribution', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Fusion Weight Distribution
    ax2 = axes[0, 1]
    ax2.hist(fusion, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(fusion.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {fusion.mean():.3f}')
    ax2.axvline(fusion.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {fusion.median():.3f}')
    ax2.set_xlabel('Fusion Weight', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Adaptive Fusion Weight Distribution', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Effective Weight Distribution
    ax3 = axes[0, 2]
    ax3.hist(effective, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(effective.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {effective.mean():.3f}')
    ax3.axvline(effective.median(), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Median: {effective.median():.3f}')
    ax3.set_xlabel('Effective Weight', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('Effective Weight (Quality × Fusion)', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Quality vs Fusion scatter
    ax4 = axes[1, 0]
    ax4.scatter(quality, fusion, alpha=0.3, s=10)
    ax4.set_xlabel('Quality Score', fontweight='bold')
    ax4.set_ylabel('Fusion Weight', fontweight='bold')
    ax4.set_title('Quality vs Fusion Weight', fontweight='bold', fontsize=14)
    ax4.grid(alpha=0.3)

    # Add correlation
    corr = np.corrcoef(quality, fusion)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax4.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. Quality vs Effective scatter
    ax5 = axes[1, 1]
    ax5.scatter(quality, effective, alpha=0.3, s=10, color='green')
    ax5.set_xlabel('Quality Score', fontweight='bold')
    ax5.set_ylabel('Effective Weight', fontweight='bold')
    ax5.set_title('Quality vs Effective Weight', fontweight='bold', fontsize=14)
    ax5.grid(alpha=0.3)

    # Add diagonal reference line (y=x)
    ax5.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='y=x')
    ax5.legend()

    # 6. Statistics summary
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate key statistics
    summary_text = "GATING STATISTICS SUMMARY\n"
    summary_text += "=" * 50 + "\n\n"

    summary_text += f"Text Quality Score:\n"
    summary_text += f"  Mean:   {quality.mean():.4f}\n"
    summary_text += f"  Median: {quality.median():.4f}\n"
    summary_text += f"  Std:    {quality.std():.4f}\n"
    summary_text += f"  Min:    {quality.min():.4f}\n"
    summary_text += f"  Max:    {quality.max():.4f}\n\n"

    summary_text += f"Fusion Weight:\n"
    summary_text += f"  Mean:   {fusion.mean():.4f}\n"
    summary_text += f"  Median: {fusion.median():.4f}\n"
    summary_text += f"  Std:    {fusion.std():.4f}\n\n"

    summary_text += f"Effective Weight (Quality × Fusion):\n"
    summary_text += f"  Mean:   {effective.mean():.4f}\n"
    summary_text += f"  Median: {effective.median():.4f}\n"
    summary_text += f"  Std:    {effective.std():.4f}\n\n"

    # Diagnosis
    summary_text += "=" * 50 + "\n"
    summary_text += "DIAGNOSIS:\n\n"

    if effective.mean() < 0.2:
        summary_text += "⚠️ PROBLEM: Effective weight too low!\n"
        summary_text += "   Text is being heavily suppressed.\n\n"

    if quality.mean() < 0.3:
        summary_text += "⚠️ PROBLEM: Quality scores too low!\n"
        summary_text += "   Quality gate is too conservative.\n\n"

    if fusion.mean() < 0.3:
        summary_text += "⚠️ PROBLEM: Fusion weights too low!\n"
        summary_text += "   Fusion gate is underutilizing text.\n\n"

    if effective.mean() > 0.7:
        summary_text += "⚠️ WARNING: Effective weight very high!\n"
        summary_text += "   May not adapt to poor text quality.\n\n"

    summary_text += f"MAE: {results['mae']:.4f}\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gating_diagnostics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 诊断图已保存: {save_path}")
    plt.close()

    # Print summary to console
    print("\n" + "=" * 70)
    print("门控机制统计摘要")
    print("=" * 70)
    print(f"\n质量得分 (Quality Score):")
    print(f"  均值: {quality.mean():.4f}  中位数: {quality.median():.4f}  标准差: {quality.std():.4f}")
    print(f"\n融合权重 (Fusion Weight):")
    print(f"  均值: {fusion.mean():.4f}  中位数: {fusion.median():.4f}  标准差: {fusion.std():.4f}")
    print(f"\n有效权重 (Effective Weight = Quality × Fusion):")
    print(f"  均值: {effective.mean():.4f}  中位数: {effective.median():.4f}  标准差: {effective.std():.4f}")
    print(f"\nMAE: {results['mae']:.4f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='诊断Gated Cross-Attention性能问题')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='jarvis',
                        help='数据集名称')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                        help='目标属性')
    parser.add_argument('--root_dir', type=str,
                        default='/public/home/ghzhang/crysmmnet-main/dataset',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='最大批次数（用于快速诊断）')
    parser.add_argument('--output_dir', type=str, default='./gating_diagnosis',
                        help='输出目录')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Gated Cross-Attention 诊断工具")
    print("=" * 70)

    # Load model
    print(f"\n📂 加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' not in checkpoint:
        print("❌ Checkpoint中未找到配置信息")
        return

    config = checkpoint['config']
    model = ALIGNN(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    model.eval()

    print(f"✅ 模型加载完成")
    print(f"   使用Gated Cross-Attention: {config.use_gated_cross_attention}")

    if not config.use_gated_cross_attention:
        print("⚠️ 警告: 模型未启用Gated Cross-Attention!")
        print("   请使用 --use_gated_cross_attention True 训练模型")
        return

    # Load data
    print(f"\n📊 加载数据集: {args.dataset}/{args.property}")

    try:
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
            dataset=args.dataset,
            target=args.property,
            n_train=None,
            n_val=None,
            n_test=None,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=args.batch_size,
            atom_features=config.atom_features if hasattr(config, 'atom_features') else 'cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=config.line_graph if hasattr(config, 'line_graph') else True,
            split_seed=42,
            workers=0,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_diagnosis',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.output_dir,
            root_dir=args.root_dir
        )

        print(f"✅ 数据集加载完成: {len(test_loader.dataset)} 测试样本")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # Diagnose
    print(f"\n🔬 开始诊断...")
    results = diagnose_gating_behavior(
        model, test_loader, args.device, args.max_batches
    )

    # Visualize
    print(f"\n📊 生成诊断可视化...")
    plot_diagnostics(results, args.output_dir)

    print(f"\n✅ 诊断完成! 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
