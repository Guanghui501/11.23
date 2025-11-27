#!/usr/bin/env python
"""
双模型特征可视化脚本
用于生成论文级别的对比图表
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# 导入模型
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.alignn import ALIGNN
from train_with_cross_modal_attention import load_dataset, get_dataset_paths
from data import get_train_val_loaders


def load_model(path, device):
    """加载模型"""
    print(f"📦 加载模型: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config') or ckpt.get('model_config')
    model = ALIGNN(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model


def extract_features(model, loader, device, max_samples=None, feature_stage='final'):
    """
    提取特征

    Args:
        feature_stage: 特征阶段选择
            - 'base': graph_base (GCN后，所有注意力前)
            - 'middle': graph_middle (中期融合后)
            - 'fine': graph_fine (细粒度注意力后)
            - 'final': graph_features (最终特征，默认)
    """
    features = []
    targets = []
    sample_count = 0

    # 特征键映射
    stage_key_map = {
        'base': 'graph_base',
        'middle': 'graph_middle',
        'fine': 'graph_fine',
        'final': 'graph_features'
    }

    feature_key = stage_key_map.get(feature_stage, 'graph_features')
    print(f"   提取阶段: {feature_stage} (键: {feature_key})")

    with torch.no_grad():
        for batch in tqdm(loader, desc="提取特征"):
            if len(batch) == 3:
                g, text, y = batch
                lg = None
            elif len(batch) == 4:
                g, lg, text, y = batch
            else:
                raise ValueError(f"不支持的batch格式: {len(batch)}个元素")

            g = g.to(device)
            if lg is not None:
                lg = lg.to(device)

            # 处理text
            if isinstance(text, dict):
                text = {k: v.to(device) for k, v in text.items()}
            elif torch.is_tensor(text):
                text = text.to(device)

            inputs = (g, lg, text) if lg is not None else (g, text)
            out = model(inputs, return_intermediate_features=True)

            # 根据指定阶段提取特征
            feat = out.get(feature_key)

            # 如果指定阶段不存在，回退到其他阶段
            if feat is None:
                print(f"⚠️  警告: {feature_key} 不存在，尝试回退...")
                feat = out.get('graph_features', out.get('graph_final', out.get('graph_base')))

            features.append(feat.cpu().numpy())
            targets.append(y.cpu().numpy())

            sample_count += y.size(0)
            if max_samples and sample_count >= max_samples:
                break

    return np.vstack(features), np.concatenate(targets)


def centered_kernel_alignment(X, Y):
    """计算 CKA 相似度"""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(K * L)
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return hsic / denom if denom > 0 else 0.0


def plot_tsne_comparison(feat_base, feat_sga, targets, save_dir, feature_stage='final'):
    """t-SNE 可视化对比"""
    print("\n📊 生成 t-SNE 可视化...")

    # 计算 t-SNE
    print("   计算 Baseline t-SNE...")
    tsne_base = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feat_base)

    print("   计算 SGANet t-SNE...")
    tsne_sga = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(feat_sga)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 统一颜色范围
    vmin, vmax = targets.min(), targets.max()

    # 标题后缀
    stage_suffix = f" [{feature_stage.upper()} stage]"

    # Baseline
    scatter1 = axes[0].scatter(tsne_base[:, 0], tsne_base[:, 1],
                               c=targets, cmap='viridis', alpha=0.6, s=20,
                               vmin=vmin, vmax=vmax)
    axes[0].set_title('Baseline Model' + stage_suffix, fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # SGANet
    scatter2 = axes[1].scatter(tsne_sga[:, 0], tsne_sga[:, 1],
                               c=targets, cmap='viridis', alpha=0.6, s=20,
                               vmin=vmin, vmax=vmax)
    axes[1].set_title('SGANet (With Middle Fusion)' + stage_suffix, fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter2, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Target Value', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tsne_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()


def plot_pca_comparison(feat_base, feat_sga, targets, save_dir, feature_stage='final'):
    """PCA 可视化对比"""
    print("\n📊 生成 PCA 可视化...")

    # 计算 PCA
    pca_base = PCA(n_components=2).fit_transform(feat_base)
    pca_sga = PCA(n_components=2).fit_transform(feat_sga)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = targets.min(), targets.max()

    # 标题后缀
    stage_suffix = f" [{feature_stage.upper()}]"

    # Baseline
    scatter1 = axes[0].scatter(pca_base[:, 0], pca_base[:, 1],
                               c=targets, cmap='viridis', alpha=0.6, s=20,
                               vmin=vmin, vmax=vmax)
    axes[0].set_title('Baseline Model (PCA)' + stage_suffix, fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PC 1', fontsize=12)
    axes[0].set_ylabel('PC 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # SGANet
    scatter2 = axes[1].scatter(pca_sga[:, 0], pca_sga[:, 1],
                               c=targets, cmap='viridis', alpha=0.6, s=20,
                               vmin=vmin, vmax=vmax)
    axes[1].set_title('SGANet (PCA)' + stage_suffix, fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC 1', fontsize=12)
    axes[1].set_ylabel('PC 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter2, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Target Value', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pca_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()


def plot_correlation_heatmap(feat_base, feat_sga, targets, save_dir):
    """特征-目标相关性热图"""
    print("\n📊 生成相关性热图...")

    # 计算每个维度与目标的相关性
    n_dims = min(feat_base.shape[1], 50)  # 最多显示50个维度

    corr_base = np.array([pearsonr(feat_base[:, i], targets)[0] for i in range(n_dims)])
    corr_sga = np.array([pearsonr(feat_sga[:, i], targets)[0] for i in range(n_dims)])

    # 创建热图数据
    heatmap_data = np.vstack([corr_base, corr_sga])

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
                yticklabels=['Baseline', 'SGANet'],
                xticklabels=[f'D{i}' for i in range(n_dims)],
                cbar_kws={'label': 'Pearson Correlation'},
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature-Target Correlation per Dimension', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Dimension', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()


def plot_metrics_comparison(feat_base, feat_sga, targets, cka_score, save_dir):
    """综合指标对比图"""
    print("\n📊 生成综合指标对比...")

    # 计算各种指标
    def get_avg_corr(X, y):
        corrs = [abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]
        return np.mean(corrs)

    def get_max_corr(X, y):
        corrs = [abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])]
        return np.max(corrs)

    metrics_base = {
        'Avg Pearson': get_avg_corr(feat_base, targets),
        'Max Pearson': get_max_corr(feat_base, targets),
        'Feature Variance': np.mean(np.var(feat_base, axis=0)),
        'Feature Norm': np.mean(np.linalg.norm(feat_base, axis=1))
    }

    metrics_sga = {
        'Avg Pearson': get_avg_corr(feat_sga, targets),
        'Max Pearson': get_max_corr(feat_sga, targets),
        'Feature Variance': np.mean(np.var(feat_sga, axis=0)),
        'Feature Norm': np.mean(np.linalg.norm(feat_sga, axis=1))
    }

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metric_names = list(metrics_base.keys())

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        values = [metrics_base[metric], metrics_sga[metric]]
        labels = ['Baseline', 'SGANet']
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(metric, fontweight='bold', fontsize=13)
        ax.grid(axis='y', alpha=0.3)

        # 标注数值和提升百分比
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=11)

        # 计算提升
        improvement = (metrics_sga[metric] - metrics_base[metric]) / metrics_base[metric] * 100
        ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Feature Quality Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()


def plot_feature_distribution(feat_base, feat_sga, save_dir):
    """特征分布对比"""
    print("\n📊 生成特征分布对比...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 计算每个样本的特征范数
    norm_base = np.linalg.norm(feat_base, axis=1)
    norm_sga = np.linalg.norm(feat_sga, axis=1)

    # Baseline 分布
    axes[0].hist(norm_base, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].axvline(norm_base.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {norm_base.mean():.2f}')
    axes[0].set_xlabel('Feature Norm', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Baseline Model - Feature Norm Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SGANet 分布
    axes[1].hist(norm_sga, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1].axvline(norm_sga.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {norm_sga.mean():.2f}')
    axes[1].set_xlabel('Feature Norm', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SGANet - Feature Norm Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'feature_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()


def create_summary_report(feat_base, feat_sga, targets, cka_score, save_dir, feature_stage='final'):
    """生成总结报告"""
    print("\n📝 生成总结报告...")

    def get_stats(X, y):
        avg_pearson = np.mean([abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
        max_pearson = np.max([abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
        variance = np.mean(np.var(X, axis=0))
        norm = np.mean(np.linalg.norm(X, axis=1))
        return avg_pearson, max_pearson, variance, norm

    stats_base = get_stats(feat_base, targets)
    stats_sga = get_stats(feat_sga, targets)

    # 阶段说明
    stage_explanations = {
        'base': 'GCN后，所有注意力前 (差异主要来自中期融合)',
        'middle': '中期融合后立即提取',
        'fine': '细粒度注意力后',
        'final': '所有模块处理后的最终特征'
    }

    report = f"""
╔═══════════════════════════════════════════════════════════════╗
║         Twin Model Feature Space Comparison Report           ║
╚═══════════════════════════════════════════════════════════════╝

Feature Extraction Stage: {feature_stage.upper()}
{stage_explanations.get(feature_stage, '')}


1. Feature Structure Similarity (CKA Score)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CKA Score: {cka_score:.4f}

   Interpretation:
   {'✓ Feature spaces are highly similar (>0.95)' if cka_score > 0.95 else '✓ Moderate structural change (0.85-0.95)' if cka_score > 0.85 else '! Significant structural change (<0.85)'}
   → Middle fusion provides {'conservative but effective' if cka_score > 0.95 else 'moderate' if cka_score > 0.85 else 'revolutionary'} improvement


2. Physical Property Correlation
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Metric              Baseline    SGANet      Improvement
   ──────────────────────────────────────────────────────
   Avg Pearson Corr    {stats_base[0]:.4f}      {stats_sga[0]:.4f}      {(stats_sga[0]-stats_base[0])/stats_base[0]*100:+.1f}%
   Max Pearson Corr    {stats_base[1]:.4f}      {stats_sga[1]:.4f}      {(stats_sga[1]-stats_base[1])/stats_base[1]*100:+.1f}%

   Interpretation:
   ✓ Avg correlation improvement: {(stats_sga[0]-stats_base[0])/stats_base[0]*100:.1f}%
   → Features are more predictive of physical properties


3. Feature Expressiveness
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Metric              Baseline    SGANet      Change
   ──────────────────────────────────────────────────────
   Feature Variance    {stats_base[2]:.4f}      {stats_sga[2]:.4f}      {(stats_sga[2]-stats_base[2])/stats_base[2]*100:+.1f}%
   Feature Norm        {stats_base[3]:.4f}      {stats_sga[3]:.4f}      {(stats_sga[3]-stats_base[3])/stats_base[3]*100:+.1f}%

   Interpretation:
   {'✓ No feature collapse detected (variance increased)' if stats_sga[2] > stats_base[2] else '⚠ Potential feature collapse (variance decreased)'}
   → Feature expressiveness {'enhanced' if stats_sga[2] > stats_base[2] else 'reduced'}


4. Overall Assessment
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ Structural Stability:  {'Excellent' if cka_score > 0.95 else 'Good' if cka_score > 0.85 else 'Moderate'}
   ✓ Predictive Quality:    {'Significantly Improved' if (stats_sga[0]-stats_base[0])/stats_base[0] > 0.1 else 'Moderately Improved' if (stats_sga[0]-stats_base[0])/stats_base[0] > 0.05 else 'Slightly Improved'}
   ✓ Feature Richness:      {'Enhanced' if stats_sga[2] > stats_base[2] else 'Unchanged'}

   Recommendation:
   {'✓ Middle fusion module is effective and ready for publication!' if (stats_sga[0]-stats_base[0])/stats_base[0] > 0.1 and cka_score > 0.9 else '→ Results show improvement but may need further tuning'}


═══════════════════════════════════════════════════════════════

Generated by: visualize_twin_models.py
"""

    # 保存报告
    report_path = os.path.join(save_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"✅ 报告保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='双模型特征可视化对比')
    parser.add_argument('--ckpt_base', required=True, help='基线模型checkpoint路径')
    parser.add_argument('--ckpt_sga', required=True, help='SGANet模型checkpoint路径')
    parser.add_argument('--root_dir', default='/public/home/ghzhang/crysmmnet-main/dataset',
                       help='数据集根目录')
    parser.add_argument('--dataset', default='jarvis', help='数据集名称')
    parser.add_argument('--property', default='mbj_bandgap', help='目标属性')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大样本数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--save_dir', default='./twin_model_visualization',
                       help='结果保存目录')
    parser.add_argument('--device', default='cuda', help='计算设备')
    parser.add_argument('--feature_stage', type=str, default='final',
                       choices=['base', 'middle', 'fine', 'final'],
                       help='提取特征的阶段: base=GCN后, middle=中期融合后, fine=细粒度注意力后, final=最终特征(默认)')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    # 显示特征提取阶段
    stage_descriptions = {
        'base': 'GCN后，所有注意力前 (Baseline: ALIGNN+GCN | SGANet: ALIGNN+中期融合+GCN)',
        'middle': '中期融合后立即提取 (仅SGANet有效)',
        'fine': '细粒度注意力后 (原子-文本token交互后)',
        'final': '最终特征 (所有模块处理后)'
    }
    print(f"\n🎯 特征提取阶段: {args.feature_stage}")
    print(f"   说明: {stage_descriptions[args.feature_stage]}")
    if args.feature_stage == 'base':
        print(f"   ⭐ 推荐用于评估中期融合的独立贡献")

    # 加载数据
    print(f"\n📂 加载数据集: {args.dataset} - {args.property}")
    cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
    dataset = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)

    # 采样
    if args.max_samples and len(dataset) > args.max_samples:
        print(f"⚠️  采样 {args.max_samples} 个样本")
        import random
        random.seed(42)
        dataset = random.sample(dataset, args.max_samples)

    # 创建数据加载器
    _, _, test_loader, _ = get_train_val_loaders(
        dataset='user_data',
        dataset_array=dataset,
        target='target',
        batch_size=args.batch_size,
        atom_features='cgcnn',
        neighbor_strategy='k-nearest',
        line_graph=True,
        workers=0,
        pin_memory=False,
        n_train=10,
        n_val=10,
        n_test=len(dataset)-20,
        split_seed=42,
        save_dataloader=False,
        filename='temp_viz',
        id_tag='jid',
        use_canonize=True,
        cutoff=8.0,
        max_neighbors=12,
        output_dir=args.save_dir
    )

    # 加载模型并提取特征
    print(f"\n📦 提取基线模型特征:")
    model_base = load_model(args.ckpt_base, device)
    feat_base, targets = extract_features(model_base, test_loader, device, args.max_samples, args.feature_stage)

    print(f"\n📦 提取SGANet模型特征:")
    model_sga = load_model(args.ckpt_sga, device)
    feat_sga, _ = extract_features(model_sga, test_loader, device, args.max_samples, args.feature_stage)

    print(f"\n✅ 特征提取完成:")
    print(f"   Baseline: {feat_base.shape}")
    print(f"   SGANet:   {feat_sga.shape}")
    print(f"   Targets:  {targets.shape}")

    # 计算 CKA
    print("\n🔍 计算特征相似度...")
    cka_score = centered_kernel_alignment(feat_base, feat_sga)
    print(f"   CKA Score: {cka_score:.4f}")

    # 生成所有可视化
    print("\n" + "="*60)
    print("开始生成可视化图表...")
    print("="*60)

    plot_tsne_comparison(feat_base, feat_sga, targets, args.save_dir, args.feature_stage)
    plot_pca_comparison(feat_base, feat_sga, targets, args.save_dir, args.feature_stage)
    plot_correlation_heatmap(feat_base, feat_sga, targets, args.save_dir)
    plot_metrics_comparison(feat_base, feat_sga, targets, cka_score, args.save_dir)
    plot_feature_distribution(feat_base, feat_sga, args.save_dir)
    create_summary_report(feat_base, feat_sga, targets, cka_score, args.save_dir, args.feature_stage)

    print("\n" + "="*60)
    print(f"🎉 所有可视化完成! 结果保存在: {args.save_dir}")
    print("="*60)
    print("\n生成的文件:")
    print("  1. tsne_comparison.png          - t-SNE 降维对比")
    print("  2. pca_comparison.png           - PCA 降维对比")
    print("  3. correlation_heatmap.png      - 特征-目标相关性热图")
    print("  4. metrics_comparison.png       - 综合指标对比")
    print("  5. feature_distribution.png     - 特征分布对比")
    print("  6. comparison_report.txt        - 详细文本报告")
    print("")


if __name__ == "__main__":
    main()
