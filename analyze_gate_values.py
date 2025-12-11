#!/usr/bin/env python
"""
门控值 (α) 可视化和分析工具

这个脚本提供了完整的α值分析功能：
1. 提取模型的α值
2. 统计分析（按材料类型、元素类型）
3. 生成多种可视化图表
4. 创建论文级别的热图和分布图

用法:
    python analyze_gate_values.py \
        --checkpoint best_model.pt \
        --dataset jarvis/mbj_bandgap \
        --output_dir alpha_analysis/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from models.alignn import ALIGNN


class GateValueExtractor:
    """提取模型中期融合的门控值"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

        # 存储α值的hook
        self.alpha_values = []
        self.register_hooks()

    def register_hooks(self):
        """注册hook来捕获α值"""

        def hook_fn(module, input, output):
            """Hook函数，捕获门控值"""
            if hasattr(module, 'gate_values'):
                # 如果中期融合模块存储了α值
                self.alpha_values.append(module.gate_values.detach().cpu())

        # 为中期融合模块注册hook
        if hasattr(self.model, 'middle_fusion_modules'):
            for name, module in self.model.middle_fusion_modules.items():
                module.register_forward_hook(hook_fn)

    def extract_alpha_from_batch(self, batch):
        """从一个batch提取α值"""
        g, lg, text, labels = batch

        self.alpha_values = []  # 清空

        with torch.no_grad():
            # Forward pass
            output = self.model(
                [g.to(self.device), lg.to(self.device), text],
                return_attention=True,
                return_features=True
            )

        # 提取α值（如果模型直接返回）
        if isinstance(output, dict) and 'gate_values' in output:
            alphas = output['gate_values']
        elif len(self.alpha_values) > 0:
            alphas = torch.cat(self.alpha_values, dim=0)
        else:
            # 如果没有α值，返回None
            return None, None, None

        # 提取原子信息
        atom_types = g.ndata['atom_features'].cpu().numpy()

        return alphas, atom_types, labels


def extract_all_gate_values(model, data_loader, device, n_samples=500):
    """
    提取数据集中的所有门控值

    Returns:
        gate_values_dict: {
            'alphas': List[np.ndarray],  # 每个样本的α值
            'atom_types': List[np.ndarray],  # 每个样本的原子类型
            'labels': List[float],  # 每个样本的标签
            'material_ids': List[str]  # 材料ID
        }
    """

    extractor = GateValueExtractor(model, device)

    gate_values_dict = {
        'alphas': [],
        'atom_types': [],
        'labels': [],
        'n_atoms': []
    }

    print(f"\n🔄 提取门控值（从{n_samples}个样本）...")

    count = 0
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        if count >= n_samples:
            break

        alphas, atom_types, labels = extractor.extract_alpha_from_batch(batch)

        if alphas is None:
            print("⚠️  模型未返回门控值，请确保模型有中期融合模块")
            continue

        # 按样本分割
        batch_size = len(labels)
        for i in range(batch_size):
            if count >= n_samples:
                break

            gate_values_dict['alphas'].append(alphas[i].cpu().numpy())
            gate_values_dict['atom_types'].append(atom_types[i])
            gate_values_dict['labels'].append(labels[i].item())
            gate_values_dict['n_atoms'].append(len(alphas[i]))

            count += 1

    print(f"✅ 已提取 {count} 个样本的门控值")

    return gate_values_dict


def visualize_alpha_distribution(gate_values_dict, save_path):
    """
    可视化α值的整体分布

    生成 Figure: Gate Value Distribution
    - (a) 直方图
    - (b) 箱型图
    - (c) 小提琴图
    - (d) CDF累积分布
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 合并所有α值
    all_alphas = np.concatenate(gate_values_dict['alphas'])

    # (a) 直方图
    ax = axes[0, 0]
    ax.hist(all_alphas, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(all_alphas.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {all_alphas.mean():.3f}')
    ax.axvline(np.median(all_alphas), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_alphas):.3f}')
    ax.set_xlabel('Gate Value (α)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('(a) Gate Value Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Std: {all_alphas.std():.3f}\nMin: {all_alphas.min():.3f}\nMax: {all_alphas.max():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (b) 箱型图（按样本数量分组）
    ax = axes[0, 1]

    # 按材料的原子数分组
    n_atoms_bins = [0, 10, 20, 50, 100, 1000]
    binned_alphas = [[] for _ in range(len(n_atoms_bins) - 1)]

    for alphas, n_atoms in zip(gate_values_dict['alphas'], gate_values_dict['n_atoms']):
        for i in range(len(n_atoms_bins) - 1):
            if n_atoms_bins[i] < n_atoms <= n_atoms_bins[i+1]:
                binned_alphas[i].extend(alphas)
                break

    bp = ax.boxplot(binned_alphas, labels=[f'{n_atoms_bins[i]}-{n_atoms_bins[i+1]}'
                                            for i in range(len(n_atoms_bins)-1)],
                    patch_artist=True)

    # 美化箱型图
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(binned_alphas)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Number of Atoms', fontsize=12)
    ax.set_ylabel('Gate Value (α)', fontsize=12)
    ax.set_title('(b) α Distribution by Material Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (c) 小提琴图（按目标值分组）
    ax = axes[1, 0]

    # 按标签值分组（例如带隙）
    labels = np.array(gate_values_dict['labels'])
    label_bins = np.percentile(labels, [0, 33, 67, 100])
    binned_by_label = [[] for _ in range(3)]

    for alphas, label in zip(gate_values_dict['alphas'], labels):
        if label <= label_bins[1]:
            binned_by_label[0].extend(alphas)
        elif label <= label_bins[2]:
            binned_by_label[1].extend(alphas)
        else:
            binned_by_label[2].extend(alphas)

    parts = ax.violinplot(binned_by_label, positions=[1, 2, 3],
                          showmeans=True, showmedians=True)

    # 美化小提琴图
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plt.cm.RdYlBu(i / 3))
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Low\nTarget', 'Medium\nTarget', 'High\nTarget'])
    ax.set_ylabel('Gate Value (α)', fontsize=12)
    ax.set_title('(c) α Distribution by Target Value', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (d) CDF累积分布
    ax = axes[1, 1]

    sorted_alphas = np.sort(all_alphas)
    cumulative = np.arange(1, len(sorted_alphas) + 1) / len(sorted_alphas)

    ax.plot(sorted_alphas, cumulative, linewidth=2, color='#2ecc71')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axvline(np.median(all_alphas), color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gate Value (α)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('(d) Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ α分布图已保存: {save_path}")

    plt.close()


def analyze_by_element(gate_values_dict, save_path):
    """
    按元素类型分析α值

    生成 Figure: Gate Values by Element Type
    """

    # 元素ID到名称的映射（前92个元素）
    element_names = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U'
    ]

    # 收集每个元素的α值
    element_alphas = defaultdict(list)

    for alphas, atom_types in zip(gate_values_dict['alphas'], gate_values_dict['atom_types']):
        for alpha, atom_type in zip(alphas, atom_types):
            element_id = int(atom_type[0]) if isinstance(atom_type, np.ndarray) else int(atom_type)
            if element_id < len(element_names):
                element_alphas[element_names[element_id]].append(alpha)

    # 过滤：只保留出现次数 > 50 的元素
    element_alphas = {k: v for k, v in element_alphas.items() if len(v) > 50}

    # 按平均α值排序
    element_stats = {
        elem: {
            'mean': np.mean(alphas),
            'std': np.std(alphas),
            'count': len(alphas)
        }
        for elem, alphas in element_alphas.items()
    }

    sorted_elements = sorted(element_stats.items(), key=lambda x: x[1]['mean'])

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # (a) 柱状图：平均α值
    ax = axes[0]

    elements = [elem for elem, _ in sorted_elements]
    means = [stats['mean'] for _, stats in sorted_elements]
    stds = [stats['std'] for _, stats in sorted_elements]

    colors = plt.cm.RdYlGn_r(np.array(means))  # 绿色=高α（依赖图），红色=低α（依赖文本）

    bars = ax.bar(range(len(elements)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor='black', alpha=0.8)

    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=45, ha='right')
    ax.set_ylabel('Mean Gate Value (α)', fontsize=12)
    ax.set_title('(a) Average α by Element (Error bars = Std)', fontsize=13, fontweight='bold')
    ax.axhline(np.mean(means), color='blue', linestyle='--', linewidth=2,
               label=f'Overall Mean: {np.mean(means):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加颜色条说明
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                               norm=plt.Normalize(vmin=min(means), vmax=max(means)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('α value\n(Low=Text, High=Graph)', fontsize=10)

    # (b) 箱型图：详细分布
    ax = axes[1]

    # 只显示前20个元素（否则太挤）
    top_elements = elements[:20]
    top_alphas = [element_alphas[elem] for elem in top_elements]

    bp = ax.boxplot(top_alphas, labels=top_elements, patch_artist=True)

    for patch, elem in zip(bp['boxes'], top_elements):
        color = plt.cm.RdYlGn_r(element_stats[elem]['mean'])
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(top_elements, rotation=45, ha='right')
    ax.set_ylabel('Gate Value (α)', fontsize=12)
    ax.set_title('(b) α Distribution for Top 20 Elements', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 按元素分析图已保存: {save_path}")

    # 打印统计
    print("\n📊 元素α值统计（Top 10高依赖文本 vs Top 10高依赖图结构）:")
    print("="*70)
    print(f"{'Element':<10} {'Mean α':<10} {'Std':<10} {'Count':<10} {'Reliance'}")
    print("-"*70)

    print("\n低α（更依赖文本）:")
    for elem, stats in sorted_elements[:10]:
        print(f"{elem:<10} {stats['mean']:<10.3f} {stats['std']:<10.3f} {stats['count']:<10} Text-heavy")

    print("\n高α（更依赖图结构）:")
    for elem, stats in sorted_elements[-10:]:
        print(f"{elem:<10} {stats['mean']:<10.3f} {stats['std']:<10.3f} {stats['count']:<10} Graph-heavy")

    plt.close()

    return element_stats


def create_alpha_heatmap_for_materials(model, data_loader, device,
                                       material_indices=[0, 1, 2],
                                       save_path='alpha_heatmap.pdf'):
    """
    为特定材料创建α值热图

    展示每个原子的α值

    Args:
        material_indices: 要可视化的材料索引列表
    """

    extractor = GateValueExtractor(model, device)

    # 提取指定材料的数据
    materials_data = []

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max(material_indices) + 1:
            break

        if batch_idx in material_indices:
            alphas, atom_types, labels = extractor.extract_alpha_from_batch(batch)

            if alphas is not None:
                materials_data.append({
                    'alphas': alphas[0].cpu().numpy(),  # 第一个样本
                    'atom_types': atom_types[0],
                    'label': labels[0].item(),
                    'n_atoms': len(alphas[0])
                })

    # 创建热图
    n_materials = len(materials_data)
    fig, axes = plt.subplots(1, n_materials, figsize=(5*n_materials, 6))

    if n_materials == 1:
        axes = [axes]

    element_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    for idx, (ax, mat_data) in enumerate(zip(axes, materials_data)):
        alphas = mat_data['alphas']
        atom_types = mat_data['atom_types']

        # 准备数据
        n_atoms = len(alphas)
        alpha_matrix = alphas.reshape(-1, 1)  # 列向量

        # 获取元素名称
        atom_labels = [element_names[int(at[0])] if int(at[0]) < len(element_names) else f'X{int(at[0])}'
                      for at in atom_types]

        # 绘制热图
        im = ax.imshow(alpha_matrix, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')

        # 设置刻度
        ax.set_yticks(range(n_atoms))
        ax.set_yticklabels([f'{i}: {label}' for i, label in enumerate(atom_labels)],
                          fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels(['α'])

        # 在每个格子上标注数值
        for i in range(n_atoms):
            text = ax.text(0, i, f'{alphas[i]:.2f}',
                          ha="center", va="center", color="black",
                          fontsize=10, fontweight='bold')

        # 标题
        ax.set_title(f'Material {idx+1}\nTarget: {mat_data["label"]:.3f}\n{n_atoms} atoms',
                    fontsize=11, fontweight='bold')

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Gate Value (α)\nLow=Text, High=Graph', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 材料α热图已保存: {save_path}")

    plt.close()


def analyze_alpha_vs_target(gate_values_dict, save_path):
    """
    分析α值与目标性质的关系

    生成 Figure: Gate Value vs Target Property
    """

    # 计算每个样本的平均α
    avg_alphas = [np.mean(alphas) for alphas in gate_values_dict['alphas']]
    labels = gate_values_dict['labels']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) 散点图
    ax = axes[0]

    scatter = ax.scatter(labels, avg_alphas, alpha=0.5, s=30,
                        c=avg_alphas, cmap='RdYlGn_r',
                        edgecolors='black', linewidth=0.5)

    # 拟合趋势线
    z = np.polyfit(labels, avg_alphas, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(labels), max(labels), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2,
            label=f'Trend: α = {z[0]:.4f}·y + {z[1]:.3f}')

    # 计算相关性
    correlation = np.corrcoef(labels, avg_alphas)[0, 1]

    ax.set_xlabel('Target Property Value', fontsize=12)
    ax.set_ylabel('Average Gate Value (α)', fontsize=12)
    ax.set_title(f'(a) α vs Target (Correlation: {correlation:.3f})',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean α', fontsize=10)

    # (b) 按目标值分组的箱型图
    ax = axes[1]

    # 分成5组
    n_bins = 5
    label_bins = np.percentile(labels, np.linspace(0, 100, n_bins+1))
    binned_alphas = [[] for _ in range(n_bins)]

    for avg_alpha, label in zip(avg_alphas, labels):
        for i in range(n_bins):
            if label_bins[i] <= label < label_bins[i+1]:
                binned_alphas[i].append(avg_alpha)
                break

    bp = ax.boxplot(binned_alphas,
                    labels=[f'{label_bins[i]:.2f}-\n{label_bins[i+1]:.2f}'
                           for i in range(n_bins)],
                    patch_artist=True)

    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, n_bins))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Target Value Range', fontsize=12)
    ax.set_ylabel('Average Gate Value (α)', fontsize=12)
    ax.set_title('(b) α Distribution by Target Range', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ α与目标值关系图已保存: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析和可视化门控值')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='jarvis/mbj_bandgap',
                       help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='分析的样本数')
    parser.add_argument('--output_dir', type=str, default='alpha_analysis',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🔍 门控值 (α) 分析工具")
    print("="*80)

    # 1. 加载模型
    print("\n📂 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model' in checkpoint:
        model_state = checkpoint['model']
        config = checkpoint.get('config', None)
    else:
        model_state = checkpoint
        config = None

    model = ALIGNN(config) if config else ALIGNN()
    model.load_state_dict(model_state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"✅ 模型已加载到 {device}")

    # 检查是否有中期融合
    if not hasattr(model, 'middle_fusion_modules') or len(model.middle_fusion_modules) == 0:
        print("\n❌ 错误：模型没有中期融合模块！")
        print("   请确保使用的是带有中期融合的SAGE-Net模型。")
        return

    print(f"✅ 检测到中期融合模块: {list(model.middle_fusion_modules.keys())}")

    # 2. 加载数据
    print(f"\n📊 加载数据集: {args.dataset}")
    dataset_name, target = args.dataset.split('/')

    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset=dataset_name,
        target=target,
        batch_size=args.batch_size,
        workers=0,
        pin_memory=False
    )

    print(f"✅ 测试集: {len(test_loader.dataset)} 样本")

    # 3. 提取门控值
    gate_values_dict = extract_all_gate_values(
        model, test_loader, device, n_samples=args.n_samples
    )

    if len(gate_values_dict['alphas']) == 0:
        print("\n❌ 未能提取到门控值！")
        return

    # 4. 生成可视化
    print("\n🎨 生成可视化图表...")

    # Figure 1: α分布
    visualize_alpha_distribution(
        gate_values_dict,
        output_dir / 'figure_alpha_distribution.pdf'
    )

    # Figure 2: 按元素分析
    element_stats = analyze_by_element(
        gate_values_dict,
        output_dir / 'figure_alpha_by_element.pdf'
    )

    # Figure 3: α vs 目标值
    analyze_alpha_vs_target(
        gate_values_dict,
        output_dir / 'figure_alpha_vs_target.pdf'
    )

    # Figure 4: 特定材料的热图
    print("\n🔥 生成材料热图...")
    create_alpha_heatmap_for_materials(
        model, test_loader, device,
        material_indices=[0, 5, 10],
        save_path=output_dir / 'figure_alpha_heatmap_materials.pdf'
    )

    # 5. 保存统计数据
    print("\n💾 保存统计数据...")

    stats_df = pd.DataFrame([
        {
            'element': elem,
            'mean_alpha': stats['mean'],
            'std_alpha': stats['std'],
            'count': stats['count']
        }
        for elem, stats in element_stats.items()
    ])
    stats_df.to_csv(output_dir / 'element_alpha_statistics.csv', index=False)

    print(f"✅ 统计数据已保存: {output_dir / 'element_alpha_statistics.csv'}")

    print("\n" + "="*80)
    print("✅ 分析完成！所有图表已保存到:", output_dir)
    print("="*80)

    print("\n📊 生成的图表:")
    print(f"  1. figure_alpha_distribution.pdf - α值整体分布")
    print(f"  2. figure_alpha_by_element.pdf - 按元素类型分析")
    print(f"  3. figure_alpha_vs_target.pdf - α值与目标性质关系")
    print(f"  4. figure_alpha_heatmap_materials.pdf - 材料热图")
    print(f"  5. element_alpha_statistics.csv - 元素统计数据")


if __name__ == '__main__':
    main()
