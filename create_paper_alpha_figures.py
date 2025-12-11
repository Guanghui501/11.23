#!/usr/bin/env python
"""
生成论文级别的α值可视化图表

专门用于论文投稿的高质量图表生成。

生成的图表:
- Figure: Gate Value Analysis (4个子图)
  (a) α值分布直方图
  (b) 按元素类型的α值（柱状图）
  (c) α值热图（3个代表性材料）
  (d) α与目标性质的关系

用法:
    python create_paper_alpha_figures.py \
        --alpha_file alpha_values.npz \
        --output figure_gate_analysis.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns


def create_comprehensive_alpha_figure(alpha_file, output_path):
    """
    创建综合的α值分析图（论文用）

    布局: 2x2子图
    (a) α分布直方图
    (b) 按元素类型的平均α值
    (c) 代表性材料的α热图
    (d) α vs 目标性质
    """

    # 加载数据
    print(f"📂 加载数据: {alpha_file}")
    data = np.load(alpha_file, allow_pickle=True)

    alphas = data['alphas']  # List of arrays
    labels = data['labels']  # Array
    atom_types = data['atom_types']  # List of arrays

    # 合并所有α值
    all_alphas = np.concatenate(alphas)

    # 创建画布
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ========== (a) α分布直方图 ==========
    ax_a = fig.add_subplot(gs[0, 0])

    counts, bins, patches = ax_a.hist(all_alphas, bins=60,
                                      color='#3498db', alpha=0.7,
                                      edgecolor='black', linewidth=1)

    # 颜色梯度
    for i, patch in enumerate(patches):
        color = plt.cm.RdYlGn_r((bins[i] + bins[i+1]) / 2)
        patch.set_facecolor(color)

    # 统计线
    mean_alpha = all_alphas.mean()
    median_alpha = np.median(all_alphas)

    ax_a.axvline(mean_alpha, color='#e74c3c', linestyle='--',
                linewidth=2.5, label=f'Mean: {mean_alpha:.3f}')
    ax_a.axvline(median_alpha, color='#f39c12', linestyle='--',
                linewidth=2.5, label=f'Median: {median_alpha:.3f}')

    # 填充区域
    ax_a.axvspan(0, 0.3, alpha=0.1, color='red', label='Text-heavy')
    ax_a.axvspan(0.7, 1, alpha=0.1, color='green', label='Graph-heavy')

    ax_a.set_xlabel('Gate Value (α)', fontsize=14, fontweight='bold')
    ax_a.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax_a.set_title('(a) Gate Value Distribution',
                  fontsize=15, fontweight='bold', pad=10)
    ax_a.legend(fontsize=11, loc='upper right')
    ax_a.grid(True, alpha=0.3, linestyle='--')

    # 添加统计框
    stats_text = (f'N = {len(all_alphas):,}\n'
                 f'Std = {all_alphas.std():.3f}\n'
                 f'Min = {all_alphas.min():.3f}\n'
                 f'Max = {all_alphas.max():.3f}')
    ax_a.text(0.02, 0.98, stats_text, transform=ax_a.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== (b) 按元素类型 ==========
    ax_b = fig.add_subplot(gs[0, 1])

    # 元素列表
    element_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

    # 收集每个元素的α值
    element_alphas = {elem: [] for elem in element_names}

    for sample_alphas, sample_atoms in zip(alphas, atom_types):
        for alpha, atom_type in zip(sample_alphas, sample_atoms):
            elem_id = int(atom_type[0] if hasattr(atom_type, '__len__') else atom_type)
            if elem_id < len(element_names):
                element_alphas[element_names[elem_id]].append(alpha)

    # 过滤并排序
    valid_elements = {k: v for k, v in element_alphas.items() if len(v) > 20}
    element_stats = {
        elem: {'mean': np.mean(vals), 'std': np.std(vals), 'count': len(vals)}
        for elem, vals in valid_elements.items()
    }

    sorted_elements = sorted(element_stats.items(), key=lambda x: x[1]['mean'])

    # 只显示前20个
    top_20 = sorted_elements[:20]

    elements = [elem for elem, _ in top_20]
    means = [stats['mean'] for _, stats in top_20]
    stds = [stats['std'] for _, stats in top_20]

    # 颜色编码
    colors = plt.cm.RdYlGn_r(np.array(means))

    bars = ax_b.barh(range(len(elements)), means, xerr=stds,
                    color=colors, edgecolor='black', alpha=0.8,
                    error_kw={'linewidth': 1.5, 'ecolor': 'black'})

    ax_b.set_yticks(range(len(elements)))
    ax_b.set_yticklabels(elements, fontsize=11)
    ax_b.set_xlabel('Mean Gate Value (α)', fontsize=14, fontweight='bold')
    ax_b.set_title('(b) Gate Value by Element Type',
                  fontsize=15, fontweight='bold', pad=10)
    ax_b.axvline(mean_alpha, color='blue', linestyle='--',
                linewidth=2, alpha=0.7, label='Overall Mean')
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 添加颜色说明
    ax_b.text(0.98, 0.02, 'Red = Text-heavy\nGreen = Graph-heavy',
             transform=ax_b.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== (c) 材料热图 ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # 选择3个代表性材料
    # 选择策略: 低α(text-heavy), 中α(balanced), 高α(graph-heavy)
    avg_alphas = np.array([np.mean(a) for a in alphas])

    # 找到代表性样本
    low_idx = np.argmin(avg_alphas)
    high_idx = np.argmax(avg_alphas)
    mid_idx = np.argsort(np.abs(avg_alphas - 0.5))[0]

    selected_indices = [low_idx, mid_idx, high_idx]
    selected_materials = [alphas[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]
    selected_atoms = [atom_types[i] for i in selected_indices]

    # 创建热图矩阵（并排3个材料）
    max_atoms = max(len(m) for m in selected_materials)

    heatmap_data = np.full((max_atoms, 3), np.nan)

    for col, mat_alphas in enumerate(selected_materials):
        heatmap_data[:len(mat_alphas), col] = mat_alphas

    # 绘制热图
    im = ax_c.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto',
                    vmin=0, vmax=1, interpolation='nearest')

    # 设置刻度
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels([
        f'Material A\n(α={avg_alphas[low_idx]:.2f})\nText-heavy',
        f'Material B\n(α={avg_alphas[mid_idx]:.2f})\nBalanced',
        f'Material C\n(α={avg_alphas[high_idx]:.2f})\nGraph-heavy'
    ], fontsize=10)

    ax_c.set_ylabel('Atom Index', fontsize=14, fontweight='bold')
    ax_c.set_title('(c) Gate Value Heatmaps for Representative Materials',
                  fontsize=15, fontweight='bold', pad=10)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax_c, orientation='vertical', pad=0.02)
    cbar.set_label('α (Low=Text, High=Graph)', fontsize=11, fontweight='bold')

    # 标注元素类型（对于第一个材料）
    for i in range(len(selected_materials[0])):
        atom_id = int(selected_atoms[0][i][0] if hasattr(selected_atoms[0][i], '__len__')
                     else selected_atoms[0][i])
        if atom_id < len(element_names):
            elem = element_names[atom_id]
            ax_c.text(-0.5, i, elem, fontsize=8, ha='right', va='center')

    # ========== (d) α vs 目标性质 ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # 散点图
    scatter = ax_d.scatter(labels, avg_alphas, alpha=0.6, s=50,
                          c=avg_alphas, cmap='RdYlGn_r',
                          edgecolors='black', linewidth=0.5)

    # 拟合线
    z = np.polyfit(labels, avg_alphas, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(labels), max(labels), 100)
    ax_d.plot(x_line, p(x_line), 'r--', linewidth=2.5,
             label=f'Fit: α = {z[0]:.4f}·y + {z[1]:.3f}')

    # 相关性
    corr = np.corrcoef(labels, avg_alphas)[0, 1]

    ax_d.set_xlabel('Target Property Value', fontsize=14, fontweight='bold')
    ax_d.set_ylabel('Mean Gate Value (α)', fontsize=14, fontweight='bold')
    ax_d.set_title(f'(d) α vs Target Property (Correlation: {corr:.3f})',
                  fontsize=15, fontweight='bold', pad=10)
    ax_d.legend(fontsize=11, loc='best')
    ax_d.grid(True, alpha=0.3, linestyle='--')

    # 标注代表性点
    for idx, label_text in zip(selected_indices, ['A', 'B', 'C']):
        ax_d.scatter(labels[idx], avg_alphas[idx],
                    s=200, marker='*', color='gold',
                    edgecolors='black', linewidth=2, zorder=10)
        ax_d.annotate(label_text, (labels[idx], avg_alphas[idx]),
                     fontsize=14, fontweight='bold', ha='center', va='center')

    # 颜色条
    cbar_d = plt.colorbar(scatter, ax=ax_d, orientation='vertical', pad=0.02)
    cbar_d.set_label('Mean α', fontsize=11)

    # 总标题
    fig.suptitle('Gate Value (α) Analysis: Middle Fusion Mechanism',
                fontsize=18, fontweight='bold', y=0.995)

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 论文级别图表已保存: {output_path}")

    plt.close()


def create_simple_heatmap_grid(alpha_file, output_path, n_materials=6):
    """
    创建简单的材料热图网格（用于论文补充材料）

    展示6个不同材料的α值分布
    """

    # 加载数据
    data = np.load(alpha_file, allow_pickle=True)
    alphas = data['alphas']
    labels = data['labels']
    atom_types = data['atom_types']

    element_names = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                     'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                     'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    # 选择材料（分布在α值范围内）
    avg_alphas = np.array([np.mean(a) for a in alphas])
    percentiles = np.percentile(avg_alphas, np.linspace(0, 100, n_materials+2)[1:-1])

    selected_indices = []
    for pct in percentiles:
        idx = np.argmin(np.abs(avg_alphas - pct))
        selected_indices.append(idx)

    # 创建网格
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        ax = axes[i]

        mat_alphas = alphas[idx]
        mat_atoms = atom_types[idx]
        mat_label = labels[idx]

        # 准备数据
        n_atoms = len(mat_alphas)
        alpha_col = mat_alphas.reshape(-1, 1)

        # 获取元素名
        atom_labels = []
        for atom in mat_atoms:
            elem_id = int(atom[0] if hasattr(atom, '__len__') else atom)
            if elem_id < len(element_names):
                atom_labels.append(element_names[elem_id])
            else:
                atom_labels.append(f'X{elem_id}')

        # 绘制
        im = ax.imshow(alpha_col, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')

        # 刻度
        ax.set_yticks(range(n_atoms))
        ax.set_yticklabels([f'{j}: {label}' for j, label in enumerate(atom_labels)],
                          fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels(['α'])

        # 标注数值
        for j in range(n_atoms):
            ax.text(0, j, f'{mat_alphas[j]:.2f}',
                   ha="center", va="center",
                   color="black", fontsize=9, fontweight='bold')

        # 标题
        avg_alpha = np.mean(mat_alphas)
        ax.set_title(f'Material {i+1}\nAvg α={avg_alpha:.3f}, Target={mat_label:.3f}',
                    fontsize=11, fontweight='bold')

        # 颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.1, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 材料热图网格已保存: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成论文α值图表')
    parser.add_argument('--alpha_file', type=str, required=True,
                       help='α值数据文件 (.npz)')
    parser.add_argument('--output', type=str, default='figure_gate_analysis.pdf',
                       help='输出文件路径')
    parser.add_argument('--heatmap_grid', action='store_true',
                       help='额外生成热图网格')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🎨 生成论文级别α值图表")
    print("="*70)

    # 主图
    create_comprehensive_alpha_figure(args.alpha_file, args.output)

    # 热图网格
    if args.heatmap_grid:
        heatmap_path = args.output.replace('.pdf', '_heatmap_grid.pdf')
        create_simple_heatmap_grid(args.alpha_file, heatmap_path)

    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)


if __name__ == '__main__':
    main()
