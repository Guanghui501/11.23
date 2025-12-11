#!/usr/bin/env python
"""
生成 Figure 2: Broadcast Gated Fusion Mechanism 详细图

这个脚本可以生成高质量的PDF/PNG版本，适合论文投稿。
支持自定义颜色、字体大小等参数。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines
import numpy as np

def create_figure2(save_path="figure2_broadcast_gated_fusion.pdf", dpi=300):
    """
    创建 Broadcast Gated Fusion 机制详细图

    Args:
        save_path: 保存路径（支持 .pdf, .png, .svg）
        dpi: 分辨率
    """

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义颜色方案
    colors = {
        'graph': '#e8f5e9',      # 浅绿（图特征）
        'text': '#ffebee',       # 浅红（文本特征）
        'operation': '#f3e5f5',  # 浅紫（操作）
        'highlight': '#fff9c4',  # 浅黄（高亮/核心创新）
        'output': '#e3f2fd',     # 浅蓝（输出）
    }

    edge_colors = {
        'graph': '#388e3c',
        'text': '#c62828',
        'operation': '#7b1fa2',
        'highlight': '#f57c00',
        'output': '#1976d2',
    }

    # ==================== 标题 ====================
    ax.text(7, 9.5, 'Broadcast Gated Fusion Mechanism (Middle Fusion)',
            fontsize=18, fontweight='bold', ha='center', va='top')

    # ==================== INPUT LAYER (顶部) ====================
    y_input = 8.5

    # Node Features Input
    node_box = FancyBboxPatch((0.5, y_input), 2.5, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor=edge_colors['graph'],
                              facecolor=colors['graph'],
                              linewidth=2)
    ax.add_patch(node_box)
    ax.text(1.75, y_input + 0.75, 'Node Features', fontsize=12,
            fontweight='bold', ha='center', va='center')
    ax.text(1.75, y_input + 0.45, r'$H_{node}^{(L)} \in \mathbb{R}^{N \times 256}$',
            fontsize=11, ha='center', va='center', style='italic')
    ax.text(1.75, y_input + 0.15, 'N atoms, 256-dim', fontsize=9,
            ha='center', va='center', color='#666')

    # Text Semantic Input
    text_box = FancyBboxPatch((4, y_input), 2.5, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor=edge_colors['text'],
                             facecolor=colors['text'],
                             linewidth=2)
    ax.add_patch(text_box)
    ax.text(5.25, y_input + 0.75, 'Text Semantic', fontsize=12,
            fontweight='bold', ha='center', va='center')
    ax.text(5.25, y_input + 0.45, r'$T_{cls} \in \mathbb{R}^{768}$',
            fontsize=11, ha='center', va='center', style='italic')
    ax.text(5.25, y_input + 0.15, 'Global text (CLS token)', fontsize=9,
            ha='center', va='center', color='#666')

    # ==================== STEP 1: TEXT PROJECTION ====================
    y_proj = 6.8

    ax.text(0.3, 7.3, 'STEP 1:', fontsize=13, fontweight='bold', color='#d32f2f')

    # Arrow from text input to projection
    arrow1 = FancyArrowPatch((5.25, y_input), (5.25, y_proj + 1.2),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='#666')
    ax.add_patch(arrow1)

    # Projection operation box
    proj_box = FancyBboxPatch((4, y_proj), 2.5, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor=edge_colors['operation'],
                             facecolor=colors['operation'],
                             linewidth=2)
    ax.add_patch(proj_box)
    ax.text(5.25, y_proj + 0.75, 'Linear Projection', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(5.25, y_proj + 0.45, r'$T_{proj} = W_{proj} \cdot T_{cls}$',
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(5.25, y_proj + 0.15, r'$\mathbb{R}^{768} \rightarrow \mathbb{R}^{256}$',
            fontsize=9, ha='center', va='center', color='#666')

    # Projected output box
    proj_out_box = FancyBboxPatch((4.2, y_proj - 0.8), 2.1, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor=edge_colors['text'],
                                 facecolor=colors['text'],
                                 linewidth=1.5)
    ax.add_patch(proj_out_box)
    ax.text(5.25, y_proj - 0.5, r'$T_{proj} \in \mathbb{R}^{256}$',
            fontsize=10, ha='center', va='center', style='italic')

    # Arrow
    arrow2 = FancyArrowPatch((5.25, y_proj), (5.25, y_proj - 0.2),
                            arrowstyle='->', mutation_scale=15, linewidth=2,
                            color='#666')
    ax.add_patch(arrow2)

    # ==================== STEP 2: BROADCAST ====================
    y_broadcast = 5.0

    ax.text(0.3, 5.5, 'STEP 2:', fontsize=13, fontweight='bold', color='#d32f2f')

    # Arrow from projection to broadcast (thick)
    arrow3 = FancyArrowPatch((5.25, y_proj - 0.8), (5.25, y_broadcast + 1.0),
                            arrowstyle='->', mutation_scale=25, linewidth=3,
                            color='#ff6f00')
    ax.add_patch(arrow3)

    # Broadcast operation (highlighted)
    broadcast_box = FancyBboxPatch((3.5, y_broadcast), 3.5, 0.9,
                                  boxstyle="round,pad=0.1",
                                  edgecolor=edge_colors['highlight'],
                                  facecolor=colors['highlight'],
                                  linewidth=2.5)
    ax.add_patch(broadcast_box)
    ax.text(5.25, y_broadcast + 0.65, 'Broadcast (Repeat N times)', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(5.25, y_broadcast + 0.35, r'$T_{broadcast} = \mathrm{Repeat}(T_{proj}, N)$',
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(5.25, y_broadcast + 0.08, r'$\mathbb{R}^{256} \rightarrow \mathbb{R}^{N \times 256}$',
            fontsize=9, ha='center', va='center', color='#666')

    # Arrow from node features (bypass)
    arrow4 = FancyArrowPatch((1.75, y_input), (1.75, 3.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='#666')
    ax.add_patch(arrow4)

    # Concatenation node
    concat_circle = Circle((3.2, 3.8), 0.2, edgecolor='#666',
                          facecolor='white', linewidth=2)
    ax.add_patch(concat_circle)
    ax.text(3.2, 3.8, '⊕', fontsize=16, ha='center', va='center', fontweight='bold')

    # Arrows to concatenation
    arrow5 = FancyArrowPatch((5.25, y_broadcast), (3.4, 3.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#ff6f00')
    ax.add_patch(arrow5)

    arrow6 = FancyArrowPatch((1.75, 3.8), (3.0, 3.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='#666')
    ax.add_patch(arrow6)

    # ==================== STEP 3: GATED FUSION ====================
    y_gate = 2.3

    ax.text(0.3, 3.3, 'STEP 3:', fontsize=13, fontweight='bold', color='#d32f2f')

    # Arrow from concat to gating
    arrow7 = FancyArrowPatch((3.2, 3.6), (3.2, y_gate + 1.0),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='#666')
    ax.add_patch(arrow7)

    # Gate computation box
    gate_box = FancyBboxPatch((0.5, y_gate), 3, 0.9,
                             boxstyle="round,pad=0.1",
                             edgecolor=edge_colors['highlight'],
                             facecolor=colors['highlight'],
                             linewidth=2.5)
    ax.add_patch(gate_box)
    ax.text(2, y_gate + 0.65, 'Gate Computation', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(2, y_gate + 0.35, r'$\alpha_i = \sigma(W_{gate} \cdot [h_i \oplus T_{proj}])$',
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(2, y_gate + 0.08, r'$\alpha_i \in [0, 1]$ (adaptive weight)',
            fontsize=9, ha='center', va='center', color='#666')

    # Fusion operation box
    fusion_box = FancyBboxPatch((4, y_gate), 3, 0.9,
                               boxstyle="round,pad=0.1",
                               edgecolor=edge_colors['highlight'],
                               facecolor=colors['highlight'],
                               linewidth=2.5)
    ax.add_patch(fusion_box)
    ax.text(5.5, y_gate + 0.65, 'Weighted Fusion', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(5.5, y_gate + 0.35, r"$h'_i = \alpha_i \cdot h_i + (1-\alpha_i) \cdot T_{proj}$",
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(5.5, y_gate + 0.08, 'Balance graph & text features',
            fontsize=9, ha='center', va='center', color='#666')

    # Arrow connecting gate and fusion
    arrow8 = FancyArrowPatch((3.5, y_gate + 0.45), (4.0, y_gate + 0.45),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#ff6f00')
    ax.add_patch(arrow8)

    # ==================== STEP 4: LAYER NORM ====================
    y_norm = 0.8

    ax.text(8, 3.3, 'STEP 4:', fontsize=13, fontweight='bold', color='#d32f2f')

    # Arrow from fusion to norm
    arrow9 = FancyArrowPatch((7, y_gate + 0.45), (9, y_norm + 0.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='#666')
    ax.add_patch(arrow9)

    # Layer normalization box
    norm_box = FancyBboxPatch((8, y_norm), 3.5, 0.9,
                             boxstyle="round,pad=0.1",
                             edgecolor=edge_colors['operation'],
                             facecolor=colors['operation'],
                             linewidth=2)
    ax.add_patch(norm_box)
    ax.text(9.75, y_norm + 0.65, 'Layer Norm + Residual', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(9.75, y_norm + 0.35, r"$H_{output} = \mathrm{LayerNorm}(h' + h)$",
            fontsize=10, ha='center', va='center', style='italic')
    ax.text(9.75, y_norm + 0.08, 'Preserve info & stabilize',
            fontsize=9, ha='center', va='center', color='#666')

    # Residual connection (dashed arrow)
    arrow10 = FancyArrowPatch((1.75, 3.5), (8.5, 1.7),
                             arrowstyle='->', mutation_scale=15, linewidth=1.5,
                             color='#666', linestyle='dashed')
    ax.add_patch(arrow10)
    ax.text(5, 2.6, 'Residual', fontsize=9, color='#666', style='italic')

    # ==================== OUTPUT ====================
    y_output = 0.1

    # Arrow to output
    arrow11 = FancyArrowPatch((9.75, y_norm), (9.75, y_output + 0.6),
                             arrowstyle='->', mutation_scale=25, linewidth=3,
                             color='#ff6f00')
    ax.add_patch(arrow11)

    # Output box
    output_box = FancyBboxPatch((8, y_output), 3.5, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor=edge_colors['graph'],
                               facecolor=colors['graph'],
                               linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(9.75, y_output + 0.4, 'Semantically-Guided Features', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(9.75, y_output + 0.1, r'$H_{guided}^{(L+1)} \in \mathbb{R}^{N \times 256}$',
            fontsize=10, ha='center', va='center', style='italic')

    # ==================== ANNOTATIONS ====================

    # Key Innovation Box (右上角)
    innovation_box = FancyBboxPatch((8.5, 7.5), 4.8, 1.8,
                                   boxstyle="round,pad=0.15",
                                   edgecolor='#e65100',
                                   facecolor='#fff3e0',
                                   linewidth=2)
    ax.add_patch(innovation_box)
    ax.text(10.9, 9.1, '🔑 Key Innovation', fontsize=12,
            fontweight='bold', ha='center', va='center', color='#e65100')
    ax.text(8.7, 8.7, '✓ Inject semantics during encoding', fontsize=9.5,
            ha='left', va='center')
    ax.text(8.7, 8.4, '✓ Adaptive per-node gating (α)', fontsize=9.5,
            ha='left', va='center')
    ax.text(8.7, 8.1, '✓ Preserve graph structure', fontsize=9.5,
            ha='left', va='center')
    ax.text(8.7, 7.8, '✓ Better than late fusion!', fontsize=9.5,
            ha='left', va='center', color='#d32f2f', fontweight='bold')

    # Example values box
    example_box = FancyBboxPatch((0.5, 4.5), 2.8, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='#00838f',
                                facecolor='#e0f7fa',
                                linewidth=1.5)
    ax.add_patch(example_box)
    ax.text(1.9, 5.15, 'Example:', fontsize=10,
            fontweight='bold', ha='center', va='center')
    ax.text(1.9, 4.9, r'SiO$_2$: α ≈ 0.7 (high text)', fontsize=8.5,
            ha='center', va='center')
    ax.text(1.9, 4.65, 'Alloy: α ≈ 0.3 (low text)', fontsize=8.5,
            ha='center', va='center')

    # Step labels on left
    ax.text(0, y_input + 0.5, 'INPUT', fontsize=11, fontweight='bold',
            color='#1976d2', rotation=0)
    ax.text(0, y_output + 0.3, 'OUTPUT', fontsize=11, fontweight='bold',
            color='#1976d2', rotation=0)

    # 调整布局
    plt.tight_layout()

    # 保存
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Figure 2 saved to: {save_path}")

    return fig, ax


def create_simplified_version(save_path="figure2_simple.pdf", dpi=300):
    """
    创建简化版本（用于演示文稿）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 简化的流程图
    # Node Features
    node_box = FancyBboxPatch((0.5, 3.5), 2, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='#388e3c', facecolor='#e8f5e9',
                             linewidth=2)
    ax.add_patch(node_box)
    ax.text(1.5, 4.5, 'Graph\nFeatures', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(1.5, 3.8, r'$H \in \mathbb{R}^{N \times 256}$',
            fontsize=9, ha='center', va='center', style='italic')

    # Text Semantic
    text_box = FancyBboxPatch((0.5, 1), 2, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='#c62828', facecolor='#ffebee',
                             linewidth=2)
    ax.add_patch(text_box)
    ax.text(1.5, 2, 'Text\nSemantic', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(1.5, 1.3, r'$T \in \mathbb{R}^{768}$',
            fontsize=9, ha='center', va='center', style='italic')

    # Projection
    proj_arrow = FancyArrowPatch((2.5, 1.75), (4, 1.75),
                                arrowstyle='->', mutation_scale=20,
                                linewidth=2, color='#666')
    ax.add_patch(proj_arrow)
    ax.text(3.25, 2.1, 'Project', fontsize=9, ha='center')

    proj_box = FancyBboxPatch((4, 1.2), 1.5, 1.1,
                             boxstyle="round,pad=0.05",
                             edgecolor='#7b1fa2', facecolor='#f3e5f5',
                             linewidth=2)
    ax.add_patch(proj_box)
    ax.text(4.75, 1.75, r'$T_{256}$', fontsize=10,
            ha='center', va='center', style='italic')

    # Broadcast
    broadcast_arrow = FancyArrowPatch((5.5, 1.75), (6.5, 3),
                                     arrowstyle='->', mutation_scale=20,
                                     linewidth=2.5, color='#ff6f00')
    ax.add_patch(broadcast_arrow)
    ax.text(6, 2.2, 'Broadcast', fontsize=9, ha='center', color='#ff6f00',
            fontweight='bold')

    # Gate Fusion
    fusion_box = FancyBboxPatch((6.5, 2.5), 2.5, 2,
                               boxstyle="round,pad=0.1",
                               edgecolor='#f57c00', facecolor='#fff9c4',
                               linewidth=2.5)
    ax.add_patch(fusion_box)
    ax.text(7.75, 4.1, 'Gated Fusion', fontsize=11,
            fontweight='bold', ha='center', va='center')
    ax.text(7.75, 3.6, r'$\alpha = \sigma(W \cdot [H;T])$', fontsize=9,
            ha='center', va='center', style='italic')
    ax.text(7.75, 3.1, r"$H' = \alpha H + (1-\alpha) T$", fontsize=9,
            ha='center', va='center', style='italic')

    # Graph to fusion
    graph_arrow = FancyArrowPatch((2.5, 4.25), (6.5, 3.5),
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=2, color='#666')
    ax.add_patch(graph_arrow)

    # Output
    output_arrow = FancyArrowPatch((9, 3.5), (10.5, 3.5),
                                  arrowstyle='->', mutation_scale=25,
                                  linewidth=3, color='#ff6f00')
    ax.add_patch(output_arrow)

    output_box = FancyBboxPatch((10.5, 2.8), 1.3, 1.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='#388e3c', facecolor='#e8f5e9',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(11.15, 3.5, r"$H'$", fontsize=12,
            fontweight='bold', ha='center', va='center')

    # Title
    ax.text(6, 5.5, 'Middle Fusion: Broadcast Gated Mechanism',
            fontsize=14, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✅ Simplified Figure 2 saved to: {save_path}")

    return fig, ax


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Figure 2 for paper')
    parser.add_argument('--output', type=str, default='figure2_broadcast_gated_fusion.pdf',
                       help='Output file path (.pdf, .png, .svg)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster formats')
    parser.add_argument('--simplified', action='store_true',
                       help='Generate simplified version for presentations')

    args = parser.parse_args()

    print("🎨 Generating Figure 2: Broadcast Gated Fusion Mechanism...")

    if args.simplified:
        create_simplified_version(args.output, args.dpi)
    else:
        create_figure2(args.output, args.dpi)

    print("✅ Done! Use this figure in your paper.")
