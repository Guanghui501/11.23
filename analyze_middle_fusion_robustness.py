#!/usr/bin/env python3
"""
对比有/无Middle Fusion在不同遮挡率下的性能
重点展示100%遮挡时keep_keywords的差异
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
masking_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 全模块 (Middle + Cross + FG)
with_middle_random = [8.7791, 9.2461, 9.6275, 9.7521, 11.4404, 15.1314, 14.869, 15.4438, 14.5584, 16.2865, 9.9949]
with_middle_keep = [8.7791, 10.6084, 11.067, 11.1887, 11.5935, 12.2777, 13.1226, 13.9111, 15.5857, 15.9732, 9.9949]

# 无中期融合 (Cross + FG only)
without_middle_random = [9.1038, 9.4398, 9.7458, 9.7559, 9.9645, 12.1416, 13.1709, 12.6573, 13.08, 13.8607, 9.9307]
without_middle_keep = [9.1038, 10.0099, 10.9102, 10.9739, 10.9394, 11.0332, 11.1649, 11.1987, 12.318, 13.1201, 12.318]  # 注意100%时12.318

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: Random Token策略对比
axes[0].plot(masking_ratios, with_middle_random, 'o-', linewidth=2, markersize=8,
             label='With Middle Fusion', color='#2E86AB')
axes[0].plot(masking_ratios, without_middle_random, 's-', linewidth=2, markersize=8,
             label='Without Middle Fusion', color='#A23B72')
axes[0].set_xlabel('Text Masking Ratio (%)', fontsize=12)
axes[0].set_ylabel('MAE', fontsize=12)
axes[0].set_title('Random Token Masking Strategy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(8, 18)

# 图2: Keep Keywords策略对比
axes[1].plot(masking_ratios, with_middle_keep, 'o-', linewidth=2, markersize=8,
             label='With Middle Fusion', color='#2E86AB')
axes[1].plot(masking_ratios, without_middle_keep, 's-', linewidth=2, markersize=8,
             label='Without Middle Fusion', color='#A23B72')
axes[1].set_xlabel('Text Masking Ratio (%)', fontsize=12)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Keep Keywords Strategy', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(8, 18)

# 高亮100%遮挡时的差异
axes[1].annotate('Huge gap at 100%!\n12.318 vs 9.995',
                xy=(100, 12.318), xytext=(80, 15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 图3: 100%遮挡时所有策略的对比
strategies = ['random\ntoken', 'random\nword', 'random\nchunk', 'sentence', 'keep\nkeywords']
with_middle_100 = [9.9949, 9.9949, 9.9949, 9.9949, 9.9949]
without_middle_100 = [9.9307, 9.9307, 9.9307, 10.0112, 12.318]

x = np.arange(len(strategies))
width = 0.35

bars1 = axes[2].bar(x - width/2, with_middle_100, width, label='With Middle',
                    color='#2E86AB', alpha=0.8)
bars2 = axes[2].bar(x + width/2, without_middle_100, width, label='Without Middle',
                    color='#A23B72', alpha=0.8)

axes[2].set_xlabel('Masking Strategy', fontsize=12)
axes[2].set_ylabel('MAE', fontsize=12)
axes[2].set_title('100% Text Masking: All Strategies', fontsize=14, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(strategies, fontsize=9)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_ylim(9, 13)

# 高亮keep_keywords的差异
axes[2].annotate(f'+23.9%\nworse!',
                xy=(4 + width/2, 12.318), xytext=(3.5, 12.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# 添加水平线显示基线
axes[2].axhline(y=9.9949, color='green', linestyle='--', linewidth=1, alpha=0.5,
               label='With Middle baseline')

plt.tight_layout()
plt.savefig('middle_fusion_robustness_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存: middle_fusion_robustness_analysis.png")

# 打印统计信息
print("\n" + "="*80)
print("关键发现")
print("="*80)
print(f"\n1. 100%遮挡时的MAE:")
print(f"   有Middle fusion - random_token: {with_middle_random[-1]:.4f}")
print(f"   有Middle fusion - keep_keywords: {with_middle_keep[-1]:.4f}")
print(f"   无Middle fusion - random_token: {without_middle_random[-1]:.4f}")
print(f"   无Middle fusion - keep_keywords: {without_middle_keep[-1]:.4f}")
print(f"\n2. keep_keywords在100%遮挡时的差异:")
print(f"   有Middle: {with_middle_keep[-1]:.4f}")
print(f"   无Middle: {without_middle_keep[-1]:.4f}")
print(f"   差距: {(without_middle_keep[-1] - with_middle_keep[-1]) / with_middle_keep[-1] * 100:.1f}%")
print(f"\n3. Middle fusion的鲁棒性优势:")
print(f"   有Middle时，100%遮挡下所有策略MAE完全相同")
print(f"   无Middle时，keep_keywords比其他策略差23.9%")
print("\n" + "="*80)
