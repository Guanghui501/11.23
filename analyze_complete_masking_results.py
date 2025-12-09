#!/usr/bin/env python3
"""
对比有/无Middle Fusion在新数据集上的完整分析
重点发现：
1. random_token策略在50-70%时崩溃（两种架构都崩溃）
2. 100%遮挡时有Middle反而更差
3. sentence策略表现最鲁棒
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
masking_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 无中期融合数据
without_middle = {
    'random_token': {
        'mae': [0.2742, 0.3126, 0.3254, 0.3511, 0.4600, 3.5030, 3.4241, 2.3910, 1.4776, 0.8508, 0.7820],
        'r2': [0.8877, 0.8552, 0.8541, 0.8370, 0.7469, -4.5588, -4.0726, -1.8942, -0.4092, 0.3445, 0.6045]
    },
    'random_word': {
        'mae': [0.2742, 0.2956, 0.3216, 0.3223, 0.3329, 0.3258, 0.4243, 2.5791, 2.3149, 0.9317, 0.7820],
        'r2': [0.8877, 0.8790, 0.8284, 0.8641, 0.8469, 0.8600, 0.7607, -2.5054, -1.8275, 0.3353, 0.6045]
    },
    'random_chunk': {
        'mae': [0.2742, 0.2841, 0.3007, 0.3164, 0.3227, 0.3156, 0.3847, 0.9366, 1.3908, 1.1256, 0.7820],
        'r2': [0.8877, 0.8820, 0.8573, 0.8591, 0.8477, 0.8730, 0.7979, -0.0357, -0.7932, -0.2629, 0.6045]
    },
    'sentence': {
        'mae': [0.2742, 0.2744, 0.2800, 0.2918, 0.2983, 0.3164, 0.3234, 0.3305, 0.3577, 0.3642, 0.7820],
        'r2': [0.8877, 0.8877, 0.8852, 0.8712, 0.8768, 0.8545, 0.8474, 0.8479, 0.8399, 0.8353, 0.6045]
    },
    'keep_keywords': {
        'mae': [0.2742, 0.3717, 0.3776, 0.3529, 0.3421, 0.3418, 0.3535, 0.4809, 1.1316, 0.8989, 0.7820],
        'r2': [0.8877, 0.8347, 0.8287, 0.8611, 0.8704, 0.8617, 0.8449, 0.7095, 0.1329, 0.3821, 0.6045]
    }
}

# 有中期融合数据
with_middle = {
    'random_token': {
        'mae': [0.2514, 0.2933, 0.2798, 0.3068, 0.3919, 2.8649, 3.1381, 2.5509, 2.4038, 2.4098, 1.9334],
        'r2': [0.9148, 0.8230, 0.8882, 0.7916, 0.7932, -3.7573, -4.4690, -3.3775, -2.9663, -2.7330, -1.3321]
    },
    'random_word': {
        'mae': [0.2514, 0.2684, 0.2833, 0.2793, 0.2976, 0.2815, 0.3791, 1.5764, 2.3325, 1.9335, 1.9334],
        'r2': [0.9148, 0.9104, 0.8744, 0.8894, 0.8544, 0.8918, 0.7441, -1.0805, -2.6042, -1.7168, -1.3321]
    },
    'random_chunk': {
        'mae': [0.2514, 0.2721, 0.2958, 0.3068, 0.3124, 0.3256, 0.3491, 0.4006, 0.6564, 0.7716, 1.9334],
        'r2': [0.9148, 0.8850, 0.8395, 0.8481, 0.8771, 0.8794, 0.8381, 0.7711, 0.3475, 0.1831, -1.3321]
    },
    'sentence': {
        'mae': [0.2514, 0.2522, 0.2577, 0.2729, 0.2686, 0.2876, 0.3013, 0.2976, 0.3073, 0.3293, 1.9334],
        'r2': [0.9148, 0.9148, 0.9124, 0.8906, 0.9096, 0.8717, 0.8351, 0.8680, 0.8701, 0.8372, -1.3321]
    },
    'keep_keywords': {
        'mae': [0.2514, 0.3125, 0.3302, 0.3232, 0.3284, 0.3417, 0.3676, 0.5138, 0.7768, 1.4206, 1.9334],
        'r2': [0.9148, 0.8852, 0.8537, 0.8711, 0.8455, 0.8454, 0.8371, 0.6127, 0.4182, -0.5297, -1.3321]
    }
}

# 创建综合对比图
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

strategies = ['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords']
colors = ['#E63946', '#F77F00', '#06AED5', '#2A9D8F', '#E9C46A']

# 图1-5: 各策略MAE对比
for idx, strategy in enumerate(strategies):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])

    without_mae = without_middle[strategy]['mae']
    with_mae = with_middle[strategy]['mae']

    ax.plot(masking_ratios, without_mae, 'o-', linewidth=2, markersize=6,
            label='Without Middle', color='#A23B72', alpha=0.8)
    ax.plot(masking_ratios, with_mae, 's-', linewidth=2, markersize=6,
            label='With Middle', color='#2E86AB', alpha=0.8)

    ax.set_xlabel('Masking Ratio (%)', fontsize=10)
    ax.set_ylabel('MAE', fontsize=10)
    ax.set_title(f'{strategy.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 设置y轴范围（避免极端值）
    ax.set_ylim(0, min(4, max(max(without_mae), max(with_mae)) * 1.1))

    # 高亮崩溃区域
    if max(without_mae) > 2 or max(with_mae) > 2:
        ax.axhspan(2, 4, alpha=0.1, color='red')
        ax.text(50, 3.5, 'Collapse Zone', fontsize=9, color='red',
                alpha=0.7, ha='center', fontweight='bold')

# 图6: 100%遮挡对比
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(strategies))
width = 0.35

without_100 = [without_middle[s]['mae'][-1] for s in strategies]
with_100 = [with_middle[s]['mae'][-1] for s in strategies]

bars1 = ax6.bar(x - width/2, without_100, width, label='Without Middle',
                color='#A23B72', alpha=0.8)
bars2 = ax6.bar(x + width/2, with_100, width, label='With Middle',
                color='#2E86AB', alpha=0.8)

ax6.set_xlabel('Strategy', fontsize=10)
ax6.set_ylabel('MAE at 100% Masking', fontsize=10)
ax6.set_title('100% Masking: All Strategies', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# 图7: R²对比（sentence策略）
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(masking_ratios, without_middle['sentence']['r2'], 'o-', linewidth=2,
         label='Without Middle', color='#A23B72')
ax7.plot(masking_ratios, with_middle['sentence']['r2'], 's-', linewidth=2,
         label='With Middle', color='#2E86AB')
ax7.set_xlabel('Masking Ratio (%)', fontsize=10)
ax7.set_ylabel('R² Score', fontsize=10)
ax7.set_title('R² Score: Sentence Strategy (Most Robust)', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

# 图8: 崩溃区域统计
ax8 = fig.add_subplot(gs[2, 1])
collapse_ratios = [50, 60, 70, 80, 90]
without_collapses = []
with_collapses = []

for ratio_idx in [5, 6, 7, 8, 9]:  # 50-90%
    without_count = sum(1 for s in strategies if without_middle[s]['r2'][ratio_idx] < 0)
    with_count = sum(1 for s in strategies if with_middle[s]['r2'][ratio_idx] < 0)
    without_collapses.append(without_count)
    with_collapses.append(with_count)

x = np.arange(len(collapse_ratios))
ax8.bar(x - width/2, without_collapses, width, label='Without Middle',
        color='#A23B72', alpha=0.8)
ax8.bar(x + width/2, with_collapses, width, label='With Middle',
        color='#2E86AB', alpha=0.8)

ax8.set_xlabel('Masking Ratio (%)', fontsize=10)
ax8.set_ylabel('Number of Collapsed Strategies (R²<0)', fontsize=10)
ax8.set_title('Strategy Collapse Count', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(collapse_ratios)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis='y')
ax8.set_ylim(0, 5.5)

# 图9: 关键发现总结
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

findings = """
KEY FINDINGS

1. Baseline (0% masking):
   • With Middle: MAE=0.251 ✓ Better
   • Without Middle: MAE=0.274
   • Improvement: 8.3%

2. Critical Collapse (50-70%):
   • random_token FAILS on BOTH
   • MAE >2.5, R²<-3.0
   • Middle fusion does NOT prevent

3. 100% Masking Surprise:
   • Without Middle: MAE=0.78, R²=0.60 ✓
   • With Middle: MAE=1.93, R²=-1.33 ✗
   • Middle worse at extreme!

4. Most Robust Strategy:
   • sentence: Stable 0-80%
   • Best for both architectures
   • Graceful degradation

5. Architecture Recommendation:
   • Middle fusion: Better baseline
   • But: NOT more robust overall
   • Strategy choice matters more!
"""

ax9.text(0.05, 0.95, findings, transform=ax9.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('middle_fusion_complete_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 完整分析图已保存: middle_fusion_complete_analysis.png")

# 打印详细统计
print("\n" + "="*80)
print("详细分析报告")
print("="*80)

print("\n1. 基线性能 (0% 遮挡):")
print(f"   无Middle: MAE={without_middle['random_token']['mae'][0]:.4f}, R²={without_middle['random_token']['r2'][0]:.4f}")
print(f"   有Middle: MAE={with_middle['random_token']['mae'][0]:.4f}, R²={with_middle['random_token']['r2'][0]:.4f}")
print(f"   改进: {(without_middle['random_token']['mae'][0] - with_middle['random_token']['mae'][0]) / without_middle['random_token']['mae'][0] * 100:.1f}%")

print("\n2. random_token 崩溃分析 (50-70%):")
for ratio_idx, ratio in zip([5, 6, 7], [50, 60, 70]):
    print(f"   {ratio}%遮挡:")
    print(f"     无Middle: MAE={without_middle['random_token']['mae'][ratio_idx]:.4f}, R²={without_middle['random_token']['r2'][ratio_idx]:.4f}")
    print(f"     有Middle: MAE={with_middle['random_token']['mae'][ratio_idx]:.4f}, R²={with_middle['random_token']['r2'][ratio_idx]:.4f}")

print("\n3. 100% 遮挡对比:")
print(f"   无Middle: MAE={without_middle['random_token']['mae'][-1]:.4f}, R²={without_middle['random_token']['r2'][-1]:.4f}")
print(f"   有Middle: MAE={with_middle['random_token']['mae'][-1]:.4f}, R²={with_middle['random_token']['r2'][-1]:.4f}")
print(f"   差异: Middle比无Middle差 {(with_middle['random_token']['mae'][-1] - without_middle['random_token']['mae'][-1]) / without_middle['random_token']['mae'][-1] * 100:.1f}%")

print("\n4. 最鲁棒策略 (sentence):")
print("   0-80%遮挡范围:")
for ratio_idx, ratio in enumerate([0, 20, 40, 60, 80]):
    idx = ratio_idx * 2
    print(f"     {ratio}%: 无Middle MAE={without_middle['sentence']['mae'][idx]:.4f}, 有Middle MAE={with_middle['sentence']['mae'][idx]:.4f}")

print("\n5. 策略鲁棒性排名 (0-80%平均MAE):")
avg_maes = {}
for strategy in strategies:
    without_avg = np.mean(without_middle[strategy]['mae'][:9])  # 0-80%
    with_avg = np.mean(with_middle[strategy]['mae'][:9])
    avg_maes[strategy] = {'without': without_avg, 'with': with_avg}

sorted_strategies = sorted(avg_maes.items(), key=lambda x: x[1]['with'])
for rank, (strategy, maes) in enumerate(sorted_strategies, 1):
    print(f"   {rank}. {strategy}: 有Middle={maes['with']:.4f}, 无Middle={maes['without']:.4f}")

print("\n" + "="*80)
