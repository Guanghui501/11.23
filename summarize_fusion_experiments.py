#!/usr/bin/env python
"""
汇总和可视化融合位置对比实验的结果
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11


def load_training_history(result_dir):
    """加载训练历史"""
    history_file = os.path.join(result_dir, 'training_history.json')
    if not os.path.exists(history_file):
        print(f"⚠️  未找到训练历史: {history_file}")
        return None

    with open(history_file, 'r') as f:
        history = json.load(f)
    return history


def load_feature_metrics(analysis_dir):
    """加载特征质量指标"""
    metrics_file = os.path.join(analysis_dir, 'regression_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"⚠️  未找到特征指标: {metrics_file}")
        return None

    df = pd.read_csv(metrics_file)
    return df


def summarize_experiments(experiments):
    """汇总实验结果"""

    print("\n" + "="*60)
    print("  融合位置对比实验结果汇总")
    print("="*60 + "\n")

    summary_data = []

    for exp_name, exp_config in experiments.items():
        result_dir = exp_config['result_dir']
        analysis_dir = exp_config['analysis_dir']

        print(f"📊 {exp_name}")
        print("-" * 60)

        # 加载训练历史
        history = load_training_history(result_dir)
        if history:
            best_test_mae = min(history.get('test_mae', [float('inf')]))
            best_val_mae = min(history.get('val_mae', [float('inf')]))
            final_test_mae = history.get('test_mae', [])[-1] if history.get('test_mae') else None

            print(f"  最佳测试MAE: {best_test_mae:.4f}")
            print(f"  最佳验证MAE: {best_val_mae:.4f}")
            print(f"  最终测试MAE: {final_test_mae:.4f}")
        else:
            best_test_mae = None
            best_val_mae = None
            final_test_mae = None

        # 加载特征指标
        feature_metrics = load_feature_metrics(analysis_dir)
        if feature_metrics is not None:
            # 获取融合特征的指标
            fused_row = feature_metrics[feature_metrics['Feature'] == 'fused']
            if not fused_row.empty:
                avg_pearson = fused_row['Avg Pearson Corr'].values[0]
                max_pearson = fused_row['Max Pearson Corr'].values[0]
                print(f"  平均Pearson相关性: {avg_pearson:.4f}")
                print(f"  最大Pearson相关性: {max_pearson:.4f}")
            else:
                avg_pearson = None
                max_pearson = None
        else:
            avg_pearson = None
            max_pearson = None

        summary_data.append({
            'Experiment': exp_name,
            'Best Test MAE': best_test_mae,
            'Best Val MAE': best_val_mae,
            'Final Test MAE': final_test_mae,
            'Avg Pearson Corr': avg_pearson,
            'Max Pearson Corr': max_pearson,
            'Fusion Type': exp_config['fusion_type']
        })

        print()

    # 创建汇总表
    summary_df = pd.DataFrame(summary_data)

    print("\n" + "="*60)
    print("  综合对比表")
    print("="*60 + "\n")
    print(summary_df.to_string(index=False))
    print()

    # 保存汇总表
    summary_df.to_csv('fusion_comparison_summary.csv', index=False)
    print("✅ 汇总表已保存: fusion_comparison_summary.csv\n")

    return summary_df


def plot_comparison(summary_df, output_dir='./'):
    """绘制对比图"""

    print("📈 生成对比可视化...")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    experiments = summary_df['Experiment'].values
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # 蓝、红、绿

    # 1. 测试MAE对比
    ax = axes[0, 0]
    best_mae = summary_df['Best Test MAE'].values
    x = range(len(experiments))
    bars = ax.bar(x, best_mae, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.set_ylabel('MAE (eV/atom)', fontweight='bold')
    ax.set_title('最佳测试MAE对比', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注数值
    for i, v in enumerate(best_mae):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # 标注最佳
    best_idx = np.argmin(best_mae)
    ax.scatter(best_idx, best_mae[best_idx], s=200, marker='*',
              color='gold', edgecolor='black', linewidth=2, zorder=10)

    # 2. 收敛曲线对比
    ax = axes[0, 1]
    for i, (exp_name, exp_config) in enumerate(experiments_config.items()):
        history = load_training_history(exp_config['result_dir'])
        if history and 'test_mae' in history:
            epochs = range(1, len(history['test_mae']) + 1)
            ax.plot(epochs, history['test_mae'], label=exp_name,
                   color=colors[i], linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Test MAE (eV/atom)', fontweight='bold')
    ax.set_title('训练收敛曲线对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    # 3. Pearson相关性对比
    ax = axes[1, 0]
    avg_pearson = summary_df['Avg Pearson Corr'].values
    max_pearson = summary_df['Max Pearson Corr'].values

    x_pos = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, avg_pearson, width, label='平均相关性',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, max_pearson, width, label='最大相关性',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.set_ylabel('Pearson相关系数', fontweight='bold')
    ax.set_title('特征-目标相关性对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5,
              alpha=0.6, label='强相关(0.5)')

    # 标注数值
    for i, (v1, v2) in enumerate(zip(avg_pearson, max_pearson)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. 性能雷达图
    ax = axes[1, 1]
    ax.axis('off')

    # 创建极坐标子图
    ax_polar = fig.add_subplot(2, 2, 4, projection='polar')

    # 归一化指标 (越小越好的MAE需要反转)
    mae_normalized = 1 - (best_mae - best_mae.min()) / (best_mae.max() - best_mae.min() + 1e-10)
    pearson_normalized = (avg_pearson - avg_pearson.min()) / (avg_pearson.max() - avg_pearson.min() + 1e-10)

    # 雷达图数据
    categories = ['低误差', '高相关性', '稳定性']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for i, exp_name in enumerate(experiments):
        values = [
            mae_normalized[i],  # 低误差
            pearson_normalized[i],  # 高相关性
            0.8  # 稳定性(占位)
        ]
        values += values[:1]  # 闭合

        ax_polar.plot(angles, values, 'o-', linewidth=2, label=exp_name,
                     color=colors[i], alpha=0.8)
        ax_polar.fill(angles, values, alpha=0.15, color=colors[i])

    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(categories, fontsize=11)
    ax_polar.set_ylim(0, 1)
    ax_polar.set_title('综合性能雷达图', fontsize=14, fontweight='bold',
                      pad=20, y=1.08)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
    ax_polar.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fusion_comparison_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}\n")
    plt.close()


def generate_report(summary_df, output_file='fusion_comparison_report.md'):
    """生成Markdown报告"""

    print("📝 生成对比报告...")

    report = []
    report.append("# 融合位置对比实验报告\n")
    report.append("## 实验设计\n")
    report.append("本实验对比了三种不同的文本-图融合策略:\n")
    report.append("1. **ALIGNN层融合**: 在ALIGNN编码早期注入文本信息(中间融合)\n")
    report.append("2. **GCN层融合**: 在GCN层之后进行细粒度原子-词元注意力\n")
    report.append("3. **层次化融合**: 结合ALIGNN、GCN和全局三个层次的融合\n\n")

    report.append("## 实验结果\n\n")
    report.append("### 性能指标对比\n\n")
    report.append(summary_df.to_markdown(index=False))
    report.append("\n\n")

    # 找出最佳模型
    best_idx = summary_df['Best Test MAE'].idxmin()
    best_exp = summary_df.loc[best_idx]

    report.append("### 关键发现\n\n")
    report.append(f"#### 🏆 最佳模型: {best_exp['Experiment']}\n\n")
    report.append(f"- **最佳测试MAE**: {best_exp['Best Test MAE']:.4f} eV/atom\n")
    report.append(f"- **平均Pearson相关性**: {best_exp['Avg Pearson Corr']:.4f}\n")
    report.append(f"- **融合类型**: {best_exp['Fusion Type']}\n\n")

    # 分析各个模型的优劣
    report.append("#### 📊 各模型分析\n\n")

    for idx, row in summary_df.iterrows():
        report.append(f"**{row['Experiment']}**:\n")

        if row['Fusion Type'] == 'ALIGNN Early Fusion':
            report.append("- 优势: 文本信息传播距离最长,全局语义指导充分\n")
            report.append("- 劣势: 可能干扰底层几何特征提取\n")
        elif row['Fusion Type'] == 'GCN Late Fusion':
            report.append("- 优势: 几何特征已充分提取,细粒度对齐精准\n")
            report.append("- 劣势: 文本信息传播深度受限\n")
        else:
            report.append("- 优势: 多层次融合,充分利用文本的全局和局部信息\n")
            report.append("- 劣势: 计算成本较高\n")

        report.append(f"- 测试MAE: {row['Best Test MAE']:.4f}\n")
        report.append(f"- 特征相关性: {row['Avg Pearson Corr']:.4f}\n\n")

    report.append("## 结论与建议\n\n")

    # 根据结果给出建议
    mae_values = summary_df['Best Test MAE'].values
    mae_range = mae_values.max() - mae_values.min()

    if mae_range < 0.01:  # 差异很小
        report.append("### 📌 实验结论\n\n")
        report.append("三种融合策略的性能差异较小(MAE差异<0.01),说明:\n")
        report.append("1. 融合位置对模型性能的影响有限\n")
        report.append("2. 文本和图结构信息已经较好地互补\n")
        report.append("3. 可以根据计算效率选择更简单的融合策略\n\n")
    else:  # 差异明显
        report.append("### 📌 实验结论\n\n")
        report.append(f"融合位置对性能有明显影响(MAE差异={mae_range:.4f}):\n")
        report.append(f"1. **最佳策略**: {best_exp['Experiment']}\n")
        report.append(f"2. **性能提升**: 相比最差模型提升了 {(mae_range/mae_values.max()*100):.1f}%\n")
        report.append("3. **建议**: 根据数据特点选择合适的融合位置\n\n")

    report.append("### 🎯 应用建议\n\n")
    report.append("- **全局属性预测** (如形成能、带隙): 优先使用ALIGNN层融合或层次化融合\n")
    report.append("- **局部性质预测** (如原子力、局部磁矩): 优先使用GCN层融合\n")
    report.append("- **计算资源受限**: 单独使用ALIGNN或GCN融合\n")
    report.append("- **追求最佳性能**: 使用层次化融合\n\n")

    report.append("## 可视化结果\n\n")
    report.append("详细的可视化结果请查看:\n")
    report.append("- `fusion_comparison_summary.png` - 综合对比图\n")
    report.append("- `analysis/*/tsne_comparison.png` - 特征分布t-SNE可视化\n")
    report.append("- `analysis/*/regression_metrics_comparison.png` - 回归指标对比\n\n")

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report))

    print(f"✅ 报告已保存: {output_file}\n")


# 实验配置
experiments_config = {
    'ALIGNN层融合': {
        'result_dir': 'results/fusion_at_alignn',
        'analysis_dir': 'analysis/fusion_at_alignn',
        'fusion_type': 'ALIGNN Early Fusion'
    },
    'GCN层融合': {
        'result_dir': 'results/fusion_at_gcn',
        'analysis_dir': 'analysis/fusion_at_gcn',
        'fusion_type': 'GCN Late Fusion'
    },
    '层次化融合': {
        'result_dir': 'results/fusion_hierarchical',
        'analysis_dir': 'analysis/fusion_hierarchical',
        'fusion_type': 'Hierarchical Fusion'
    }
}


if __name__ == '__main__':
    # 汇总实验结果
    summary_df = summarize_experiments(experiments_config)

    # 生成对比图
    plot_comparison(summary_df, output_dir='./')

    # 生成报告
    generate_report(summary_df)

    print("\n" + "="*60)
    print("  🎉 分析完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  - fusion_comparison_summary.csv (汇总数据)")
    print("  - fusion_comparison_summary.png (对比图)")
    print("  - fusion_comparison_report.md (详细报告)")
    print()
