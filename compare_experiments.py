#!/usr/bin/env python
"""
实验结果对比脚本
快速对比多个调优实验的性能指标
"""

import os
import argparse
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def load_experiment_results(exp_dir):
    """加载单个实验的结果"""
    results = {
        'name': os.path.basename(exp_dir),
        'path': exp_dir
    }

    # 尝试加载config
    config_file = os.path.join(exp_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            results['config'] = config

            # 提取关键参数
            if 'model' in config:
                model_config = config['model']
                results['middle_fusion_layers'] = model_config.get('middle_fusion_layers', 'N/A')
                results['middle_fusion_hidden_dim'] = model_config.get('middle_fusion_hidden_dim', 'N/A')
                results['middle_fusion_num_heads'] = model_config.get('middle_fusion_num_heads', 'N/A')
                results['graph_dropout'] = model_config.get('graph_dropout', 'N/A')
                results['cross_modal_num_heads'] = model_config.get('cross_modal_num_heads', 'N/A')
                results['use_cross_modal_attention'] = model_config.get('use_cross_modal_attention', 'N/A')

            results['learning_rate'] = config.get('learning_rate', 'N/A')
            results['weight_decay'] = config.get('weight_decay', 'N/A')
            results['batch_size'] = config.get('batch_size', 'N/A')

    # 尝试加载训练历史
    history_file = os.path.join(exp_dir, 'history_val.csv')
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)

        # 提取最佳验证指标
        if 'mae' in history.columns:
            best_val_mae = history['mae'].min()
            results['best_val_mae'] = best_val_mae
            results['best_epoch'] = history['mae'].idxmin()

        if 'loss' in history.columns:
            results['best_val_loss'] = history['loss'].min()

        # 提取最终指标
        if len(history) > 0:
            results['final_val_mae'] = history['mae'].iloc[-1] if 'mae' in history.columns else None
            results['final_epoch'] = len(history)

    # 尝试加载训练历史（train）
    train_history_file = os.path.join(exp_dir, 'history_train.csv')
    if os.path.exists(train_history_file):
        train_history = pd.read_csv(train_history_file)

        # 计算train-val gap
        if 'mae' in train_history.columns and 'best_val_mae' in results:
            # 取验证MAE最佳时的训练MAE
            best_epoch = results.get('best_epoch', len(train_history) - 1)
            if best_epoch < len(train_history):
                train_mae_at_best = train_history['mae'].iloc[best_epoch]
                results['train_mae_at_best'] = train_mae_at_best
                results['train_val_gap'] = results['best_val_mae'] / train_mae_at_best

    return results


def compare_experiments(experiment_dirs, save_dir):
    """对比多个实验"""
    print("🔍 加载实验结果...")

    all_results = []
    for exp_dir in experiment_dirs:
        if os.path.isdir(exp_dir):
            print(f"   加载: {os.path.basename(exp_dir)}")
            results = load_experiment_results(exp_dir)
            all_results.append(results)

    if len(all_results) == 0:
        print("❌ 没有找到有效的实验结果")
        return

    print(f"✅ 成功加载 {len(all_results)} 个实验\n")

    # 创建对比表格
    print("📊 生成对比表格...")

    df_data = []
    for r in all_results:
        row = {
            'Experiment': r['name'],
            'Best Val MAE': r.get('best_val_mae', 'N/A'),
            'Best Epoch': r.get('best_epoch', 'N/A'),
            'Train MAE': r.get('train_mae_at_best', 'N/A'),
            'Train-Val Gap': r.get('train_val_gap', 'N/A'),
            'Fusion Layers': r.get('middle_fusion_layers', 'N/A'),
            'Fusion Hidden': r.get('middle_fusion_hidden_dim', 'N/A'),
            'Fusion Heads': r.get('middle_fusion_num_heads', 'N/A'),
            'Graph Dropout': r.get('graph_dropout', 'N/A'),
            'Cross Heads': r.get('cross_modal_num_heads', 'N/A'),
            'LR': r.get('learning_rate', 'N/A'),
            'Weight Decay': r.get('weight_decay', 'N/A'),
            'Batch Size': r.get('batch_size', 'N/A')
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # 保存为CSV
    csv_path = os.path.join(save_dir, 'experiments_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 对比表格已保存: {csv_path}\n")

    # 打印表格
    print("=" * 120)
    print("实验结果对比")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    print()

    # 找出最佳实验
    valid_results = [r for r in all_results if 'best_val_mae' in r]
    if valid_results:
        best_exp = min(valid_results, key=lambda x: x['best_val_mae'])
        print(f"🏆 最佳实验: {best_exp['name']}")
        print(f"   Best Val MAE: {best_exp['best_val_mae']:.4f}")
        print(f"   Best Epoch: {best_exp.get('best_epoch', 'N/A')}")
        if 'train_val_gap' in best_exp:
            print(f"   Train-Val Gap: {best_exp['train_val_gap']:.2f}x")
        print()

    # 可视化对比
    print("📈 生成对比可视化...")
    visualize_comparison(all_results, save_dir)

    # 生成详细报告
    print("📝 生成详细报告...")
    generate_report(all_results, save_dir)


def visualize_comparison(results, save_dir):
    """可视化实验对比"""

    # 准备数据
    valid_results = [r for r in results if 'best_val_mae' in r]
    if len(valid_results) == 0:
        print("⚠️  没有足够的数据进行可视化")
        return

    names = [r['name'] for r in valid_results]
    val_maes = [r['best_val_mae'] for r in valid_results]
    train_maes = [r.get('train_mae_at_best', np.nan) for r in valid_results]
    train_val_gaps = [r.get('train_val_gap', np.nan) for r in valid_results]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 验证MAE对比
    ax = axes[0, 0]
    x = range(len(names))
    bars = ax.bar(x, val_maes, alpha=0.7, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Best Validation MAE', fontweight='bold')
    ax.set_title('Validation MAE Comparison', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, val_maes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 标注最佳
    best_idx = np.argmin(val_maes)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)

    # 2. 训练-验证MAE对比
    ax = axes[0, 1]
    x = range(len(names))
    width = 0.35

    valid_train = [m for m in train_maes if not np.isnan(m)]
    if len(valid_train) > 0:
        bars1 = ax.bar([i - width/2 for i in x], train_maes, width,
                      label='Train MAE', alpha=0.7, color='orange')
        bars2 = ax.bar([i + width/2 for i in x], val_maes, width,
                      label='Val MAE', alpha=0.7, color='steelblue')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('MAE', fontweight='bold')
        ax.set_title('Train vs Validation MAE', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # 3. Train-Val Gap对比
    ax = axes[1, 0]
    valid_gaps = [g for g in train_val_gaps if not np.isnan(g)]
    if len(valid_gaps) > 0:
        x_valid = [i for i, g in enumerate(train_val_gaps) if not np.isnan(g)]
        names_valid = [names[i] for i in x_valid]

        bars = ax.bar(range(len(valid_gaps)), valid_gaps, alpha=0.7, color='coral')
        ax.set_xticks(range(len(valid_gaps)))
        ax.set_xticklabels(names_valid, rotation=45, ha='right')
        ax.set_ylabel('Train-Val Gap (Ratio)', fontweight='bold')
        ax.set_title('Train-Val Gap Comparison (Lower is Better)', fontweight='bold', fontsize=14)
        ax.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Gap=2.0')
        ax.axhline(y=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Gap=1.5')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, valid_gaps)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

    # 4. 参数对比（融合层配置）
    ax = axes[1, 1]
    fusion_configs = []
    fusion_labels = []
    fusion_maes = []

    for r in valid_results:
        layers = str(r.get('middle_fusion_layers', 'N/A'))
        hidden = str(r.get('middle_fusion_hidden_dim', 'N/A'))
        heads = str(r.get('middle_fusion_num_heads', 'N/A'))
        config_str = f"L:{layers}\nH:{hidden}\nHd:{heads}"

        fusion_configs.append(config_str)
        fusion_labels.append(r['name'][:20])
        fusion_maes.append(r['best_val_mae'])

    x = range(len(fusion_configs))
    bars = ax.bar(x, fusion_maes, alpha=0.7, color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels(fusion_configs, rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Best Validation MAE', fontweight='bold')
    ax.set_title('MAE vs Fusion Configuration', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 标注实验名
    for i, (bar, label) in enumerate(zip(bars, fusion_labels)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               label, ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'experiments_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}")
    plt.close()


def generate_report(results, save_dir):
    """生成详细文本报告"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("参数调优实验对比报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 1. 整体统计
    valid_results = [r for r in results if 'best_val_mae' in r]
    if len(valid_results) > 0:
        val_maes = [r['best_val_mae'] for r in valid_results]

        report_lines.append("📊 整体统计:")
        report_lines.append(f"  • 实验总数: {len(results)}")
        report_lines.append(f"  • 有效实验: {len(valid_results)}")
        report_lines.append(f"  • 最佳 Val MAE: {min(val_maes):.4f}")
        report_lines.append(f"  • 最差 Val MAE: {max(val_maes):.4f}")
        report_lines.append(f"  • 平均 Val MAE: {np.mean(val_maes):.4f}")
        report_lines.append(f"  • 标准差: {np.std(val_maes):.4f}")
        report_lines.append("")

    # 2. 各实验详情
    report_lines.append("📋 实验详情:")
    report_lines.append("")

    # 按MAE排序
    sorted_results = sorted(valid_results, key=lambda x: x['best_val_mae'])

    for rank, r in enumerate(sorted_results, 1):
        report_lines.append(f"  {rank}. {r['name']}")
        report_lines.append(f"     Best Val MAE: {r['best_val_mae']:.4f} (Epoch {r.get('best_epoch', 'N/A')})")

        if 'train_mae_at_best' in r:
            report_lines.append(f"     Train MAE: {r['train_mae_at_best']:.4f}")

        if 'train_val_gap' in r:
            gap = r['train_val_gap']
            if gap < 1.5:
                gap_status = "✓ 优秀"
            elif gap < 2.0:
                gap_status = "○ 良好"
            else:
                gap_status = "✗ 过拟合"
            report_lines.append(f"     Train-Val Gap: {gap:.2f}x {gap_status}")

        report_lines.append(f"     配置:")
        report_lines.append(f"       - Fusion Layers: {r.get('middle_fusion_layers', 'N/A')}")
        report_lines.append(f"       - Fusion Hidden: {r.get('middle_fusion_hidden_dim', 'N/A')}")
        report_lines.append(f"       - Fusion Heads: {r.get('middle_fusion_num_heads', 'N/A')}")
        report_lines.append(f"       - Graph Dropout: {r.get('graph_dropout', 'N/A')}")
        report_lines.append(f"       - Learning Rate: {r.get('learning_rate', 'N/A')}")
        report_lines.append(f"       - Weight Decay: {r.get('weight_decay', 'N/A')}")
        report_lines.append("")

    # 3. 关键发现
    report_lines.append("🔍 关键发现:")
    report_lines.append("")

    # 最佳实验
    best_exp = sorted_results[0]
    report_lines.append(f"  • 最佳实验: {best_exp['name']}")
    report_lines.append(f"    Val MAE: {best_exp['best_val_mae']:.4f}")

    if len(sorted_results) > 1:
        baseline = sorted_results[-1]
        improvement = ((baseline['best_val_mae'] - best_exp['best_val_mae']) /
                      baseline['best_val_mae'] * 100)
        report_lines.append(f"    相比最差提升: {improvement:.2f}%")

    report_lines.append("")

    # 4. 建议
    report_lines.append("💡 建议:")
    report_lines.append("")

    best_gap = best_exp.get('train_val_gap', float('inf'))
    if best_gap < 1.5:
        report_lines.append("  ✅ 最佳模型泛化性能优秀，可以用于论文")
    elif best_gap < 2.0:
        report_lines.append("  ○ 最佳模型泛化性能良好，可以考虑微调")
    else:
        report_lines.append("  ⚠️  最佳模型仍有过拟合，建议:")
        report_lines.append("     - 进一步增加正则化")
        report_lines.append("     - 减小模型容量")
        report_lines.append("     - 尝试数据增强")

    report_lines.append("")

    # 融合层配置分析
    fusion_analysis = {}
    for r in valid_results:
        layers = str(r.get('middle_fusion_layers', 'N/A'))
        if layers not in fusion_analysis:
            fusion_analysis[layers] = []
        fusion_analysis[layers].append(r['best_val_mae'])

    if len(fusion_analysis) > 1:
        report_lines.append("  融合层配置分析:")
        for layers, maes in sorted(fusion_analysis.items(), key=lambda x: np.mean(x[1])):
            avg_mae = np.mean(maes)
            report_lines.append(f"    Layers {layers}: 平均 MAE = {avg_mae:.4f} (n={len(maes)})")

        best_fusion_config = min(fusion_analysis.items(), key=lambda x: np.mean(x[1]))
        report_lines.append(f"    → 推荐融合层配置: {best_fusion_config[0]}")
        report_lines.append("")

    report_lines.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report_lines)
    save_path = os.path.join(save_dir, 'experiments_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ 详细报告已保存: {save_path}\n")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description='对比多个调优实验的结果')
    parser.add_argument('--experiment_dirs', type=str, nargs='+', required=True,
                       help='实验目录列表（支持通配符）')
    parser.add_argument('--save_dir', type=str, default='./experiment_comparison',
                       help='结果保存目录')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 展开通配符
    all_dirs = []
    for pattern in args.experiment_dirs:
        matched = glob.glob(pattern)
        all_dirs.extend(matched)

    # 去重并排序
    all_dirs = sorted(set(all_dirs))

    print(f"🔍 找到 {len(all_dirs)} 个实验目录\n")

    if len(all_dirs) == 0:
        print("❌ 没有找到实验目录")
        return

    # 对比实验
    compare_experiments(all_dirs, args.save_dir)

    print(f"\n🎉 对比完成! 结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
