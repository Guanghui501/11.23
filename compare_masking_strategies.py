#!/usr/bin/env python
"""
对比不同遮挡策略的结果

用法:
    python compare_masking_strategies.py --input_dir ./masking_evaluation
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(input_dir):
    """加载所有策略的结果"""
    strategies = ['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords']
    results = {}

    for strategy in strategies:
        json_file = os.path.join(input_dir, strategy, f'text_masking_results_{strategy}.json')

        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[strategy] = json.load(f)
                print(f"✓ 加载 {strategy} 的结果")
        else:
            print(f"⚠ 警告: 找不到 {strategy} 的结果文件: {json_file}")

    return results


def plot_comparison(results, output_dir):
    """绘制策略对比图"""
    if not results:
        print("错误: 没有找到任何结果数据")
        return

    # 设置风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 策略显示名称映射
    strategy_names = {
        'random_token': 'Random Token',
        'random_word': 'Random Word',
        'random_chunk': 'Random Chunk',
        'sentence': 'Sentence',
        'keep_keywords': 'Keep Keywords'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. MAE对比
    ax1 = axes[0, 0]
    for strategy, data in results.items():
        ax1.plot(data['masking_ratios'], data['mae'], 'o-',
                linewidth=2, markersize=6, label=strategy_names.get(strategy, strategy))
    ax1.set_xlabel('Text Masking Ratio', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('MAE Comparison Across Masking Strategies', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. RMSE对比
    ax2 = axes[0, 1]
    for strategy, data in results.items():
        ax2.plot(data['masking_ratios'], data['rmse'], 's-',
                linewidth=2, markersize=6, label=strategy_names.get(strategy, strategy))
    ax2.set_xlabel('Text Masking Ratio', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE Comparison Across Masking Strategies', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. R²对比
    ax3 = axes[1, 0]
    for strategy, data in results.items():
        ax3.plot(data['masking_ratios'], data['r2'], '^-',
                linewidth=2, markersize=6, label=strategy_names.get(strategy, strategy))
    ax3.set_xlabel('Text Masking Ratio', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('R² Score Comparison Across Masking Strategies', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. 相对性能下降（以0%遮挡为基准）
    ax4 = axes[1, 1]
    for strategy, data in results.items():
        baseline_mae = data['mae'][0]  # 0% masking
        relative_increase = [(mae - baseline_mae) / baseline_mae * 100 for mae in data['mae']]
        ax4.plot(data['masking_ratios'], relative_increase, 'o-',
                linewidth=2, markersize=6, label=strategy_names.get(strategy, strategy))
    ax4.set_xlabel('Text Masking Ratio', fontsize=12)
    ax4.set_ylabel('Relative MAE Increase (%)', fontsize=12)
    ax4.set_title('Relative Performance Degradation', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存图片
    plot_file = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {plot_file}")
    plt.close()


def generate_comparison_report(results, output_dir):
    """生成对比报告"""
    report_file = os.path.join(output_dir, 'strategy_comparison_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("文本遮挡策略对比报告\n")
        f.write("="*100 + "\n\n")

        # 1. 基线性能（0% masking）
        f.write("1. 基线性能（0% 遮挡）\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Strategy':<20} {'MAE':<15} {'RMSE':<15} {'R²':<15}\n")
        f.write("-"*100 + "\n")
        for strategy, data in results.items():
            f.write(f"{strategy:<20} {data['mae'][0]:<15.4f} {data['rmse'][0]:<15.4f} {data['r2'][0]:<15.4f}\n")
        f.write("\n\n")

        # 2. 完全遮挡性能（100% masking）
        f.write("2. 完全遮挡性能（100% 遮挡）\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Strategy':<20} {'MAE':<15} {'RMSE':<15} {'R²':<15}\n")
        f.write("-"*100 + "\n")
        for strategy, data in results.items():
            f.write(f"{strategy:<20} {data['mae'][-1]:<15.4f} {data['rmse'][-1]:<15.4f} {data['r2'][-1]:<15.4f}\n")
        f.write("\n\n")

        # 3. 性能下降分析
        f.write("3. 性能下降分析（从0%到100%遮挡）\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Strategy':<20} {'MAE Increase':<20} {'RMSE Increase':<20} {'R² Decrease':<20}\n")
        f.write("-"*100 + "\n")
        for strategy, data in results.items():
            mae_increase = ((data['mae'][-1] - data['mae'][0]) / data['mae'][0]) * 100
            rmse_increase = ((data['rmse'][-1] - data['rmse'][0]) / data['rmse'][0]) * 100
            r2_decrease = ((data['r2'][0] - data['r2'][-1]) / abs(data['r2'][0])) * 100

            f.write(f"{strategy:<20} {mae_increase:<20.2f}% {rmse_increase:<20.2f}% {r2_decrease:<20.2f}%\n")
        f.write("\n\n")

        # 4. 鲁棒性排名（基于MAE增长率，越小越好）
        f.write("4. 鲁棒性排名（基于MAE增长率，越小越鲁棒）\n")
        f.write("-"*100 + "\n")

        robustness_scores = {}
        for strategy, data in results.items():
            mae_increase = ((data['mae'][-1] - data['mae'][0]) / data['mae'][0]) * 100
            robustness_scores[strategy] = mae_increase

        # 按MAE增长率排序
        sorted_strategies = sorted(robustness_scores.items(), key=lambda x: x[1])

        f.write(f"{'Rank':<10} {'Strategy':<20} {'MAE Increase':<20} {'评价':<20}\n")
        f.write("-"*100 + "\n")

        for rank, (strategy, score) in enumerate(sorted_strategies, 1):
            if score < 50:
                evaluation = "优秀 ⭐⭐⭐"
            elif score < 100:
                evaluation = "良好 ⭐⭐"
            elif score < 200:
                evaluation = "一般 ⭐"
            else:
                evaluation = "较弱"

            f.write(f"{rank:<10} {strategy:<20} {score:<20.2f}% {evaluation:<20}\n")

        f.write("\n\n")

        # 5. 关键发现
        f.write("5. 关键发现\n")
        f.write("-"*100 + "\n")

        most_robust = sorted_strategies[0][0]
        least_robust = sorted_strategies[-1][0]

        f.write(f"• 最鲁棒的策略: {most_robust} (MAE增长率: {sorted_strategies[0][1]:.2f}%)\n")
        f.write(f"• 最不鲁棒的策略: {least_robust} (MAE增长率: {sorted_strategies[-1][1]:.2f}%)\n")
        f.write(f"• 鲁棒性差异: {sorted_strategies[-1][1] - sorted_strategies[0][1]:.2f}%\n\n")

        # 分析不同策略的特点
        f.write("策略特点分析:\n\n")

        if 'random_token' in robustness_scores:
            f.write(f"  - random_token: 随机遮挡单个token，测试词汇级鲁棒性\n")
            f.write(f"    MAE增长率: {robustness_scores['random_token']:.2f}%\n\n")

        if 'random_word' in robustness_scores:
            f.write(f"  - random_word: 随机遮挡完整单词，测试单词级鲁棒性\n")
            f.write(f"    MAE增长率: {robustness_scores['random_word']:.2f}%\n\n")

        if 'random_chunk' in robustness_scores:
            f.write(f"  - random_chunk: 遮挡连续文本块，测试局部信息缺失的影响\n")
            f.write(f"    MAE增长率: {robustness_scores['random_chunk']:.2f}%\n\n")

        if 'sentence' in robustness_scores:
            f.write(f"  - sentence: 遮挡完整句子，测试句子级信息缺失的影响\n")
            f.write(f"    MAE增长率: {robustness_scores['sentence']:.2f}%\n\n")

        if 'keep_keywords' in robustness_scores:
            f.write(f"  - keep_keywords: 只保留关键词，测试最小信息下的预测能力\n")
            f.write(f"    MAE增长率: {robustness_scores['keep_keywords']:.2f}%\n\n")

        f.write("\n")
        f.write("="*100 + "\n")

    print(f"对比报告已保存到: {report_file}")


def create_summary_table(results, output_dir):
    """创建汇总表格（CSV格式）"""
    import csv

    csv_file = os.path.join(output_dir, 'strategy_comparison_summary.csv')

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        header = ['Masking Ratio']
        for strategy in results.keys():
            header.extend([f'{strategy}_MAE', f'{strategy}_RMSE', f'{strategy}_R2'])
        writer.writerow(header)

        # 写入数据
        # 获取第一个策略的遮挡率（假设所有策略使用相同的遮挡率）
        first_strategy = list(results.keys())[0]
        masking_ratios = results[first_strategy]['masking_ratios']

        for i, ratio in enumerate(masking_ratios):
            row = [f"{ratio:.1%}"]
            for strategy in results.keys():
                row.append(f"{results[strategy]['mae'][i]:.4f}")
                row.append(f"{results[strategy]['rmse'][i]:.4f}")
                row.append(f"{results[strategy]['r2'][i]:.4f}")
            writer.writerow(row)

    print(f"汇总表格已保存到: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='对比不同遮挡策略的结果')
    parser.add_argument('--input_dir', type=str, default='./masking_evaluation',
                       help='包含所有策略结果的目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认与input_dir相同）')

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir

    print("="*80)
    print("文本遮挡策略对比分析")
    print("="*80)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {output_dir}")
    print("="*80 + "\n")

    # 加载结果
    print("加载结果...")
    results = load_results(args.input_dir)

    if not results:
        print("\n❌ 错误: 没有找到任何结果文件")
        print(f"请确保 {args.input_dir} 目录下包含各策略的结果")
        print("\n目录结构应该如下:")
        print("  masking_evaluation/")
        print("    random_token/")
        print("      text_masking_results_random_token.json")
        print("    random_word/")
        print("      text_masking_results_random_word.json")
        print("    ...")
        return

    print(f"\n找到 {len(results)} 个策略的结果\n")

    # 生成对比图
    print("生成对比图...")
    plot_comparison(results, output_dir)

    # 生成对比报告
    print("\n生成对比报告...")
    generate_comparison_report(results, output_dir)

    # 生成汇总表格
    print("\n生成汇总表格...")
    create_summary_table(results, output_dir)

    print("\n" + "="*80)
    print("对比分析完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - {os.path.join(output_dir, 'strategy_comparison.png')}")
    print(f"  - {os.path.join(output_dir, 'strategy_comparison_report.txt')}")
    print(f"  - {os.path.join(output_dir, 'strategy_comparison_summary.csv')}")
    print("\n")


if __name__ == "__main__":
    main()
