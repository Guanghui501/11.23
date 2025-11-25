#!/usr/bin/env python
"""
对比融合层搜索结果

用法:
    python compare_search_results.py --search_dir ./fusion_layer_search/
"""

import argparse
import os
import pandas as pd
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='对比融合层搜索结果')
    parser.add_argument('--search_dir', type=str, default='./fusion_layer_search/',
                        help='搜索结果目录')
    return parser.parse_args()


def load_results(search_dir):
    """加载搜索结果"""
    results_file = os.path.join(search_dir, 'results_summary.csv')

    if not os.path.exists(results_file):
        print(f"❌ 找不到结果文件: {results_file}")
        return None

    df = pd.read_csv(results_file)
    return df


def print_summary(df):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("融合层位置搜索 - 结果汇总")
    print("="*80 + "\n")

    # 按验证 MAE 排序
    df_sorted = df.sort_values('best_val_mae')

    print("🏆 按验证集 MAE 排序（越小越好）:\n")
    print(df_sorted.to_string(index=False))
    print("\n")

    # 找到最佳配置
    best_config = df_sorted.iloc[0]

    print("="*80)
    print("✅ 最佳配置")
    print("="*80)
    print(f"Fusion Layers:    {best_config['fusion_layers']}")
    print(f"验证集 MAE:       {best_config['best_val_mae']:.4f}")
    print(f"测试集 MAE:       {best_config['best_test_mae']:.4f}")

    if best_config['final_w_graph'] != 'N/A':
        print(f"最终 w_graph:     {best_config['final_w_graph']:.4f}")
        print(f"最终 w_text:      {best_config['final_w_text']:.4f}")
        print(f"图/文本比例:      {best_config['ratio']:.2f}x")

    print("\n")

    # 性能对比
    print("="*80)
    print("📊 性能对比")
    print("="*80 + "\n")

    baseline = df[df['fusion_layers'] == '2'].iloc[0] if '2' in df['fusion_layers'].values else df_sorted.iloc[-1]

    print(f"基线配置 (layers=2): MAE = {baseline['best_val_mae']:.4f}")
    print(f"最佳配置 (layers={best_config['fusion_layers']}): MAE = {best_config['best_val_mae']:.4f}")

    improvement = ((baseline['best_val_mae'] - best_config['best_val_mae']) / baseline['best_val_mae']) * 100
    print(f"相对提升: {improvement:+.2f}%")

    print("\n")

    # 权重分析
    if 'ratio' in df.columns and df['ratio'].dtype != object:
        print("="*80)
        print("🔍 权重比例分析")
        print("="*80 + "\n")

        df_valid = df[df['ratio'] != 'N/A'].copy()
        if not df_valid.empty:
            df_valid['ratio'] = pd.to_numeric(df_valid['ratio'], errors='coerce')

            print("各配置的图/文本权重比例:\n")
            for _, row in df_valid.iterrows():
                ratio = row['ratio']
                layers = row['fusion_layers']

                if ratio > 10:
                    status = "✅ 图强主导"
                elif ratio > 5:
                    status = "✅ 图占主导"
                elif ratio > 3:
                    status = "✓ 图偏优"
                elif ratio > 2:
                    status = "⚠️ 图略优"
                else:
                    status = "❌ 警告：文本过高"

                print(f"  Layers {layers:8s}: {ratio:5.2f}x  {status}")

            print("\n")


def print_recommendations(df):
    """打印建议"""
    print("="*80)
    print("💡 下一步建议")
    print("="*80 + "\n")

    df_sorted = df.sort_values('best_val_mae')
    top3 = df_sorted.head(3)

    print("📌 Top 3 配置推荐用于阶段2（中等数据精细调整）:\n")

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"{i}. Fusion Layers = {row['fusion_layers']}")
        print(f"   验证 MAE: {row['best_val_mae']:.4f}")

        if row['ratio'] != 'N/A':
            print(f"   权重比例: {row['ratio']:.2f}x")
        print()

    best_layers = df_sorted.iloc[0]['fusion_layers']

    print("🚀 推荐命令（使用最佳配置）:\n")
    print(f"# 阶段2: 中等数据精细调整")
    print(f"./fine_tune_search.sh --fusion_layers \"{best_layers}\"\n")
    print(f"# 或直接进行完整训练")
    print(f"python train_with_cross_modal_attention.py \\")
    print(f"    --use_middle_fusion True \\")
    print(f"    --middle_fusion_layers \"{best_layers}\" \\")
    print(f"    --epochs 100 \\")
    print(f"    --output_dir ./output_best_config/\n")


def plot_comparison(df, search_dir):
    """绘制对比图"""
    try:
        import matplotlib.pyplot as plt

        df_sorted = df.sort_values('best_val_mae')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1: MAE 对比
        ax1 = axes[0]
        x = range(len(df_sorted))
        ax1.bar(x, df_sorted['best_val_mae'], alpha=0.7, label='Validation MAE')
        ax1.bar(x, df_sorted['best_test_mae'], alpha=0.5, label='Test MAE')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_sorted['fusion_layers'], rotation=45)
        ax1.set_xlabel('Fusion Layers')
        ax1.set_ylabel('MAE')
        ax1.set_title('Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 权重比例
        if 'ratio' in df.columns:
            ax2 = axes[1]
            df_valid = df_sorted[df_sorted['ratio'] != 'N/A'].copy()

            if not df_valid.empty:
                df_valid['ratio'] = pd.to_numeric(df_valid['ratio'], errors='coerce')

                ax2.bar(range(len(df_valid)), df_valid['ratio'], alpha=0.7, color='purple')
                ax2.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='Healthy (>3x)')
                ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Warning (<2x)')
                ax2.set_xticks(range(len(df_valid)))
                ax2.set_xticklabels(df_valid['fusion_layers'], rotation=45)
                ax2.set_xlabel('Fusion Layers')
                ax2.set_ylabel('Graph/Text Ratio')
                ax2.set_title('Weight Ratio (Graph Dominance)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = os.path.join(search_dir, 'comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 对比图已保存: {output_file}\n")

    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图\n")


def main():
    args = parse_args()

    print(f"\n读取结果: {args.search_dir}")

    df = load_results(args.search_dir)

    if df is None or df.empty:
        print("❌ 没有找到有效结果")
        return

    print(f"找到 {len(df)} 个配置的结果\n")

    # 打印摘要
    print_summary(df)

    # 绘图
    plot_comparison(df, args.search_dir)

    # 打印建议
    print_recommendations(df)

    print("="*80)
    print("✅ 分析完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
