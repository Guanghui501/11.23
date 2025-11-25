#!/usr/bin/env python
"""
快速分析 DynamicFusionModule 训练结果

用法：
    python analyze_fusion_weights.py --output_dir ./output_dynamic_fusion/formation_energy_peratom/
"""

import argparse
import os
import sys

try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ Warning: pandas/matplotlib 未安装，仅显示文本统计")


def analyze_weights(csv_file):
    """分析权重演化"""
    if not HAS_PLOTTING:
        # 简单文本分析
        print("\n" + "="*80)
        print("权重统计（文本模式）")
        print("="*80)

        with open(csv_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print("❌ 权重日志为空")
            return

        # 读取最后一行
        last_line = lines[-1].strip().split(',')
        header = lines[0].strip().split(',')

        print(f"\n总记录数: {len(lines) - 1}")
        print(f"\n最终权重 (Epoch {last_line[0]}):")

        for i, col in enumerate(header):
            if i < len(last_line):
                print(f"  {col}: {last_line[i]}")

        return

    # 使用 pandas 分析
    df = pd.read_csv(csv_file)

    print("\n" + "="*80)
    print("权重统计分析")
    print("="*80)

    print(f"\n总记录数: {len(df)}")
    print(f"训练轮数: {df['epoch'].min()} - {df['epoch'].max()}")

    # 统计信息
    print("\n权重统计:")
    print(df.describe())

    # 最终值
    print(f"\n最终权重 (Epoch {df['epoch'].iloc[-1]}):")
    for col in df.columns:
        if col != 'epoch':
            val = df[col].iloc[-1]
            print(f"  {col}: {val:.6f}")

    # 趋势分析
    print("\n趋势分析:")
    for col in df.columns:
        if col != 'epoch' and 'w_graph' in col:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            change = ((final - initial) / initial) * 100
            print(f"  {col}: {initial:.4f} → {final:.4f} (变化: {change:+.2f}%)")

        if col != 'epoch' and 'w_text' in col:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            change = ((final - initial) / initial) * 100
            print(f"  {col}: {initial:.4f} → {final:.4f} (变化: {change:+.2f}%)")

    # 健康检查
    print("\n健康检查:")
    for col in df.columns:
        if 'eff_ratio' in col:
            final_ratio = df[col].iloc[-1]
            if final_ratio > 10:
                print(f"  ✅ {col}: {final_ratio:.2f}x (图强主导)")
            elif final_ratio > 3:
                print(f"  ✅ {col}: {final_ratio:.2f}x (图占主导)")
            elif final_ratio > 2:
                print(f"  ⚠️ {col}: {final_ratio:.2f}x (图略占优)")
            else:
                print(f"  ❌ {col}: {final_ratio:.2f}x (警告：文本权重过高！)")

    return df


def plot_weights(df, output_dir):
    """绘制权重演化图"""
    if not HAS_PLOTTING:
        return

    # 找到所有层
    layers = set()
    for col in df.columns:
        if 'layer_' in col:
            layer_name = col.split('_')[0] + '_' + col.split('_')[1]
            layers.add(layer_name)

    for layer in sorted(layers):
        plt.figure(figsize=(12, 5))

        # 子图1: w_graph 和 w_text
        plt.subplot(1, 2, 1)
        w_g_col = f'{layer}_w_graph'
        w_t_col = f'{layer}_w_text'

        if w_g_col in df.columns and w_t_col in df.columns:
            plt.plot(df['epoch'], df[w_g_col], label='w_graph', linewidth=2, marker='o', markersize=4)
            plt.plot(df['epoch'], df[w_t_col], label='w_text', linewidth=2, marker='s', markersize=4)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.title(f'{layer}: Router Weights (Softmax)', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

        # 子图2: 有效比例
        plt.subplot(1, 2, 2)
        ratio_col = f'{layer}_eff_ratio'

        if ratio_col in df.columns:
            plt.plot(df['epoch'], df[ratio_col], linewidth=2, color='purple', marker='D', markersize=4)
            plt.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='Healthy (3x)')
            plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Warning (2x)')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Graph/Text Ratio', fontsize=12)
            plt.title(f'{layer}: Effective Weight Ratio', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{layer}_weights.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析 DynamicFusionModule 权重')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='训练输出目录（包含 fusion_weights.csv）')
    args = parser.parse_args()

    # 查找权重文件
    csv_file = os.path.join(args.output_dir, 'fusion_weights.csv')

    if not os.path.exists(csv_file):
        print(f"\n❌ 错误: 找不到权重文件: {csv_file}")
        print(f"\n可能的原因:")
        print(f"  1. 训练轮数 < 5（监控每5轮记录一次）")
        print(f"  2. --use_middle_fusion 未启用")
        print(f"  3. 输出目录不正确")
        sys.exit(1)

    print(f"\n读取权重文件: {csv_file}")

    # 分析
    df = analyze_weights(csv_file)

    # 绘图
    if HAS_PLOTTING and df is not None:
        print(f"\n生成可视化图表...")
        plot_weights(df, args.output_dir)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
