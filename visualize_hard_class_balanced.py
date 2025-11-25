#!/usr/bin/env python
"""
难分样本可视化 (Hard Class Subset) - 平衡版本
在原版基础上增加样本平衡功能
"""

import os
import sys
import argparse
import numpy as np
import random

# 导入原始脚本的所有函数
sys.path.insert(0, os.path.dirname(__file__))
from visualize_hard_class_subset import (
    load_model, extract_crystal_systems_from_dataset,
    filter_by_crystal_systems, extract_features,
    compute_clustering_metrics, compute_class_separation,
    apply_reduction, plot_hard_class_comparison,
    plot_separation_metrics, CRYSTAL_SYSTEMS
)

from pathlib import Path
from tqdm import tqdm
import torch
from jarvis.core.atoms import Atoms
from data import get_torch_dataset
from torch.utils.data import DataLoader


def balance_samples_by_class(filtered_dataset, filtered_systems, target_systems, balance_method='downsample'):
    """
    平衡两个类别的样本数量

    Args:
        filtered_dataset: 过滤后的数据集
        filtered_systems: 过滤后的晶系列表
        target_systems: 目标晶系列表 [class1, class2]
        balance_method: 'downsample' (下采样多数类) 或 'upsample' (上采样少数类)

    Returns:
        balanced_dataset: 平衡后的数据集
        balanced_systems: 平衡后的晶系列表
    """
    class1, class2 = target_systems

    # 分离两个类别
    class1_indices = [i for i, cs in enumerate(filtered_systems) if cs == class1]
    class2_indices = [i for i, cs in enumerate(filtered_systems) if cs == class2]

    n1 = len(class1_indices)
    n2 = len(class2_indices)

    print(f"\n🔄 样本平衡 (方法: {balance_method})")
    print(f"   原始样本数:")
    print(f"     {CRYSTAL_SYSTEMS[class1]}: {n1}")
    print(f"     {CRYSTAL_SYSTEMS[class2]}: {n2}")

    if balance_method == 'downsample':
        # 下采样到较小的类别数量
        target_size = min(n1, n2)

        if n1 > target_size:
            random.shuffle(class1_indices)
            class1_indices = class1_indices[:target_size]

        if n2 > target_size:
            random.shuffle(class2_indices)
            class2_indices = class2_indices[:target_size]

    elif balance_method == 'upsample':
        # 上采样到较大的类别数量
        target_size = max(n1, n2)

        if n1 < target_size:
            # 重复采样
            class1_indices = class1_indices + random.choices(class1_indices, k=target_size - n1)

        if n2 < target_size:
            class2_indices = class2_indices + random.choices(class2_indices, k=target_size - n2)

    else:
        raise ValueError(f"Unknown balance method: {balance_method}")

    # 合并并重新构建数据集
    balanced_indices = class1_indices + class2_indices
    random.shuffle(balanced_indices)

    balanced_dataset = [filtered_dataset[i] for i in balanced_indices]
    balanced_systems = [filtered_systems[i] for i in balanced_indices]

    # 统计平衡后的数量
    n1_balanced = balanced_systems.count(class1)
    n2_balanced = balanced_systems.count(class2)

    print(f"   平衡后样本数:")
    print(f"     {CRYSTAL_SYSTEMS[class1]}: {n1_balanced}")
    print(f"     {CRYSTAL_SYSTEMS[class2]}: {n2_balanced}")
    print(f"   总样本数: {len(balanced_dataset)}")

    return balanced_dataset, balanced_systems


def main():
    parser = argparse.ArgumentParser(description='难分样本可视化 (平衡版本)')
    parser.add_argument('--checkpoint_without_fusion', type=str, required=True)
    parser.add_argument('--checkpoint_with_fusion', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='jarvis')
    parser.add_argument('--property', type=str, default='mbj_bandgap')
    parser.add_argument('--class_pair', type=str, default='cubic,tetragonal')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--balance_method', type=str, default='downsample',
                       choices=['downsample', 'upsample'],
                       help='平衡方法: downsample(下采样) 或 upsample(上采样)')
    parser.add_argument('--reduction_method', type=str, default='tsne',
                       choices=['tsne', 'umap'])
    parser.add_argument('--output_dir', type=str, default='hard_class_balanced_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 解析晶系对
    class_pair = [cs.strip().lower() for cs in args.class_pair.split(',')]
    if len(class_pair) != 2:
        raise ValueError("--class_pair 必须指定两个晶系")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("难分样本可视化 (平衡版本)")
    print("=" * 80)
    print(f"目标晶系对: {CRYSTAL_SYSTEMS[class_pair[0]]} vs {CRYSTAL_SYSTEMS[class_pair[1]]}")
    print(f"平衡方法: {args.balance_method}")
    print("=" * 80)

    # 构建路径
    cif_dir = os.path.join(args.data_dir, f'{args.dataset}/{args.property}/cif/')
    id_prop_file = os.path.join(args.data_dir, f'{args.dataset}/{args.property}/description.csv')

    # 加载模型
    print("\n" + "=" * 80)
    print("1️⃣ 加载模型")
    print("=" * 80)
    model_without, _ = load_model(args.checkpoint_without_fusion, args.device)
    model_with, _ = load_model(args.checkpoint_with_fusion, args.device)

    # 加载数据
    print("\n" + "=" * 80)
    print("2️⃣ 加载数据集")
    print("=" * 80)

    import csv
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"总样本数: {len(data)}")

    # 采样
    if len(data) > args.n_samples:
        random.shuffle(data)
        data = data[:args.n_samples]
        print(f"随机采样 {args.n_samples} 个样本")

    # 构建dataset_array
    dataset_array = []
    skipped = 0

    for j in tqdm(range(len(data)), desc="加载样本"):
        try:
            sample_id = data[j][0]
            target_val = float(data[j][2])
            description = data[j][3] if len(data[j]) > 3 else ""

            cif_file = os.path.join(cif_dir, f'{sample_id}.cif')
            if not os.path.exists(cif_file):
                skipped += 1
                continue

            atoms = Atoms.from_cif(cif_file)

            info = {
                "atoms": atoms.to_dict(),
                "jid": sample_id,
                "text": description if description else atoms.composition.reduced_formula,
                "target": target_val
            }

            dataset_array.append(info)

        except Exception as e:
            skipped += 1

    print(f"✓ 成功加载: {len(dataset_array)} 样本, 跳过: {skipped} 样本")

    # 提取晶系
    print("\n" + "=" * 80)
    print("3️⃣ 提取晶系信息")
    print("=" * 80)
    crystal_systems, sample_ids = extract_crystal_systems_from_dataset(dataset_array, cif_dir)

    # 筛选目标晶系
    print("\n" + "=" * 80)
    print("4️⃣ 筛选目标晶系")
    print("=" * 80)
    filtered_dataset, filtered_systems, filtered_indices = filter_by_crystal_systems(
        dataset_array, crystal_systems, class_pair
    )

    # 平衡样本数量
    print("\n" + "=" * 80)
    print("5️⃣ 平衡样本数量")
    print("=" * 80)
    balanced_dataset, balanced_systems = balance_samples_by_class(
        filtered_dataset, filtered_systems, class_pair, args.balance_method
    )

    if len(balanced_dataset) < 10:
        print(f"❌ 错误: 平衡后的样本数太少 ({len(balanced_dataset)})，无法进行分析")
        return

    # 创建数据加载器
    print("\n" + "=" * 80)
    print("6️⃣ 创建数据加载器")
    print("=" * 80)

    test_data = get_torch_dataset(
        dataset=balanced_dataset,
        id_tag="jid",
        target="target",
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        use_canonize=False,
        name=f"{args.dataset}_{args.property}_balanced",
        line_graph=True,
        cutoff=8.0,
        max_neighbors=12,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=test_data.collate_line_graph,
    )

    print(f"✓ 数据加载完成: {len(test_data)} 样本")

    # 提取特征
    print("\n" + "=" * 80)
    print("7️⃣ 提取特征")
    print("=" * 80)

    print("无中期融合模型:")
    features_without, targets = extract_features(model_without, test_loader, args.device)

    print("\n有中期融合模型:")
    features_with, _ = extract_features(model_with, test_loader, args.device)

    # 计算聚类指标
    print("\n" + "=" * 80)
    print("8️⃣ 计算聚类指标")
    print("=" * 80)

    metrics_without = compute_clustering_metrics(features_without, balanced_systems)
    metrics_with = compute_clustering_metrics(features_with, balanced_systems)

    print(f"\n无中期融合:")
    for k, v in metrics_without.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n有中期融合:")
    for k, v in metrics_with.items():
        print(f"  {k}: {v:.4f}")

    # 计算类分离度
    print("\n" + "=" * 80)
    print("9️⃣ 计算类分离度")
    print("=" * 80)

    sep_without = compute_class_separation(features_without, balanced_systems, class_pair[0], class_pair[1])
    sep_with = compute_class_separation(features_with, balanced_systems, class_pair[0], class_pair[1])

    print(f"\n无中期融合:")
    print(f"  类间距离: {sep_without['inter_class_dist']:.4f}")
    print(f"  类内距离1 ({class_pair[0]}): {sep_without['intra_class_dist_1']:.4f}")
    print(f"  类内距离2 ({class_pair[1]}): {sep_without['intra_class_dist_2']:.4f}")
    print(f"  分离比率: {sep_without['separation_ratio']:.4f}")

    print(f"\n有中期融合:")
    print(f"  类间距离: {sep_with['inter_class_dist']:.4f}")
    print(f"  类内距离1 ({class_pair[0]}): {sep_with['intra_class_dist_1']:.4f}")
    print(f"  类内距离2 ({class_pair[1]}): {sep_with['intra_class_dist_2']:.4f}")
    print(f"  分离比率: {sep_with['separation_ratio']:.4f}")

    # 降维
    print("\n" + "=" * 80)
    print("🔟 降维可视化")
    print("=" * 80)

    embedded_without = apply_reduction(features_without, method=args.reduction_method, n_components=2)
    embedded_with = apply_reduction(features_with, method=args.reduction_method, n_components=2)

    # 可视化
    print("\n" + "=" * 80)
    print("1️⃣1️⃣ 生成可视化图像")
    print("=" * 80)

    comparison_path = output_dir / f"hard_class_{class_pair[0]}_vs_{class_pair[1]}_balanced.png"
    plot_hard_class_comparison(embedded_without, embedded_with, balanced_systems,
                               metrics_without, metrics_with,
                               sep_without, sep_with,
                               class_pair, comparison_path)

    separation_path = output_dir / f"separation_metrics_{class_pair[0]}_vs_{class_pair[1]}_balanced.png"
    plot_separation_metrics(sep_without, sep_with, class_pair, separation_path)

    # 保存结果摘要
    summary_path = output_dir / f"summary_{class_pair[0]}_vs_{class_pair[1]}_balanced.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("难分样本可视化分析结果 (平衡版本)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"数据集: {args.dataset} - {args.property}\n")
        f.write(f"晶系对: {CRYSTAL_SYSTEMS[class_pair[0]]} vs {CRYSTAL_SYSTEMS[class_pair[1]]}\n")
        f.write(f"平衡方法: {args.balance_method}\n")
        f.write(f"平衡后样本数: {len(balanced_systems)}\n")
        f.write(f"  {CRYSTAL_SYSTEMS[class_pair[0]]}: {balanced_systems.count(class_pair[0])}\n")
        f.write(f"  {CRYSTAL_SYSTEMS[class_pair[1]]}: {balanced_systems.count(class_pair[1])}\n")
        f.write(f"降维方法: {args.reduction_method.upper()}\n\n")

        f.write("=" * 80 + "\n")
        f.write("聚类指标对比\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'指标':<30} {'无融合':<15} {'有融合':<15} {'改进':<15}\n")
        f.write("-" * 80 + "\n")

        for key in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            val_without = metrics_without[key]
            val_with = metrics_with[key]

            if not np.isnan(val_without) and not np.isnan(val_with):
                if key == 'davies_bouldin':
                    improvement = (val_without - val_with) / val_without * 100
                    arrow = "↓" if val_with < val_without else "↑"
                else:
                    improvement = (val_with - val_without) / abs(val_without) * 100
                    arrow = "↑" if val_with > val_without else "↓"

                f.write(f"{key:<30} {val_without:<15.4f} {val_with:<15.4f} "
                       f"{arrow}{abs(improvement):<13.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("类分离度指标\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'指标':<40} {'无融合':<15} {'有融合':<15} {'改进':<15}\n")
        f.write("-" * 80 + "\n")

        # 类间距离
        val_without = sep_without['inter_class_dist']
        val_with = sep_with['inter_class_dist']
        improvement = (val_with - val_without) / val_without * 100
        arrow = "↑" if val_with > val_without else "↓"
        f.write(f"{'Inter-class Distance':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

        # 平均类内距离
        val_without = (sep_without['intra_class_dist_1'] + sep_without['intra_class_dist_2']) / 2
        val_with = (sep_with['intra_class_dist_1'] + sep_with['intra_class_dist_2']) / 2
        improvement = (val_without - val_with) / val_without * 100
        arrow = "↓" if val_with < val_without else "↑"
        f.write(f"{'Avg Intra-class Distance':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

        # 分离比率
        val_without = sep_without['separation_ratio']
        val_with = sep_with['separation_ratio']
        improvement = (val_with - val_without) / val_without * 100
        arrow = "↑" if val_with > val_without else "↓"
        f.write(f"{'Separation Ratio':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

    print(f"✓ 结果摘要已保存: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {output_dir}")
    print(f"  - hard_class_{class_pair[0]}_vs_{class_pair[1]}_balanced.png")
    print(f"  - separation_metrics_{class_pair[0]}_vs_{class_pair[1]}_balanced.png")
    print(f"  - summary_{class_pair[0]}_vs_{class_pair[1]}_balanced.txt")


if __name__ == '__main__':
    main()
