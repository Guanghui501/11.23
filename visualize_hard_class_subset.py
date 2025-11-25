#!/usr/bin/env python
"""
难分样本可视化 (Hard Class Subset)
聚焦于最容易混淆的晶系对，展示模型的区分能力

专注于几何上最相似的晶系对：
- Cubic (立方) vs Tetragonal (四方)
- 可扩展到其他容易混淆的晶系对

使用方法:
    python visualize_hard_class_subset.py \
        --checkpoint_without_fusion model_no_fusion.pth \
        --checkpoint_with_fusion model_with_fusion.pth \
        --data_dir /path/to/dataset \
        --class_pair cubic,tetragonal \
        --output_dir hard_class_results
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 降维和聚类分析
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP未安装，将仅使用t-SNE")

# 导入数据和模型
from jarvis.core.atoms import Atoms
from data import get_torch_dataset
from torch.utils.data import DataLoader
from models.alignn import ALIGNN
from config import TrainingConfig

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['figure.titlesize'] = 19
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


# 晶系定义
CRYSTAL_SYSTEMS = {
    'cubic': 'Cubic',
    'hexagonal': 'Hexagonal',
    'trigonal': 'Trigonal',
    'tetragonal': 'Tetragonal',
    'orthorhombic': 'Orthorhombic',
    'monoclinic': 'Monoclinic',
    'triclinic': 'Triclinic'
}

CRYSTAL_SYSTEM_COLORS = {
    'cubic': '#e74c3c',        # 红色
    'tetragonal': '#f39c12',   # 橙色
    'hexagonal': '#3498db',    # 蓝色
    'trigonal': '#27ae60',     # 绿色
    'orthorhombic': '#9b59b6', # 紫色
    'monoclinic': '#16a085',   # 青色
    'triclinic': '#d35400'     # 深橙色
}


def load_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    print(f"📂 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get('config', None)
    if model_config is None:
        raise ValueError("Checkpoint中未找到模型配置")

    model = ALIGNN(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # 打印中期融合配置
    use_middle = model_config.use_middle_fusion if hasattr(model_config, 'use_middle_fusion') else False
    layers = model_config.middle_fusion_layers if hasattr(model_config, 'middle_fusion_layers') else 'N/A'
    print(f"   中期融合: {use_middle}")
    if use_middle:
        print(f"   融合层: {layers}")

    return model, model_config


def extract_crystal_system_from_text(text):
    """从文本描述中提取晶系关键词"""
    if not text:
        return None

    text_lower = text.lower()
    for crystal_name in ['cubic', 'hexagonal', 'trigonal', 'tetragonal',
                         'orthorhombic', 'monoclinic', 'triclinic']:
        if crystal_name in text_lower:
            return crystal_name
    return None


def extract_crystal_systems_from_dataset(dataset_array, cif_dir):
    """
    从dataset_array中提取晶系信息
    优先从CIF文件提取，失败则从文本描述中提取
    """
    crystal_systems = []
    sample_ids = []
    cif_success = 0
    text_success = 0

    print("🔄 从CIF文件和文本描述中提取晶系信息...")

    for idx, item in enumerate(tqdm(dataset_array, desc="读取晶系")):
        sample_id = item['jid']
        sample_ids.append(sample_id)
        crystal_system = None

        # 方法1: 从CIF文件提取
        try:
            cif_file = os.path.join(cif_dir, f"{sample_id}.cif")
            if os.path.exists(cif_file):
                atoms = Atoms.from_cif(cif_file)

                if hasattr(atoms.lattice, 'lattice_system'):
                    crystal_system = atoms.lattice.lattice_system
                elif hasattr(atoms.lattice, 'get_lattice_system'):
                    crystal_system = atoms.lattice.get_lattice_system()
                elif hasattr(atoms, 'get_spacegroup'):
                    sg = atoms.get_spacegroup()
                    if sg:
                        crystal_system = sg.crystal_system

                if crystal_system:
                    crystal_system = crystal_system.lower()
                    cif_success += 1
        except Exception as e:
            pass

        # 方法2: 从文本描述中提取
        if not crystal_system and 'text' in item:
            crystal_system = extract_crystal_system_from_text(item['text'])
            if crystal_system:
                text_success += 1

        crystal_systems.append(crystal_system if crystal_system else 'unknown')

    print(f"\n✅ 晶系提取完成:")
    print(f"   CIF提取成功: {cif_success}")
    print(f"   文本提取成功: {text_success}")
    print(f"   提取失败(unknown): {len([cs for cs in crystal_systems if cs == 'unknown'])}")

    return crystal_systems, sample_ids


def filter_by_crystal_systems(dataset_array, crystal_systems, target_systems):
    """
    筛选出只包含目标晶系的样本

    Args:
        dataset_array: 原始数据集
        crystal_systems: 晶系列表
        target_systems: 目标晶系列表，如 ['cubic', 'tetragonal']

    Returns:
        filtered_dataset: 过滤后的数据集
        filtered_systems: 过滤后的晶系列表
        filtered_indices: 过滤后的原始索引
    """
    print(f"\n🔍 筛选目标晶系: {', '.join([CRYSTAL_SYSTEMS.get(cs, cs) for cs in target_systems])}")

    filtered_dataset = []
    filtered_systems = []
    filtered_indices = []

    for idx, (item, cs) in enumerate(zip(dataset_array, crystal_systems)):
        if cs in target_systems:
            filtered_dataset.append(item)
            filtered_systems.append(cs)
            filtered_indices.append(idx)

    print(f"✅ 筛选完成:")
    print(f"   原始样本数: {len(dataset_array)}")
    print(f"   筛选后样本数: {len(filtered_dataset)}")

    # 统计各晶系数量
    for cs in target_systems:
        count = filtered_systems.count(cs)
        print(f"   {CRYSTAL_SYSTEMS.get(cs, cs)}: {count}")

    return filtered_dataset, filtered_systems, filtered_indices


def extract_features(model, data_loader, device='cpu'):
    """提取特征"""
    model.eval()
    features_list = []
    targets_list = []

    print("🔄 提取特征...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="处理批次")):
            try:
                if len(batch) == 4:
                    g, lg, text, target = batch
                    model_input = (g.to(device), lg.to(device), text)
                else:
                    g, text, target = batch
                    model_input = (g.to(device), text)

                output = model(model_input, return_features=True)

                # 提取融合特征
                if isinstance(output, dict):
                    if 'fused_features' in output:
                        feat = output['fused_features']
                    elif 'graph_features' in output:
                        feat = output['graph_features']
                    else:
                        feat = output.get('features', None)
                        if feat is None:
                            print(f"⚠️  Batch {batch_idx}: 无法提取特征")
                            continue
                else:
                    feat = output

                features_list.append(feat.cpu().numpy())
                targets_list.append(target.cpu().numpy())

            except Exception as e:
                print(f"⚠️  处理batch {batch_idx}时出错: {e}")
                continue

    features = np.vstack(features_list)
    targets = np.concatenate(targets_list)

    print(f"✅ 提取完成: {features.shape}")
    return features, targets


def compute_clustering_metrics(features, labels):
    """计算聚类质量指标"""
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_int[label] for label in labels])

    if len(np.unique(numeric_labels)) < 2:
        return {'silhouette': np.nan, 'davies_bouldin': np.nan, 'calinski_harabasz': np.nan}

    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(features, numeric_labels)
    except:
        metrics['silhouette'] = np.nan

    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features, numeric_labels)
    except:
        metrics['davies_bouldin'] = np.nan

    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(features, numeric_labels)
    except:
        metrics['calinski_harabasz'] = np.nan

    return metrics


def compute_class_separation(features, labels, class1, class2):
    """
    计算两个类别之间的分离度

    Returns:
        inter_class_dist: 类间距离（均值之间的距离）
        intra_class_dist_1: 类1的类内距离（平均距离）
        intra_class_dist_2: 类2的类内距离（平均距离）
        separation_ratio: 分离比率 = inter_class_dist / (intra_class_dist_1 + intra_class_dist_2)
    """
    mask1 = np.array(labels) == class1
    mask2 = np.array(labels) == class2

    feat1 = features[mask1]
    feat2 = features[mask2]

    # 类间距离 (质心之间的距离)
    centroid1 = feat1.mean(axis=0)
    centroid2 = feat2.mean(axis=0)
    inter_class_dist = np.linalg.norm(centroid1 - centroid2)

    # 类内距离 (每个样本到自己类质心的平均距离)
    intra_class_dist_1 = np.mean([np.linalg.norm(f - centroid1) for f in feat1])
    intra_class_dist_2 = np.mean([np.linalg.norm(f - centroid2) for f in feat2])

    # 分离比率
    separation_ratio = inter_class_dist / (intra_class_dist_1 + intra_class_dist_2)

    return {
        'inter_class_dist': inter_class_dist,
        'intra_class_dist_1': intra_class_dist_1,
        'intra_class_dist_2': intra_class_dist_2,
        'separation_ratio': separation_ratio
    }


def apply_reduction(features, method='tsne', n_components=2):
    """降维"""
    print(f"🔄 应用{method.upper()}降维...")

    if method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(30, len(features) - 1),
            random_state=42,
            max_iter=1000
        )
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(features) - 1),
            min_dist=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    embedded = reducer.fit_transform(features)
    print(f"✅ 降维完成: {embedded.shape}")
    return embedded


def plot_hard_class_comparison(embedded_without, embedded_with, crystal_systems,
                                metrics_without, metrics_with,
                                sep_without, sep_with,
                                class_pair, output_path):
    """
    创建难分样本对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    class1, class2 = class_pair

    for idx, (embedded, metrics, sep, title) in enumerate([
        (embedded_without, metrics_without, sep_without, 'Without Middle Fusion'),
        (embedded_with, metrics_with, sep_with, 'With Middle Fusion')
    ]):
        ax = axes[idx]

        # 绘制两个类别
        for cs in [class1, class2]:
            mask = np.array(crystal_systems) == cs
            if mask.sum() == 0:
                continue

            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=CRYSTAL_SYSTEM_COLORS.get(cs, 'gray'),
                label=f"{CRYSTAL_SYSTEMS.get(cs, cs)} (n={mask.sum()})",
                alpha=0.7,
                s=80,  # 更大的点
                edgecolors='white',
                linewidths=1.0
            )

        ax.set_xlabel('Dimension 1', fontsize=16)
        ax.set_ylabel('Dimension 2', fontsize=16)

        # 增强的标题，包含更多指标
        title_text = f'{title}\n'
        title_text += f'Silhouette: {metrics["silhouette"]:.3f} | '
        title_text += f'Separation Ratio: {sep["separation_ratio"]:.3f}\n'
        title_text += f'Inter-class Dist: {sep["inter_class_dist"]:.2f} | '
        title_text += f'Intra-class Dist: {(sep["intra_class_dist_1"] + sep["intra_class_dist_2"])/2:.2f}'

        ax.set_title(title_text, fontsize=15, pad=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=14, markerscale=1.5)
        ax.grid(True, alpha=0.3)

    class1_name = CRYSTAL_SYSTEMS.get(class1, class1)
    class2_name = CRYSTAL_SYSTEMS.get(class2, class2)
    plt.suptitle(f'Hard Class Subset Visualization: {class1_name} vs {class2_name}',
                 fontsize=19, y=0.98, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存: {output_path}")
    plt.close()


def plot_separation_metrics(sep_without, sep_with, class_pair, output_path):
    """绘制类分离度指标对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class1_name = CRYSTAL_SYSTEMS.get(class_pair[0], class_pair[0])
    class2_name = CRYSTAL_SYSTEMS.get(class_pair[1], class_pair[1])

    # 1. 类间距离对比
    ax = axes[0]
    values = [sep_without['inter_class_dist'], sep_with['inter_class_dist']]
    bars = ax.bar(['Without Fusion', 'With Fusion'], values,
                 color=['#3498db', '#e74c3c'], alpha=0.75, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inter-class Distance', fontsize=14)
    ax.set_title('Inter-class Distance\n(Higher is Better)', fontsize=15, pad=12)
    ax.grid(True, axis='y', alpha=0.3)

    # 2. 平均类内距离对比
    ax = axes[1]
    intra_without = (sep_without['intra_class_dist_1'] + sep_without['intra_class_dist_2']) / 2
    intra_with = (sep_with['intra_class_dist_1'] + sep_with['intra_class_dist_2']) / 2
    values = [intra_without, intra_with]
    bars = ax.bar(['Without Fusion', 'With Fusion'], values,
                 color=['#3498db', '#e74c3c'], alpha=0.75, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Intra-class Distance', fontsize=14)
    ax.set_title('Average Intra-class Distance\n(Lower is Better)', fontsize=15, pad=12)
    ax.grid(True, axis='y', alpha=0.3)

    # 3. 分离比率对比
    ax = axes[2]
    values = [sep_without['separation_ratio'], sep_with['separation_ratio']]
    bars = ax.bar(['Without Fusion', 'With Fusion'], values,
                 color=['#3498db', '#e74c3c'], alpha=0.75, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Separation Ratio', fontsize=14)
    ax.set_title('Separation Ratio\n(Higher is Better)', fontsize=15, pad=12)
    ax.grid(True, axis='y', alpha=0.3)

    # 如果有改进，给背景加绿色
    if sep_with['separation_ratio'] > sep_without['separation_ratio']:
        ax.set_facecolor('#eafaf1')

    plt.suptitle(f'Class Separation Metrics: {class1_name} vs {class2_name}',
                 fontsize=16, y=1.00, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 分离度指标图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='难分样本可视化 (Hard Class Subset)')
    parser.add_argument('--checkpoint_without_fusion', type=str, required=True,
                       help='无中期融合的模型checkpoint')
    parser.add_argument('--checkpoint_with_fusion', type=str, required=True,
                       help='有中期融合的模型checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集目录（包含CIF文件）')
    parser.add_argument('--dataset', type=str, default='jarvis',
                       help='数据集类型')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='目标属性')
    parser.add_argument('--class_pair', type=str, default='cubic,tetragonal',
                       help='要对比的晶系对，用逗号分隔，如 "cubic,tetragonal"')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='从数据集中采样的初始样本数（在筛选晶系之前）')
    parser.add_argument('--reduction_method', type=str, default='tsne',
                       choices=['tsne', 'umap'], help='降维方法')
    parser.add_argument('--output_dir', type=str, default='hard_class_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    # 解析晶系对
    class_pair = [cs.strip().lower() for cs in args.class_pair.split(',')]
    if len(class_pair) != 2:
        raise ValueError("--class_pair 必须指定两个晶系，用逗号分隔")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("难分样本可视化 (Hard Class Subset)")
    print("=" * 80)
    print(f"目标晶系对: {CRYSTAL_SYSTEMS.get(class_pair[0], class_pair[0])} vs "
          f"{CRYSTAL_SYSTEMS.get(class_pair[1], class_pair[1])}")
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
        import random
        random.seed(42)
        data = random.sample(data, args.n_samples)
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

    if len(filtered_dataset) < 10:
        print(f"❌ 错误: 筛选后的样本数太少 ({len(filtered_dataset)})，无法进行分析")
        return

    # 创建数据加载器
    print("\n" + "=" * 80)
    print("5️⃣ 创建数据加载器")
    print("=" * 80)

    test_data = get_torch_dataset(
        dataset=filtered_dataset,
        id_tag="jid",
        target="target",
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        use_canonize=False,
        name=f"{args.dataset}_{args.property}_hard_class",
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
    print("6️⃣ 提取特征")
    print("=" * 80)

    print("无中期融合模型:")
    features_without, targets = extract_features(model_without, test_loader, args.device)

    print("\n有中期融合模型:")
    features_with, _ = extract_features(model_with, test_loader, args.device)

    # 计算聚类指标
    print("\n" + "=" * 80)
    print("7️⃣ 计算聚类指标")
    print("=" * 80)

    metrics_without = compute_clustering_metrics(features_without, filtered_systems)
    metrics_with = compute_clustering_metrics(features_with, filtered_systems)

    print(f"\n无中期融合:")
    for k, v in metrics_without.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n有中期融合:")
    for k, v in metrics_with.items():
        print(f"  {k}: {v:.4f}")

    # 计算类分离度
    print("\n" + "=" * 80)
    print("8️⃣ 计算类分离度")
    print("=" * 80)

    sep_without = compute_class_separation(features_without, filtered_systems, class_pair[0], class_pair[1])
    sep_with = compute_class_separation(features_with, filtered_systems, class_pair[0], class_pair[1])

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
    print("9️⃣ 降维可视化")
    print("=" * 80)

    embedded_without = apply_reduction(features_without, method=args.reduction_method, n_components=2)
    embedded_with = apply_reduction(features_with, method=args.reduction_method, n_components=2)

    # 可视化
    print("\n" + "=" * 80)
    print("🔟 生成可视化图像")
    print("=" * 80)

    comparison_path = output_dir / f"hard_class_{class_pair[0]}_vs_{class_pair[1]}.png"
    plot_hard_class_comparison(embedded_without, embedded_with, filtered_systems,
                               metrics_without, metrics_with,
                               sep_without, sep_with,
                               class_pair, comparison_path)

    separation_path = output_dir / f"separation_metrics_{class_pair[0]}_vs_{class_pair[1]}.png"
    plot_separation_metrics(sep_without, sep_with, class_pair, separation_path)

    # 保存结果摘要
    summary_path = output_dir / f"summary_{class_pair[0]}_vs_{class_pair[1]}.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("难分样本可视化分析结果 (Hard Class Subset)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"数据集: {args.dataset} - {args.property}\n")
        f.write(f"晶系对: {CRYSTAL_SYSTEMS.get(class_pair[0], class_pair[0])} vs "
                f"{CRYSTAL_SYSTEMS.get(class_pair[1], class_pair[1])}\n")
        f.write(f"样本数: {len(filtered_systems)}\n")
        f.write(f"  {CRYSTAL_SYSTEMS.get(class_pair[0], class_pair[0])}: "
                f"{filtered_systems.count(class_pair[0])}\n")
        f.write(f"  {CRYSTAL_SYSTEMS.get(class_pair[1], class_pair[1])}: "
                f"{filtered_systems.count(class_pair[1])}\n")
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
        f.write(f"{'Inter-class Distance (类间距离)':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

        # 平均类内距离
        val_without = (sep_without['intra_class_dist_1'] + sep_without['intra_class_dist_2']) / 2
        val_with = (sep_with['intra_class_dist_1'] + sep_with['intra_class_dist_2']) / 2
        improvement = (val_without - val_with) / val_without * 100  # 类内距离越小越好
        arrow = "↓" if val_with < val_without else "↑"
        f.write(f"{'Avg Intra-class Distance (平均类内距离)':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

        # 分离比率
        val_without = sep_without['separation_ratio']
        val_with = sep_with['separation_ratio']
        improvement = (val_with - val_without) / val_without * 100
        arrow = "↑" if val_with > val_without else "↓"
        f.write(f"{'Separation Ratio (分离比率)':<40} {val_without:<15.4f} "
               f"{val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")

    print(f"✓ 结果摘要已保存: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {output_dir}")
    print(f"  - hard_class_{class_pair[0]}_vs_{class_pair[1]}.png : 难分样本对比图")
    print(f"  - separation_metrics_{class_pair[0]}_vs_{class_pair[1]}.png : 分离度指标图")
    print(f"  - summary_{class_pair[0]}_vs_{class_pair[1]}.txt : 结果摘要")


if __name__ == '__main__':
    main()
