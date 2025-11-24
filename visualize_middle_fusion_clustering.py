#!/usr/bin/env python
"""
中期融合特征空间可视化 - 按晶系聚类分析
对比有/无中期融合的特征聚类质量

使用方法:
    python visualize_middle_fusion_clustering.py \
        --checkpoint_without_fusion model_no_fusion.pth \
        --checkpoint_with_fusion model_with_fusion.pth \
        --data_dir /path/to/dataset \
        --output_dir fusion_clustering_analysis
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

# 降维算法
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP未安装，将仅使用t-SNE")

# 导入数据和模型
from jarvis.core.atoms import Atoms
from data import get_train_val_loaders
from models.alignn import ALIGNN
from config import TrainingConfig

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 10


# 晶系定义
CRYSTAL_SYSTEMS = {
    'cubic': '立方',
    'hexagonal': '六方',
    'trigonal': '三方',
    'tetragonal': '四方',
    'orthorhombic': '正交',
    'monoclinic': '单斜',
    'triclinic': '三斜'
}

CRYSTAL_SYSTEM_COLORS = {
    'cubic': '#e74c3c',        # 红色
    'hexagonal': '#3498db',    # 蓝色
    'trigonal': '#2ecc71',     # 绿色
    'tetragonal': '#f39c12',   # 橙色
    'orthorhombic': '#9b59b6', # 紫色
    'monoclinic': '#1abc9c',   # 青色
    'triclinic': '#e67e22'     # 深橙色
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


def extract_crystal_systems_from_dataset(dataset_array, cif_dir):
    """
    从dataset_array中提取晶系信息

    Returns:
        crystal_systems: 晶系列表（与dataset_array顺序对应）
        sample_ids: 样本ID列表
    """
    crystal_systems = []
    sample_ids = []

    print("🔄 从CIF文件提取晶系信息...")

    for item in tqdm(dataset_array, desc="读取晶系"):
        sample_id = item['jid']
        sample_ids.append(sample_id)

        try:
            cif_file = os.path.join(cif_dir, f"{sample_id}.cif")
            if os.path.exists(cif_file):
                atoms = Atoms.from_cif(cif_file)
                crystal_system = atoms.lattice.lattice_system
                crystal_systems.append(crystal_system)
            else:
                crystal_systems.append('unknown')
        except Exception as e:
            crystal_systems.append('unknown')

    print(f"✅ 晶系提取完成:")
    print(f"   总样本数: {len(crystal_systems)}")
    print(f"   晶系分布:")
    for cs in sorted(set(crystal_systems)):
        count = crystal_systems.count(cs)
        print(f"     {CRYSTAL_SYSTEMS.get(cs, cs)}: {count}")

    return crystal_systems, sample_ids


def extract_features(model, data_loader, device='cpu'):
    """
    提取特征

    Returns:
        features: 特征矩阵 [n_samples, n_features]
        targets: 目标值
    """
    model.eval()
    features_list = []
    targets_list = []

    print("🔄 提取特征...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="处理批次")):
            try:
                # 解包batch
                if len(batch) == 4:
                    g, lg, text, target = batch
                    model_input = (g.to(device), lg.to(device), text)
                else:
                    g, text, target = batch
                    model_input = (g.to(device), text)

                # 前向传播获取特征
                output = model(model_input, return_features=True)

                # 提取融合特征
                if isinstance(output, dict):
                    if 'fused_features' in output:
                        feat = output['fused_features']
                    elif 'graph_features' in output:
                        feat = output['graph_features']
                    else:
                        # 尝试从输出中获取最后的特征
                        feat = output.get('features', None)
                        if feat is None:
                            print(f"⚠️  Batch {batch_idx}: 无法提取特征")
                            continue
                else:
                    # 如果返回的不是字典，尝试直接使用
                    feat = output

                features_list.append(feat.cpu().numpy())
                targets_list.append(target.cpu().numpy())

            except Exception as e:
                print(f"⚠️  处理batch {batch_idx}时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 合并所有特征
    features = np.vstack(features_list)
    targets = np.concatenate(targets_list)

    print(f"✅ 提取完成:")
    print(f"   特征维度: {features.shape}")
    print(f"   样本数: {len(features)}")

    return features, targets


def compute_clustering_metrics(features, labels):
    """计算聚类质量指标"""
    # 将字符串标签转为数值
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_int[label] for label in labels])

    # 过滤掉unknown标签
    valid_mask = np.array(labels) != 'unknown'
    if valid_mask.sum() < 2:
        return {'silhouette': np.nan, 'davies_bouldin': np.nan, 'calinski_harabasz': np.nan}

    features_valid = features[valid_mask]
    labels_valid = numeric_labels[valid_mask]

    # 确保至少有2个类别
    if len(np.unique(labels_valid)) < 2:
        return {'silhouette': np.nan, 'davies_bouldin': np.nan, 'calinski_harabasz': np.nan}

    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(features_valid, labels_valid)
    except:
        metrics['silhouette'] = np.nan

    try:
        metrics['davies_bouldin'] = davies_bouldin_score(features_valid, labels_valid)
    except:
        metrics['davies_bouldin'] = np.nan

    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(features_valid, labels_valid)
    except:
        metrics['calinski_harabasz'] = np.nan

    return metrics


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


def plot_comparison(embedded_without, embedded_with, crystal_systems,
                   metrics_without, metrics_with, output_path):
    """
    创建对比图：有无中期融合的特征聚类对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 过滤掉unknown的样本用于绘图
    valid_mask = np.array(crystal_systems) != 'unknown'

    for idx, (embedded, metrics, title) in enumerate([
        (embedded_without, metrics_without, '无中期融合'),
        (embedded_with, metrics_with, '有中期融合')
    ]):
        ax = axes[idx]

        # 绘制每个晶系
        for cs in set(crystal_systems):
            if cs == 'unknown':
                continue

            mask = (np.array(crystal_systems) == cs) & valid_mask
            if mask.sum() == 0:
                continue

            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=CRYSTAL_SYSTEM_COLORS.get(cs, 'gray'),
                label=CRYSTAL_SYSTEMS.get(cs, cs),
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidths=0.5
            )

        ax.set_xlabel('维度 1', fontsize=12)
        ax.set_ylabel('维度 2', fontsize=12)
        ax.set_title(f'{title}\n' +
                    f'Silhouette: {metrics["silhouette"]:.3f} | ' +
                    f'DB: {metrics["davies_bouldin"]:.3f} | ' +
                    f'CH: {metrics["calinski_harabasz"]:.1f}',
                    fontsize=13, pad=15)
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('特征空间聚类对比 - 按晶系分组', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存: {output_path}")
    plt.close()


def plot_metrics_comparison(metrics_without, metrics_with, output_path):
    """绘制聚类指标对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metric_names = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score']
    metric_keys = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

    colors = ['#3498db', '#e74c3c']

    for idx, (name, key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[idx]
        values = [metrics_without[key], metrics_with[key]]
        bars = ax.bar(['无中期融合', '有中期融合'], values, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12, pad=10)
        ax.grid(True, axis='y', alpha=0.3)

        # Davies-Bouldin: 越低越好
        if key == 'davies_bouldin':
            if values[1] < values[0]:
                ax.set_facecolor('#eafaf1')  # 绿色背景表示改进

    plt.suptitle('聚类质量指标对比\n(Silhouette和CH越高越好，DB越低越好)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 指标对比图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='中期融合特征聚类可视化')
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
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='用于可视化的样本数量')
    parser.add_argument('--reduction_method', type=str, default='tsne',
                       choices=['tsne', 'umap'], help='降维方法')
    parser.add_argument('--output_dir', type=str, default='fusion_clustering_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("中期融合特征空间聚类分析")
    print("=" * 80)

    # 构建CIF目录路径
    cif_dir = os.path.join(args.data_dir, f'{args.dataset}/{args.property}/cif/')

    # 加载数据（使用测试集）
    print("\n📊 加载数据集...")
    # 这里简化处理，实际使用时需要根据你的数据加载逻辑调整
    # 为了演示，假设使用test_loader

    # 加载两个模型
    print("\n" + "=" * 80)
    print("1️⃣ 加载无中期融合模型")
    print("=" * 80)
    model_without, config_without = load_model(args.checkpoint_without_fusion, args.device)

    print("\n" + "=" * 80)
    print("2️⃣ 加载有中期融合模型")
    print("=" * 80)
    model_with, config_with = load_model(args.checkpoint_with_fusion, args.device)

    # 加载数据
    print("\n" + "=" * 80)
    print("3️⃣ 加载测试数据")
    print("=" * 80)

    # 使用train_mbj_with_optuna.py中的load_dataset函数
    try:
        import sys
        import csv
        from tqdm import tqdm

        # 构建数据路径
        id_prop_file = os.path.join(args.data_dir, f'{args.dataset}/{args.property}/description.csv')

        print(f"CIF目录: {cif_dir}")
        print(f"描述文件: {id_prop_file}")

        # 简化的数据加载（直接加载，不需要复杂的文本处理）
        print("加载数据集...")
        with open(id_prop_file, 'r') as f:
            reader = csv.reader(f)
            headings = next(reader)  # 跳过表头
            data = [row for row in reader]

        print(f"总样本数: {len(data)}")

        # 限制样本数量
        if len(data) > args.n_samples:
            import random
            random.seed(42)
            data = random.sample(data, args.n_samples)
            print(f"随机选择 {args.n_samples} 个样本用于可视化")

        # 构建dataset_array
        dataset_array = []
        skipped = 0

        for j in tqdm(range(len(data)), desc="加载样本"):
            try:
                if args.dataset.lower() == 'jarvis':
                    # JARVIS格式: id, composition, target, description, file_name
                    sample_id = data[j][0]
                    composition = data[j][1]
                    target_val = float(data[j][2])
                    description = data[j][3]
                else:
                    # 其他格式
                    sample_id = data[j][0]
                    target_val = float(data[j][1])
                    description = ""

                # 读取CIF文件
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
                if skipped <= 5:
                    print(f"跳过样本 {j}: {e}")

        print(f"✓ 成功加载: {len(dataset_array)} 样本, 跳过: {skipped} 样本")

        if len(dataset_array) == 0:
            raise ValueError("没有成功加载任何样本！")

        # 调试：检查数据结构
        print(f"\n调试信息:")
        print(f"  数据类型: {type(dataset_array)}")
        print(f"  第一个样本的键: {list(dataset_array[0].keys())}")
        print(f"  第一个样本的target值: {dataset_array[0]['target']}")

        # 创建数据加载器 - 直接创建测试集，避免空的训练/验证集问题
        print("\n创建数据加载器...")

        # 使用 get_torch_dataset 直接创建数据集
        from data import get_torch_dataset
        from torch.utils.data import DataLoader

        test_data = get_torch_dataset(
            dataset=dataset_array,
            id_tag="jid",
            target="target",
            neighbor_strategy="k-nearest",
            atom_features="cgcnn",
            use_canonize=False,
            name=f"{args.dataset}_{args.property}",
            line_graph=True,
            cutoff=8.0,
            max_neighbors=12,
        )

        # 创建 DataLoader
        test_loader = DataLoader(
            test_data,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=test_data.collate_line_graph,
        )

        # prepare_batch 函数
        def prepare_batch(batch, device=args.device):
            """准备批次数据"""
            g, lg, text, target = batch
            g = g.to(device)
            lg = lg.to(device)
            target = target.to(device)
            return (g, lg, text), target

        print(f"✓ 数据加载完成: {len(test_data)} 样本")

        # 提取晶系信息（在创建data loader之前，从原始dataset_array）
        print("\n" + "=" * 80)
        print("4️⃣ 提取晶系信息")
        print("=" * 80)
        crystal_systems, sample_ids = extract_crystal_systems_from_dataset(dataset_array, cif_dir)

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查数据路径和格式")
        return

    # 提取特征 - 无中期融合
    print("\n" + "=" * 80)
    print("5️⃣ 提取特征 - 无中期融合模型")
    print("=" * 80)
    features_without, targets = extract_features(
        model_without, test_loader, args.device
    )

    # 提取特征 - 有中期融合
    print("\n" + "=" * 80)
    print("6️⃣ 提取特征 - 有中期融合模型")
    print("=" * 80)
    features_with, _ = extract_features(
        model_with, test_loader, args.device
    )

    # 计算聚类指标
    print("\n" + "=" * 80)
    print("7️⃣ 计算聚类指标")
    print("=" * 80)

    print("无中期融合:")
    metrics_without = compute_clustering_metrics(features_without, crystal_systems)
    for metric, value in metrics_without.items():
        print(f"  {metric}: {value:.4f}")

    print("\n有中期融合:")
    metrics_with = compute_clustering_metrics(features_with, crystal_systems)
    for metric, value in metrics_with.items():
        print(f"  {metric}: {value:.4f}")

    # 降维
    print("\n" + "=" * 80)
    print("8️⃣ 降维可视化")
    print("=" * 80)

    embedded_without = apply_reduction(features_without, method=args.reduction_method, n_components=2)
    embedded_with = apply_reduction(features_with, method=args.reduction_method, n_components=2)

    # 创建可视化
    print("\n" + "=" * 80)
    print("9️⃣ 生成可视化图像")
    print("=" * 80)

    comparison_path = output_dir / "clustering_comparison.png"
    plot_comparison(embedded_without, embedded_with, crystal_systems,
                   metrics_without, metrics_with, comparison_path)

    metrics_path = output_dir / "metrics_comparison.png"
    plot_metrics_comparison(metrics_without, metrics_with, metrics_path)

    # 保存结果摘要
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("中期融合特征聚类分析结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"数据集: {args.dataset} - {args.property}\n")
        f.write(f"样本数: {len(crystal_systems)}\n")
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

                f.write(f"{key:<30} {val_without:<15.4f} {val_with:<15.4f} {arrow}{abs(improvement):<13.1f}%\n")
            else:
                f.write(f"{key:<30} {val_without:<15} {val_with:<15} {'N/A':<15}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("晶系分布\n")
        f.write("=" * 80 + "\n\n")

        for cs in sorted(set(crystal_systems)):
            count = crystal_systems.count(cs)
            f.write(f"  {CRYSTAL_SYSTEMS.get(cs, cs):<15} {count:>6} 样本\n")

    print(f"✓ 结果摘要已保存: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n结果保存在: {output_dir}")
    print(f"  - clustering_comparison.png : 聚类对比图")
    print(f"  - metrics_comparison.png    : 指标对比图")
    print(f"  - summary.txt               : 结果摘要")


if __name__ == '__main__':
    main()
