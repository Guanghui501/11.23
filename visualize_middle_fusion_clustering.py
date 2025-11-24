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


def extract_features_with_crystal_system(model, data_loader, cif_dir, device='cpu'):
    """
    提取特征并获取晶系信息

    Returns:
        features: 特征矩阵 [n_samples, n_features]
        crystal_systems: 晶系列表
        targets: 目标值
        sample_ids: 样本ID
    """
    model.eval()
    features_list = []
    crystal_systems = []
    targets_list = []
    sample_ids = []

    print("🔄 提取特征和晶系信息...")

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

                # 获取样本ID和晶系（从graph中）
                batch_crystal_systems = []
                for i in range(g.batch_size):
                    try:
                        # 尝试从graph的节点数据中获取ID
                        if hasattr(g, 'ndata') and 'id' in g.ndata:
                            sample_id = g.ndata['id'][i].item()
                        else:
                            sample_id = f"sample_{batch_idx}_{i}"

                        sample_ids.append(sample_id)

                        # 从CIF文件读取晶系
                        if cif_dir and os.path.exists(cif_dir):
                            cif_file = os.path.join(cif_dir, f"{sample_id}.cif")
                            if os.path.exists(cif_file):
                                atoms = Atoms.from_cif(cif_file)
                                crystal_system = atoms.lattice.lattice_system
                                batch_crystal_systems.append(crystal_system)
                            else:
                                batch_crystal_systems.append('unknown')
                        else:
                            batch_crystal_systems.append('unknown')
                    except Exception as e:
                        batch_crystal_systems.append('unknown')
                        sample_ids.append(f"sample_{batch_idx}_{i}")

                crystal_systems.extend(batch_crystal_systems)

            except Exception as e:
                print(f"⚠️  处理batch {batch_idx}时出错: {e}")
                continue

    # 合并所有特征
    features = np.vstack(features_list)
    targets = np.concatenate(targets_list)

    print(f"✅ 提取完成:")
    print(f"   特征维度: {features.shape}")
    print(f"   样本数: {len(crystal_systems)}")
    print(f"   晶系分布:")
    for cs in set(crystal_systems):
        count = crystal_systems.count(cs)
        print(f"     {CRYSTAL_SYSTEMS.get(cs, cs)}: {count}")

    return features, crystal_systems, targets, sample_ids


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

    # 注意：这里需要根据你的实际数据加载方式调整
    # 建议提供一个简单的dataloader
    print("\n⚠️  注意：请确保提供了有效的数据加载器")
    print("示例代码需要根据你的实际数据格式进行调整")

    # TODO: 实际使用时，在这里加载你的test_loader
    # test_loader = ...

    print("\n✅ 分析完成！结果保存在:", output_dir)


if __name__ == '__main__':
    main()
