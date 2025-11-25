#!/usr/bin/env python
"""
难分样本可视化的共享工具函数
避免循环导入
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from models.alignn import ALIGNN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


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
