#!/usr/bin/env python
"""
数据集预处理脚本

将原始CIF文件和CSV描述文件转换为预处理的pickle文件，大幅加快后续训练和评估的加载速度。

用法示例:
    # JARVIS数据集 - MBJ Band Gap
    python preprocess_dataset.py \
        --dataset jarvis \
        --property mbj_bandgap \
        --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
        --output_dir /public/home/ghzhang/preprocessed_data

    # JARVIS数据集 - Formation Energy
    python preprocess_dataset.py \
        --dataset jarvis \
        --property formation_energy \
        --root_dir ./crysmmnet-main/dataset \
        --output_dir ./preprocessed_data

    # Material Project数据集
    python preprocess_dataset.py \
        --dataset mp \
        --property band_gap \
        --root_dir ./crysmmnet-main/dataset \
        --output_dir ./preprocessed_data

输出文件结构:
    output_dir/
        jarvis/
            mbj_bandgap/
                train.pkl
                val.pkl
                test.pkl
        mp/
            band_gap/
                train.pkl
                val.pkl
                test.pkl
"""

import os
import sys
import csv
import pickle
import argparse
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from jarvis.core.atoms import Atoms
from graphs import Graph
from tokenizers.normalizers import BertNormalizer


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def get_dataset_paths(root_dir, dataset, property_name):
    """获取数据集路径"""
    if dataset.lower() == 'jarvis':
        cif_dir = os.path.join(root_dir, f'jarvis/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'jarvis/{property_name}/description.csv')

    elif dataset.lower() == 'mp':
        cif_dir = os.path.join(root_dir, f'mp/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'mp/{property_name}/description.csv')

    elif dataset.lower() == 'class':
        cif_dir = os.path.join(root_dir, f'class/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'class/{property_name}/description.csv')

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return cif_dir, id_prop_file


def load_vocab_mappings():
    """加载词汇映射文件"""
    possible_paths = [
        'vocab_mappings.txt',
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),
        os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src/vocab_mappings.txt'),
        '../vocab_mappings.txt',
        '../../vocab_mappings.txt',
    ]

    vocab_file = None
    for path in possible_paths:
        if os.path.exists(path):
            vocab_file = path
            break

    if vocab_file is None:
        raise FileNotFoundError(
            "无法找到 vocab_mappings.txt 文件。请确保：\n"
            "1. 文件存在于当前目录或 crysmmnet-main/src/ 目录\n"
            "2. 当前工作目录正确\n"
            f"尝试过的路径: {possible_paths}"
        )

    print(f"使用词汇映射文件: {vocab_file}")
    with open(vocab_file, 'r') as f:
        mappings = f.read().strip().split('\n')
    mappings = {m[0]: m[2:] for m in mappings}

    return mappings


def normalize_text(text, norm, mappings):
    """归一化文本"""
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)


def load_raw_data(cif_dir, id_prop_file, dataset, property_name, norm, mappings):
    """加载原始数据"""
    print(f"\n{'='*80}")
    print(f"加载原始数据: {dataset} - {property_name}")
    print(f"CIF目录: {cif_dir}")
    print(f"描述文件: {id_prop_file}")
    print(f"{'='*80}\n")

    # 检查路径
    if not os.path.exists(cif_dir):
        raise FileNotFoundError(f"CIF目录不存在: {cif_dir}")
    if not os.path.exists(id_prop_file):
        raise FileNotFoundError(f"描述文件不存在: {id_prop_file}")

    # 读取CSV
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"CSV文件包含 {len(data)} 行数据")

    # 加载数据
    dataset_array = []
    skipped = 0

    for j in tqdm(range(len(data)), desc="加载数据"):
        try:
            # 根据数据集类型解析CSV
            if dataset.lower() == 'mp':
                if property_name == 'formation_energy':
                    id, composition, target, _, crys_desc_full, _ = data[j]
                elif property_name == 'band_gap':
                    id, composition, _, target, crys_desc_full, _ = data[j]
                elif property_name == 'shear':
                    id, composition, target, _, crys_desc_full, _ = data[j]
                elif property_name in ['bulk', 'bulk_modulus']:
                    id, composition, _, target, crys_desc_full, _ = data[j]
                else:
                    # 通用MP格式
                    id, composition, target, crys_desc_full = data[j][0], data[j][1], data[j][2], data[j][-2]
            elif dataset.lower() == 'jarvis':
                id, composition, target, crys_desc_full, _ = data[j]
            elif dataset.lower() == 'class':
                id, target, crys_desc_full = data[j]
                composition = ''
            else:
                raise ValueError(f"Unknown dataset format: {dataset}")

            # 读取CIF文件
            file_path = os.path.join(cif_dir, f'{id}.cif')
            if not os.path.exists(file_path):
                skipped += 1
                if skipped <= 5:
                    print(f"  跳过: CIF文件不存在 - {id}")
                continue

            atoms = Atoms.from_cif(file_path)

            # 归一化文本
            normalized_text = normalize_text(crys_desc_full, norm, mappings)

            # 构建样本
            info = {
                "id": id,
                "atoms": atoms.to_dict(),
                "text": normalized_text,
                "target": float(target)
            }

            # MP数据集的特殊处理（对数变换）
            if dataset.lower() == 'mp' and property_name in ['shear', 'bulk', 'bulk_modulus', 'shear_modulus']:
                info["target"] = np.log10(float(target))

            dataset_array.append(info)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  跳过样本 {data[j][0] if len(data[j]) > 0 else 'unknown'}: {e}")

    print(f"\n✓ 成功加载: {len(dataset_array)} 样本")
    if skipped > 0:
        print(f"⚠ 跳过: {skipped} 样本\n")

    return dataset_array


def build_graphs(dataset_array, cutoff=8.0, max_neighbors=12, use_canonize=True):
    """构建DGL图"""
    print(f"\n{'='*80}")
    print(f"构建DGL图")
    print(f"  Cutoff: {cutoff} Å")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  Use canonize: {use_canonize}")
    print(f"{'='*80}\n")

    processed_data = []
    failed = 0

    for sample in tqdm(dataset_array, desc="构建图"):
        try:
            atoms = Atoms.from_dict(sample['atoms'])

            # 构建atom graph 和 line graph
            g, lg = Graph.atom_dgl_multigraph(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                atom_features="cgcnn",
                compute_line_graph=True,
                use_canonize=use_canonize
            )

            processed_sample = {
                "id": sample["id"],
                "graph": (g,),  # 保持元组格式以兼容训练代码
                "line_graph": lg,
                "text": sample["text"],
                "target": sample["target"]
            }

            processed_data.append(processed_sample)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  构建图失败 {sample['id']}: {e}")

    print(f"\n✓ 成功构建: {len(processed_data)} 个图")
    if failed > 0:
        print(f"⚠ 失败: {failed} 个样本\n")

    return processed_data


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """分割数据集"""
    print(f"\n{'='*80}")
    print(f"分割数据集")
    print(f"  训练集比例: {train_ratio}")
    print(f"  验证集比例: {val_ratio}")
    print(f"  测试集比例: {test_ratio}")
    print(f"  随机种子: {seed}")
    print(f"{'='*80}\n")

    # 设置随机种子
    random.seed(seed)

    # 计算分割点
    n_total = len(data)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  # 剩余的都给测试集

    # 随机打乱
    indices = list(range(n_total))
    random.shuffle(indices)

    # 分割
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    print(f"测试集: {len(test_data)} 样本\n")

    return train_data, val_data, test_data


def save_preprocessed_data(train_data, val_data, test_data, output_dir, dataset, property_name):
    """保存预处理数据"""
    print(f"\n{'='*80}")
    print(f"保存预处理数据")
    print(f"{'='*80}\n")

    # 创建输出目录
    save_dir = os.path.join(output_dir, dataset.lower(), property_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存三个split
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        pkl_file = os.path.join(save_dir, f'{split_name}.pkl')

        with open(pkl_file, 'wb') as f:
            pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 获取文件大小
        file_size = os.path.getsize(pkl_file) / (1024 * 1024)  # MB

        print(f"✓ {split_name:>5} 集: {pkl_file}")
        print(f"        样本数: {len(split_data)}, 文件大小: {file_size:.2f} MB")

    print(f"\n所有预处理数据已保存到: {save_dir}\n")

    # 创建README文件
    readme_file = os.path.join(save_dir, 'README.txt')
    with open(readme_file, 'w') as f:
        f.write(f"预处理数据集信息\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"属性: {property_name}\n\n")
        f.write(f"训练集: {len(train_data)} 样本\n")
        f.write(f"验证集: {len(val_data)} 样本\n")
        f.write(f"测试集: {len(test_data)} 样本\n\n")
        f.write(f"数据格式:\n")
        f.write(f"  每个样本包含:\n")
        f.write(f"    - id: 样本ID\n")
        f.write(f"    - graph: DGL atom graph (元组格式)\n")
        f.write(f"    - line_graph: DGL line graph\n")
        f.write(f"    - text: 归一化的晶体描述文本\n")
        f.write(f"    - target: 目标属性值\n\n")
        f.write(f"使用方法:\n")
        f.write(f"  import pickle\n")
        f.write(f"  with open('train.pkl', 'rb') as f:\n")
        f.write(f"      train_data = pickle.load(f)\n")

    print(f"✓ README已保存到: {readme_file}\n")


def main():
    parser = argparse.ArgumentParser(description='预处理数据集')

    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['jarvis', 'mp', 'class'],
                       help='数据集名称')
    parser.add_argument('--property', type=str, required=True,
                       help='属性名称 (例如: mbj_bandgap, formation_energy, band_gap)')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='数据集根目录 (包含CIF文件和CSV的目录)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='预处理数据输出目录')

    # 图构建参数
    parser.add_argument('--cutoff', type=float, default=8.0,
                       help='邻居搜索截断半径 (Å)')
    parser.add_argument('--max_neighbors', type=int, default=12,
                       help='最大邻居数')
    parser.add_argument('--use_canonize', type=bool, default=True,
                       help='是否使用规范化坐标')

    # 数据分割参数
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    args = parser.parse_args()

    # 验证比例
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"训练/验证/测试比例之和必须为1.0，当前为: {args.train_ratio + args.val_ratio + args.test_ratio}")

    print("\n" + "="*80)
    print("数据集预处理")
    print("="*80)
    print(f"\n配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  属性: {args.property}")
    print(f"  数据根目录: {args.root_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  Cutoff: {args.cutoff} Å")
    print(f"  Max neighbors: {args.max_neighbors}")
    print(f"  分割比例: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"  随机种子: {args.seed}\n")

    # 设置随机种子
    set_seed(args.seed)

    # 获取数据集路径
    cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)

    # 初始化文本归一化器
    print("初始化文本归一化器...")
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)
    mappings = load_vocab_mappings()
    print("✓ 归一化器初始化完成\n")

    # 步骤1: 加载原始数据
    raw_data = load_raw_data(cif_dir, id_prop_file, args.dataset, args.property, norm, mappings)

    if len(raw_data) == 0:
        print("❌ 错误: 没有成功加载任何数据")
        return

    # 步骤2: 构建图
    processed_data = build_graphs(
        raw_data,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        use_canonize=args.use_canonize
    )

    if len(processed_data) == 0:
        print("❌ 错误: 没有成功构建任何图")
        return

    # 步骤3: 分割数据集
    train_data, val_data, test_data = split_dataset(
        processed_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 步骤4: 保存预处理数据
    save_preprocessed_data(
        train_data, val_data, test_data,
        args.output_dir, args.dataset, args.property
    )

    print("="*80)
    print("预处理完成！")
    print("="*80)
    print(f"\n你现在可以使用预处理数据进行训练或评估:")
    print(f"\n  训练:")
    print(f"    python train_with_cross_modal_attention.py \\")
    print(f"        --dataset {args.dataset} \\")
    print(f"        --property {args.property} \\")
    print(f"        --use_preprocessed True \\")
    print(f"        --preprocessed_dir {args.output_dir}")
    print(f"\n  文本遮挡评估:")
    print(f"    python evaluate_text_masking.py \\")
    print(f"        --checkpoint <your_checkpoint.pt> \\")
    print(f"        --preprocessed_dir {args.output_dir} \\")
    print(f"        --dataset {args.dataset} \\")
    print(f"        --property {args.property}")
    print()


if __name__ == "__main__":
    main()
