#!/usr/bin/env python
"""
从本地数据提取α值（支持本地CSV+CIF文件）

用法:
    python extract_alpha_local.py \
        --checkpoint best_model.pt \
        --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
        --dataset jarvis \
        --property hse_bandgap-2 \
        --n_samples 500 \
        --output alpha_values.npz
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from jarvis.core.atoms import Atoms
from data import get_train_val_loaders
from models.alignn import ALIGNN
from tokenizers.normalizers import BertNormalizer


def get_dataset_paths(root_dir, dataset, property_name):
    """获取数据集路径（与训练脚本相同的逻辑）"""

    # Property name映射（处理别名）
    property_map = {
        'hse_bandgap': 'hse_gap',
        'hse_bandgap-2': 'hse_gap',
        'bandgap_mbj': 'mbj_bandgap',
        'formation_energy': 'formation_energy_peratom',
    }

    if dataset.lower() == 'jarvis':
        prop_folder = property_map.get(property_name, property_name)
        cif_dir = os.path.join(root_dir, f'jarvis/{prop_folder}/cif/')
        id_prop_file = os.path.join(root_dir, f'jarvis/{prop_folder}/description.csv')

    elif dataset.lower() == 'mp':
        if property_name in ['formation_energy', 'band_gap']:
            cif_dir = os.path.join(root_dir, 'mp_2018_new/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_new/mat_text.csv')
        elif property_name in ['bulk', 'shear', 'bulk_modulus', 'shear_modulus']:
            cif_dir = os.path.join(root_dir, 'mp_2018_small/cif/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_small/description.csv')
        else:
            raise ValueError(f"Unsupported property for MP: {property_name}")

    elif dataset.lower() == 'class':
        cif_dir = os.path.join(root_dir, f'class/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'class/{property_name}/description.csv')

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return cif_dir, id_prop_file


def load_dataset(cif_dir, id_prop_file, dataset, property_name, max_samples=None):
    """加载数据集（从本地CSV和CIF文件）"""

    print(f"\n{'='*70}")
    print(f"📂 加载本地数据集")
    print(f"   数据集: {dataset}")
    print(f"   性质: {property_name}")
    print(f"   CIF目录: {cif_dir}")
    print(f"   描述文件: {id_prop_file}")
    print(f"{'='*70}\n")

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

    print(f"CSV文件总样本数: {len(data)}")

    # 文本归一化
    norm = BertNormalizer(lowercase=True)

    # 加载vocab映射
    vocab_file = os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt')
    if not os.path.exists(vocab_file):
        print("⚠️  未找到vocab_mappings.txt，使用简单归一化")
        def normalize(text):
            return norm.normalize_str(text)
    else:
        with open(vocab_file, 'r') as f:
            mappings = f.read().strip().split('\n')
        mappings = {m[0]: m[2:] for m in mappings}

        def normalize(text):
            text = [norm.normalize_str(s) for s in text.split('\n')]
            out = []
            for s in text:
                norm_s = ''
                for c in s:
                    norm_s += mappings.get(c, ' ')
                out.append(norm_s)
            return '\n'.join(out)

    # 加载样本
    dataset_array = []
    skipped = 0
    max_load = max_samples * 2 if max_samples else len(data)  # 多加载一些以防跳过

    for j in tqdm(range(min(len(data), max_load)), desc="加载数据"):
        try:
            if dataset.lower() == 'jarvis':
                id, composition, target, crys_desc_full, _ = data[j]
            elif dataset.lower() == 'mp':
                if property_name in ['formation_energy', 'band_gap']:
                    id, composition, target, _, crys_desc_full, _ = data[j]
                else:
                    id, composition, target, _, crys_desc_full, _ = data[j]
            elif dataset.lower() == 'class':
                id, target, crys_desc_full = data[j]

            # 读取CIF文件
            file_path = os.path.join(cif_dir, f'{id}.cif')
            if not os.path.exists(file_path):
                skipped += 1
                continue

            atoms = Atoms.from_cif(file_path)

            # 构建样本
            info = {
                "atoms": atoms.to_dict(),
                "jid": id,
                "text": crys_desc_full,
                "target": float(target)
            }

            dataset_array.append(info)

            # 达到所需样本数就停止
            if max_samples and len(dataset_array) >= max_samples:
                break

        except Exception as e:
            skipped += 1
            if j < 10:  # 只打印前10个错误
                print(f"  跳过样本 {j}: {e}")
            continue

    print(f"\n✅ 成功加载 {len(dataset_array)} 个样本")
    print(f"   跳过: {skipped} 个样本")

    return dataset_array


class AlphaExtractor:
    """α值提取器"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_from_batch(self, batch):
        """从batch提取α值"""
        g, lg, text, labels = batch

        with torch.no_grad():
            try:
                output = self.model(
                    [g.to(self.device), lg.to(self.device), text],
                    return_features=True,
                    return_attention=True
                )

                # 尝试获取α值
                if isinstance(output, dict) and 'gate_values' in output:
                    alphas = output['gate_values'].cpu()
                else:
                    # 如果没有直接返回，尝试从模块获取
                    alphas = None
                    if hasattr(self.model, 'middle_fusion_modules'):
                        for name, module in self.model.middle_fusion_modules.items():
                            if hasattr(module, 'alpha'):
                                alphas = module.alpha.cpu()
                                break

                if alphas is None:
                    return None, None, None

                # 提取原子信息
                atom_types = g.ndata['atom_features'].cpu().numpy()

                return alphas, atom_types, labels.cpu()

            except Exception as e:
                print(f"⚠️  提取失败: {e}")
                return None, None, None


def extract_alpha_values(checkpoint_path, dataset_array, device, n_samples=500):
    """提取α值"""

    print(f"\n📂 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        model_state = checkpoint['model']
        config = checkpoint.get('config', None)
    else:
        model_state = checkpoint
        config = None

    model = ALIGNN(config) if config else ALIGNN()
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print("✅ 模型已加载")

    # 检查中期融合
    if not hasattr(model, 'middle_fusion_modules'):
        print("\n⚠️  警告: 模型可能没有中期融合模块")
        print("   将尝试提取α值，但可能失败")
    else:
        print(f"✅ 检测到中期融合模块: {list(model.middle_fusion_modules.keys())}")

    # 创建数据加载器
    print(f"\n🔄 创建数据加载器...")

    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset_array=dataset_array,
        target='target',  # 使用'target'作为键
        batch_size=16,
        workers=0,
        pin_memory=False,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    print(f"✅ 测试集: {len(test_loader.dataset)} 样本")

    # 提取α值
    extractor = AlphaExtractor(model, device)

    all_alphas = []
    all_labels = []
    all_atom_types = []

    print(f"\n🔍 提取α值（目标: {n_samples} 个样本）...")

    count = 0
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        if count >= n_samples:
            break

        alphas, atom_types, labels = extractor.extract_from_batch(batch)

        if alphas is None:
            continue

        # 按样本分割
        batch_size = len(labels)
        n_atoms_list = batch[0].batch_num_nodes().cpu().numpy()

        atom_idx = 0
        for i in range(batch_size):
            if count >= n_samples:
                break

            n_atoms = n_atoms_list[i]

            sample_alpha = alphas[atom_idx:atom_idx+n_atoms].numpy()
            sample_atoms = atom_types[atom_idx:atom_idx+n_atoms]

            all_alphas.append(sample_alpha)
            all_labels.append(labels[i].item())
            all_atom_types.append(sample_atoms)

            atom_idx += n_atoms
            count += 1

    print(f"\n✅ 成功提取 {len(all_alphas)} 个样本的α值")

    return {
        'alphas': all_alphas,
        'labels': all_labels,
        'atom_types': all_atom_types
    }


def quick_visualize(alpha_data, save_path):
    """快速可视化"""
    import matplotlib.pyplot as plt

    all_alphas = np.concatenate(alpha_data['alphas'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 直方图
    ax = axes[0]
    ax.hist(all_alphas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(all_alphas.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {all_alphas.mean():.3f}')
    ax.set_xlabel('Gate Value (α)')
    ax.set_ylabel('Frequency')
    ax.set_title('Gate Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 箱型图
    ax = axes[1]
    avg_alphas = [np.mean(a) for a in alpha_data['alphas']]
    ax.boxplot([all_alphas, avg_alphas], labels=['Per-Atom α', 'Per-Material Avg'])
    ax.set_ylabel('Gate Value (α)')
    ax.set_title('Overall Statistics')
    ax.grid(True, alpha=0.3)

    # α vs 标签
    ax = axes[2]
    labels = alpha_data['labels']
    ax.scatter(labels, avg_alphas, alpha=0.5, s=30, c=avg_alphas,
              cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Mean Gate Value (α)')
    ax.set_title('α vs Target Property')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ 可视化已保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='从本地数据提取α值')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--dataset', type=str, default='jarvis',
                       help='数据集名称 (jarvis, mp, class)')
    parser.add_argument('--property', type=str, required=True,
                       help='性质名称 (hse_bandgap-2, mbj_bandgap等)')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='提取的样本数')
    parser.add_argument('--output', type=str, default='alpha_values.npz',
                       help='输出文件路径')
    parser.add_argument('--visualize', action='store_true',
                       help='生成快速可视化')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔍 α值提取工具（本地数据版）")
    print("="*70)

    # 获取路径
    cif_dir, id_prop_file = get_dataset_paths(
        args.root_dir, args.dataset, args.property
    )

    # 加载数据集
    dataset_array = load_dataset(
        cif_dir, id_prop_file, args.dataset, args.property,
        max_samples=args.n_samples * 2  # 多加载一些
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")

    # 提取α值
    alpha_data = extract_alpha_values(
        args.checkpoint, dataset_array, device, args.n_samples
    )

    # 保存
    output_path = Path(args.output)
    np.savez(output_path,
             alphas=alpha_data['alphas'],
             labels=alpha_data['labels'],
             atom_types=alpha_data['atom_types'])

    print(f"\n💾 α值已保存到: {output_path}")

    # 统计
    all_alphas = np.concatenate(alpha_data['alphas'])
    print(f"\n📊 统计信息:")
    print(f"  样本数: {len(alpha_data['alphas'])}")
    print(f"  原子总数: {len(all_alphas)}")
    print(f"  α均值: {all_alphas.mean():.4f}")
    print(f"  α标准差: {all_alphas.std():.4f}")
    print(f"  α范围: [{all_alphas.min():.4f}, {all_alphas.max():.4f}]")

    # 可视化
    if args.visualize:
        print(f"\n🎨 生成快速可视化...")
        quick_visualize(alpha_data, output_path.with_suffix('.png'))

    print("\n" + "="*70)
    print("✅ 完成！")
    print("="*70)


if __name__ == '__main__':
    main()
