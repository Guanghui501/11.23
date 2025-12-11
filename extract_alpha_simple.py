#!/usr/bin/env python
"""
简化版α值提取工具

如果模型没有内置返回α值的功能，这个脚本会自动patch模型
来捕获中期融合层的门控值。

用法:
    python extract_alpha_simple.py \
        --checkpoint best_model.pt \
        --output alpha_values.npz
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from models.alignn import ALIGNN


class AlphaCapture:
    """捕获中期融合的α值"""

    def __init__(self):
        self.alpha_values = []
        self.atom_features = []

    def reset(self):
        self.alpha_values = []
        self.atom_features = []


def patch_middle_fusion_module(module, alpha_capture):
    """
    为MiddleFusionModule添加hook来捕获α值

    假设中期融合模块的forward中计算了α：
    α = sigmoid(W_gate · [h; t])
    """

    original_forward = module.forward

    def new_forward(node_features, text_features, *args, **kwargs):
        """修改后的forward，捕获α值"""

        # 调用原始forward
        output = original_forward(node_features, text_features, *args, **kwargs)

        # 尝试捕获α值（假设模块内部有self.alpha属性）
        if hasattr(module, 'alpha') and module.alpha is not None:
            alpha_capture.alpha_values.append(module.alpha.detach().cpu())
        elif hasattr(module, 'gate_weight') and module.gate_weight is not None:
            alpha_capture.alpha_values.append(module.gate_weight.detach().cpu())

        return output

    module.forward = new_forward


def manually_compute_alpha(model, g, lg, text, device):
    """
    手动计算α值

    如果模型没有存储α，我们可以重新计算
    """

    model.eval()

    with torch.no_grad():
        # 1. 文本编码
        from transformers import AutoTokenizer, AutoModel
        from tokenizers.normalizers import BertNormalizer

        # 使用模型内部的tokenizer（假设已经加载）
        # 这里简化处理
        try:
            # 尝试获取文本特征
            output = model([g.to(device), lg.to(device), text],
                          return_features=True,
                          return_attention=True)

            if isinstance(output, dict):
                return output.get('gate_values', None)
        except:
            pass

    return None


def extract_alpha_from_checkpoint(checkpoint_path, data_loader, device,
                                  n_samples=100):
    """
    从checkpoint提取α值

    策略:
    1. 尝试直接从模型输出获取
    2. 如果失败，patch模块
    3. 如果还失败，手动计算
    """

    print(f"📂 加载模型: {checkpoint_path}")

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

    # 检查中期融合
    if not hasattr(model, 'middle_fusion_modules'):
        print("⚠️  模型没有middle_fusion_modules属性")
        print("   尝试查找其他融合模块...")

    # 准备α捕获器
    alpha_capture = AlphaCapture()

    # Patch中期融合模块
    if hasattr(model, 'middle_fusion_modules'):
        for name, module in model.middle_fusion_modules.items():
            print(f"✅ Patching {name}...")
            patch_middle_fusion_module(module, alpha_capture)

    # 提取α值
    all_alphas = []
    all_labels = []
    all_atom_types = []

    print(f"\n🔄 处理 {n_samples} 个样本...")

    count = 0
    for batch in data_loader:
        if count >= n_samples:
            break

        g, lg, text, labels = batch

        alpha_capture.reset()

        with torch.no_grad():
            try:
                # 尝试方法1: 直接从模型获取
                output = model([g.to(device), lg.to(device), text],
                              return_features=True,
                              return_attention=True)

                if isinstance(output, dict) and 'gate_values' in output:
                    alphas = output['gate_values']
                    print(f"  ✅ 从模型输出获取α值")
                elif len(alpha_capture.alpha_values) > 0:
                    alphas = torch.cat(alpha_capture.alpha_values, dim=0)
                    print(f"  ✅ 从hook获取α值")
                else:
                    print(f"  ⚠️  未能获取α值，跳过此batch")
                    continue

                # 保存
                batch_size = len(labels)
                atom_features = g.ndata['atom_features'].cpu().numpy()

                # 分割成单个样本
                atom_idx = 0
                for i in range(batch_size):
                    if count >= n_samples:
                        break

                    n_atoms = (g.batch_num_nodes() == i).sum().item()

                    sample_alpha = alphas[atom_idx:atom_idx+n_atoms].cpu().numpy()
                    sample_atoms = atom_features[atom_idx:atom_idx+n_atoms]

                    all_alphas.append(sample_alpha)
                    all_labels.append(labels[i].item())
                    all_atom_types.append(sample_atoms)

                    atom_idx += n_atoms
                    count += 1

                    if (count % 10) == 0:
                        print(f"  处理进度: {count}/{n_samples}")

            except Exception as e:
                print(f"  ❌ 错误: {e}")
                continue

    print(f"\n✅ 成功提取 {len(all_alphas)} 个样本的α值")

    return {
        'alphas': all_alphas,
        'labels': all_labels,
        'atom_types': all_atom_types
    }


def quick_visualize(alpha_data, save_path):
    """快速可视化α值"""

    import matplotlib.pyplot as plt

    # 合并所有α
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
    print(f"✅ 快速可视化已保存: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='简化α值提取')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dft_3d',
                       help='JARVIS数据集名称 (dft_3d, dft_2d等)')
    parser.add_argument('--target', type=str, default='mbj_bandgap',
                       help='目标性质名称')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output', type=str, default='alpha_values.npz')
    parser.add_argument('--visualize', action='store_true',
                       help='生成快速可视化')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🔍 简化版α值提取工具")
    print("="*70)

    # 加载数据
    print(f"\n📊 加载数据集: {args.dataset}")
    print(f"   目标性质: {args.target}")

    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset=args.dataset,
        target=args.target,
        batch_size=16,
        workers=0,
        pin_memory=False
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")

    # 提取α值
    alpha_data = extract_alpha_from_checkpoint(
        args.checkpoint,
        test_loader,
        device,
        n_samples=args.n_samples
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
