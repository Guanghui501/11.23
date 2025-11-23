#!/usr/bin/env python
"""
使用 Robust Attention Analyzer 的演示脚本
=========================================

这个脚本使用新的健壮注意力分析系统，能够：
1. 自动诊断注意力质量
2. 根据质量自动选择分析策略
3. 即使原子注意力相同也能提供有用分析
4. 生成详细的可视化和统计报告

使用方法:
    python demo_robust_attention.py \
        --model_path /path/to/checkpoint.pt \
        --cif_path /path/to/structure.cif \
        --text "Material description..." \
        --save_dir ./results
"""

import argparse
import torch
from pathlib import Path

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.core.specie import chem_data, get_node_attributes
import numpy as np

from models.alignn import ALIGNN, ALIGNNConfig
from robust_attention_analyzer import run_complete_analysis


def load_model(checkpoint_path, device='cuda'):
    """加载模型和配置"""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
        print("✅ 从checkpoint加载配置")
    else:
        print("⚠️  未找到config，使用默认配置")
        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            hidden_features=256,
            output_features=1,
            use_cross_modal_attention=True,
            cross_modal_hidden_dim=256,
            cross_modal_num_heads=4,
            use_middle_fusion=True,
            use_fine_grained_attention=True,
        )

    print(f"\n📋 模型配置:")
    print(f"   - use_cross_modal_attention: {config.use_cross_modal_attention}")
    print(f"   - use_middle_fusion: {config.use_middle_fusion}")
    print(f"   - use_fine_grained_attention: {config.use_fine_grained_attention}")

    model = ALIGNN(config)

    checkpoint_state = checkpoint.get('model', checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)

    if missing_keys:
        print(f"\n⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"     - {key}")

    if unexpected_keys:
        print(f"\n⚠️  Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"     - {key}")

    model = model.to(device)
    model.eval()

    print(f"\n✅ 模型已加载并设置为 eval 模式")
    print(f"   - Training mode: {model.training}")

    return model, config


def cif_to_graph(cif_path, cutoff=8.0, max_neighbors=12):
    """CIF转图"""

    atoms = Atoms.from_cif(cif_path)

    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="atomic_number",
        compute_line_graph=True,
        use_canonize=True
    )

    # 构建特征查找表
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features="cgcnn")
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features="cgcnn")
        if x is not None:
            features[z, :] = x

    # 转换特征
    z = g.ndata.pop("atom_features")
    g.ndata["atomic_number"] = z
    z = z.type(torch.LongTensor).squeeze()
    f = torch.tensor(features[z], dtype=torch.float32)
    g.ndata["atom_features"] = f

    return g, lg, atoms


def main():
    parser = argparse.ArgumentParser(description='Robust Fine-Grained Attention Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--cif_path', type=str, required=True,
                       help='CIF文件路径')
    parser.add_argument('--text', type=str, required=True,
                       help='材料描述文本')
    parser.add_argument('--save_dir', type=str, default='./robust_analysis',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔬 Robust Fine-Grained Attention Analysis")
    print("="*80)

    # 加载模型
    print("\n📦 加载模型...")
    model, config = load_model(args.model_path, device=args.device)

    # 加载结构
    print(f"\n📂 加载结构: {args.cif_path}")
    g, lg, atoms_object = cif_to_graph(args.cif_path)
    print(f"   - 原子数: {atoms_object.num_atoms}")
    print(f"   - 化学式: {atoms_object.composition.reduced_formula}")
    print(f"   - 元素: {', '.join([str(atoms_object.elements[i]) for i in range(atoms_object.num_atoms)])}")

    # 运行分析
    print(f"\n🔍 分析文本:")
    text_preview = args.text[:100] + "..." if len(args.text) > 100 else args.text
    print(f'   "{text_preview}"')

    results = run_complete_analysis(
        model=model,
        g=g,
        lg=lg,
        text=args.text,
        atoms_object=atoms_object,
        save_dir=args.save_dir
    )

    if results is None:
        print("\n❌ 分析失败")
        return

    # 打印摘要
    print("\n" + "="*80)
    print("📊 分析摘要")
    print("="*80)

    diagnosis = results.get('diagnosis', {})
    print(f"\n质量评估: {diagnosis.get('quality', 'unknown').upper()}")
    print(f"原子多样性分数: {diagnosis.get('atom_diversity', 0):.4f}")
    print(f"Head多样性分数: {diagnosis.get('head_diversity', 0):.4f}")
    print(f"平均熵: {diagnosis.get('entropy', 0):.4f}")

    if diagnosis.get('issues'):
        print(f"\n⚠️  发现的问题:")
        for issue in diagnosis['issues']:
            print(f"   - {issue}")

    if diagnosis.get('recommendations'):
        print(f"\n💡 建议:")
        for rec in diagnosis['recommendations']:
            print(f"   - {rec}")

    # 统计信息
    stats = results.get('statistics', {})
    if stats:
        print(f"\n📈 统计信息:")
        print(f"   - 注意力头数: {stats.get('num_heads', 'N/A')}")
        print(f"   - 原子数: {stats.get('num_atoms', 'N/A')}")
        print(f"   - 序列长度: {stats.get('seq_len', 'N/A')}")
        print(f"   - 平均注意力: {stats.get('mean_attention', 0):.6f}")
        print(f"   - 注意力标准差: {stats.get('std_attention', 0):.6f}")
        print(f"   - 稀疏度: {stats.get('sparsity', 0)*100:.2f}%")

    print(f"\n✅ 分析完成！结果已保存到: {args.save_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
