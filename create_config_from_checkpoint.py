#!/usr/bin/env python
"""
从checkpoint推断模型配置

当checkpoint缺少model_config时，通过分析模型结构来推断配置。

用法:
    python create_config_from_checkpoint.py \
        --checkpoint /path/to/checkpoint.pt \
        --output config.json
"""

import sys
import os
import json
import torch
import argparse

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from models.alignn import ALIGNNConfig


def infer_config_from_state_dict(state_dict):
    """从state_dict推断模型配置"""
    config_params = {}

    # 检查模型层数
    # 查找 alignn.atoms_embedding.{i}. 的最大i值
    max_alignn_layer = -1
    max_gcn_layer = -1

    for key in state_dict.keys():
        if 'atoms_embedding.' in key:
            parts = key.split('.')
            try:
                layer_idx = int(parts[parts.index('atoms_embedding') + 1])
                max_alignn_layer = max(max_alignn_layer, layer_idx)
            except (ValueError, IndexError):
                pass

        if 'edge_embedding.' in key:
            parts = key.split('.')
            try:
                layer_idx = int(parts[parts.index('edge_embedding') + 1])
                max_gcn_layer = max(max_gcn_layer, layer_idx)
            except (ValueError, IndexError):
                pass

    config_params['alignn_layers'] = max_alignn_layer + 1 if max_alignn_layer >= 0 else 4
    config_params['gcn_layers'] = max_gcn_layer + 1 if max_gcn_layer >= 0 else 4

    # 检查hidden_features
    # 查找第一个线性层的维度
    for key in state_dict.keys():
        if 'atom_embedding.weight' in key:
            hidden_dim = state_dict[key].shape[0]
            config_params['embedding_features'] = hidden_dim
            config_params['hidden_features'] = hidden_dim
            break

    # 检查跨模态注意力
    config_params['use_cross_modal_attention'] = any('cross_modal_attention' in key for key in state_dict.keys())

    if config_params['use_cross_modal_attention']:
        # 推断cross_modal参数
        for key in state_dict.keys():
            if 'cross_modal_attention.graph_to_text_attn.0.in_proj_weight' in key:
                dim = state_dict[key].shape[1]
                config_params['cross_modal_hidden_dim'] = dim

            if 'cross_modal_attention.graph_to_text_attn.0.in_proj_weight' in key:
                # MultiheadAttention的权重是 (3*embed_dim, embed_dim)
                total_dim = state_dict[key].shape[0]
                embed_dim = state_dict[key].shape[1]
                num_heads = 4  # 默认值，很难从权重推断
                config_params['cross_modal_num_heads'] = num_heads

    # 检查中期融合
    config_params['use_middle_fusion'] = any('middle_fusion' in key for key in state_dict.keys())

    if config_params['use_middle_fusion']:
        # 推断middle_fusion参数
        middle_fusion_layers = []
        for key in state_dict.keys():
            if 'middle_fusion_modules' in key:
                parts = key.split('.')
                try:
                    layer_idx = int(parts[parts.index('middle_fusion_modules') + 1])
                    middle_fusion_layers.append(layer_idx)
                except (ValueError, IndexError):
                    pass

        if middle_fusion_layers:
            # 假设融合层是连续的，从最小到最大
            config_params['middle_fusion_layers'] = list(range(min(middle_fusion_layers), max(middle_fusion_layers) + 1))

    # 检查细粒度注意力
    config_params['use_fine_grained_attention'] = any('fine_grained_attention' in key for key in state_dict.keys())

    # 检查门控融合
    has_gated_fusion = any('gated_fusion' in key for key in state_dict.keys())
    if has_gated_fusion:
        config_params['fusion_strategy'] = 'gated'
        # 推断gated_fusion_type
        if any('graph_gate' in key and 'text_gate' in key for key in state_dict.keys()):
            config_params['gated_fusion_type'] = 'dual_gate'
        else:
            config_params['gated_fusion_type'] = 'single_gate'

    return config_params


def create_config(checkpoint_path, output_path=None):
    """创建模型配置"""
    print("="*80)
    print("从Checkpoint推断模型配置")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}\n")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检查是否已有配置
    if 'model_config' in checkpoint:
        print("✓ Checkpoint已包含model_config")
        config = checkpoint['model_config']

        if isinstance(config, dict):
            config_dict = config
        elif hasattr(config, 'dict'):
            config_dict = config.dict()
        elif hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        elif hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = {}

    elif 'config' in checkpoint:
        print("✓ Checkpoint包含config (非标准字段名)")
        config = checkpoint['config']
        if isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)

    else:
        print("⚠ Checkpoint缺少model_config，尝试从模型结构推断...\n")

        # 获取state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError("无法在checkpoint中找到模型权重")

        # 推断配置
        inferred_params = infer_config_from_state_dict(state_dict)

        print("推断的参数:")
        for key, value in inferred_params.items():
            print(f"  {key}: {value}")

        # 创建默认配置并更新推断的参数
        config = ALIGNNConfig(**inferred_params)

        if hasattr(config, 'dict'):
            config_dict = config.dict()
        elif hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        else:
            config_dict = config.__dict__

        print("\n⚠ 注意: 某些参数可能不准确，请检查并根据实际情况调整")

    # 显示配置
    print("\n" + "="*80)
    print("模型配置:")
    print("="*80)
    for key, value in sorted(config_dict.items()):
        print(f"  {key:<35} : {value}")

    # 保存配置
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"\n✓ 配置已保存到: {output_path}")
    else:
        # 默认保存到checkpoint同目录
        checkpoint_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(checkpoint_dir, 'inferred_model_config.json')
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"\n✓ 配置已保存到: {output_path}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print(f"\n1. 检查并编辑配置文件（如有需要）:")
    print(f"   vim {output_path}")
    print(f"\n2. 使用配置文件运行评估:")
    print(f"   python evaluate_text_masking.py \\")
    print(f"       --checkpoint {checkpoint_path} \\")
    print(f"       --config_file {output_path} \\")
    print(f"       --preprocessed_dir <preprocessed_dir> \\")
    print(f"       --dataset jarvis \\")
    print(f"       --property mbj_bandgap")
    print()

    return config_dict


def main():
    parser = argparse.ArgumentParser(description='从checkpoint创建模型配置')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint文件路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出配置文件路径 (默认: checkpoint同目录下的inferred_model_config.json)')

    args = parser.parse_args()

    create_config(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
