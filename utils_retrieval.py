#!/usr/bin/env python
"""
Retrieval 评估工具函数
提供通用的模型加载、检查点处理等功能
"""

import torch
from pathlib import Path


def load_model_checkpoint(model, checkpoint_path, device='cuda', verbose=True):
    """
    智能加载模型检查点，自动处理不同的保存格式

    支持的格式:
    1. 完整字典: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}
    2. 仅模型权重: model.state_dict()
    3. 其他格式: {'model': ..., 'state_dict': ...}

    Args:
        model: PyTorch 模型实例
        checkpoint_path: 检查点文件路径
        device: 加载到哪个设备
        verbose: 是否打印加载信息

    Returns:
        model: 加载权重后的模型
        checkpoint_info: 检查点的其他信息 (dict)
    """
    if verbose:
        print(f"📥 加载检查点: {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取模型权重
    if isinstance(checkpoint, dict):
        # 尝试常见的键名
        possible_keys = [
            'model_state_dict',
            'model',
            'state_dict',
            'net',
            'model_dict'
        ]

        state_dict = None
        used_key = None

        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                used_key = key
                break

        if state_dict is None:
            # 可能整个 checkpoint 就是 state_dict
            # 检查是否包含模型参数的键（通常以模块名开头）
            if any(k.startswith(('atom_embedding', 'edge_embedding', 'alignn_layers',
                                'gcn_layers', 'fc', 'readout')) for k in checkpoint.keys()):
                state_dict = checkpoint
                used_key = 'direct'
                if verbose:
                    print("  ℹ️  检查点直接包含模型参数")
            else:
                raise KeyError(
                    f"无法从检查点中找到模型权重。\n"
                    f"尝试的键: {possible_keys}\n"
                    f"检查点包含的键: {list(checkpoint.keys())}"
                )

        if verbose and used_key != 'direct':
            print(f"  ✅ 从键 '{used_key}' 加载模型权重")

    else:
        # checkpoint 本身就是 state_dict
        state_dict = checkpoint
        if verbose:
            print("  ✅ 检查点是模型权重字典")

    # 加载到模型
    try:
        model.load_state_dict(state_dict, strict=True)
        if verbose:
            print("  ✅ 模型权重加载成功（strict mode）")
    except RuntimeError as e:
        # 尝试非严格模式
        if verbose:
            print(f"  ⚠️  严格模式加载失败，尝试非严格模式...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if verbose:
            if missing_keys:
                print(f"  ⚠️  缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  ⚠️  意外的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            print("  ✅ 模型权重加载成功（非严格模式）")

    # 移动到设备
    model = model.to(device)

    # 提取其他信息
    checkpoint_info = {}
    if isinstance(checkpoint, dict):
        checkpoint_info = {
            k: v for k, v in checkpoint.items()
            if k not in ['model_state_dict', 'model', 'state_dict', 'optimizer_state_dict']
        }

    if verbose and checkpoint_info:
        print(f"  ℹ️  检查点额外信息: {list(checkpoint_info.keys())}")

    return model, checkpoint_info


def print_checkpoint_info(checkpoint_path):
    """
    打印检查点文件的详细信息（用于调试）

    Args:
        checkpoint_path: 检查点文件路径
    """
    print(f"🔍 检查点信息: {checkpoint_path}")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        print("📦 检查点是字典，包含以下键:")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} items")
                if len(value) < 10:
                    for k in list(value.keys())[:5]:
                        print(f"      - {k}")
            elif isinstance(value, torch.Tensor):
                print(f"  - {key}: Tensor {value.shape}")
            elif isinstance(value, (int, float, str)):
                print(f"  - {key}: {type(value).__name__} = {value}")
            else:
                print(f"  - {key}: {type(value).__name__}")

        # 检查是否包含常见的模型权重键
        print("\n🔎 检测到的可能的模型权重键:")
        possible_keys = ['model_state_dict', 'model', 'state_dict', 'net']
        for key in possible_keys:
            if key in checkpoint:
                print(f"  ✅ '{key}' 存在")
            else:
                print(f"  ❌ '{key}' 不存在")

    else:
        print("📦 检查点直接是 state_dict")
        print(f"   包含 {len(checkpoint)} 个参数")
        print(f"   示例键: {list(checkpoint.keys())[:5]}")

    print("=" * 80)


def get_model_config_from_checkpoint(checkpoint_path):
    """
    从检查点中提取模型配置（如果有保存）

    Args:
        checkpoint_path: 检查点文件路径

    Returns:
        config: 模型配置字典，如果没有则返回 None
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        # 尝试常见的配置键名
        for key in ['config', 'model_config', 'hparams', 'args']:
            if key in checkpoint:
                return checkpoint[key]

    return None


def safe_model_load(model_class, config, checkpoint_path, device='cuda', verbose=True):
    """
    安全地初始化模型并加载权重

    Args:
        model_class: 模型类（如 ALIGNN）
        config: 模型配置
        checkpoint_path: 检查点路径
        device: 设备
        verbose: 是否打印信息

    Returns:
        model: 加载完成的模型
    """
    # 初始化模型
    if verbose:
        print("🔧 初始化模型...")
    model = model_class(config)

    # 加载权重
    model, checkpoint_info = load_model_checkpoint(
        model, checkpoint_path, device=device, verbose=verbose
    )

    # 设置为评估模式
    model.eval()

    if verbose:
        print("✅ 模型加载完成\n")

    return model


if __name__ == '__main__':
    """测试脚本：检查检查点文件格式"""
    import sys

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        print_checkpoint_info(checkpoint_path)
    else:
        print("用法: python utils_retrieval.py <checkpoint_path>")
        print("示例: python utils_retrieval.py checkpoints/best_model.pt")
