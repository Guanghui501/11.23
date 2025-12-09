#!/usr/bin/env python
"""
Checkpoint检查工具

用于检查checkpoint文件的内容，帮助诊断加载问题。

用法:
    python inspect_checkpoint.py /path/to/checkpoint.pt
"""

import sys
import torch
import argparse


def inspect_checkpoint(checkpoint_path):
    """检查checkpoint文件的内容"""
    print("="*80)
    print(f"检查 Checkpoint: {checkpoint_path}")
    print("="*80)
    print()

    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"✓ Checkpoint加载成功\n")

        # 检查checkpoint类型
        if isinstance(checkpoint, dict):
            print(f"Checkpoint类型: dict")
            print(f"包含的键 ({len(checkpoint)} 个):")
            print("-"*80)

            for key in checkpoint.keys():
                value = checkpoint[key]
                print(f"  • {key:<30} : {type(value).__name__}")

                # 显示一些详细信息
                if key == 'model' or key == 'model_state_dict' or key == 'state_dict':
                    if isinstance(value, dict):
                        print(f"      → 包含 {len(value)} 个参数")
                        # 显示前几个键
                        sample_keys = list(value.keys())[:3]
                        for sample_key in sample_keys:
                            print(f"         - {sample_key}")
                        if len(value) > 3:
                            print(f"         ... (还有 {len(value) - 3} 个)")

                elif key in ['model_config', 'config', 'args']:
                    print(f"      → {value}")

                elif key in ['epoch', 'best_val_loss', 'best_test_loss', 'val_mae', 'test_mae']:
                    print(f"      → {value}")

            print()

            # 检查关键字段
            print("="*80)
            print("关键字段检查:")
            print("-"*80)

            # 检查模型权重
            model_keys = ['model', 'model_state_dict', 'state_dict']
            found_model = False
            for key in model_keys:
                if key in checkpoint:
                    print(f"✓ 找到模型权重: '{key}'")
                    found_model = True
                    break
            if not found_model:
                print(f"✗ 未找到模型权重 (尝试过: {model_keys})")

            # 检查模型配置
            config_keys = ['model_config', 'config', 'args', 'model_args']
            found_config = False
            for key in config_keys:
                if key in checkpoint:
                    print(f"✓ 找到模型配置: '{key}'")
                    found_config = True

                    # 显示配置详情
                    config = checkpoint[key]
                    print(f"\n配置内容:")
                    if hasattr(config, '__dict__'):
                        for attr, val in config.__dict__.items():
                            print(f"    {attr}: {val}")
                    elif isinstance(config, dict):
                        for k, v in config.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"    {config}")
                    break

            if not found_config:
                print(f"✗ 未找到模型配置 (尝试过: {config_keys})")
                print(f"\n⚠ 警告: 缺少模型配置！")
                print(f"   这可能导致评估脚本失败。")
                print(f"\n建议解决方案:")
                print(f"   1. 使用 create_config_from_checkpoint.py 创建配置文件")
                print(f"   2. 或使用 fix_checkpoint.py 修复checkpoint")

            # 检查训练信息
            print(f"\n训练信息:")
            if 'epoch' in checkpoint:
                print(f"  训练轮数: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
            if 'best_test_loss' in checkpoint:
                print(f"  最佳测试损失: {checkpoint['best_test_loss']:.4f}")
            if 'val_mae' in checkpoint:
                print(f"  验证集 MAE: {checkpoint['val_mae']:.4f}")
            if 'test_mae' in checkpoint:
                print(f"  测试集 MAE: {checkpoint['test_mae']:.4f}")

        else:
            print(f"Checkpoint类型: {type(checkpoint).__name__}")
            print(f"⚠ 警告: Checkpoint不是字典类型，可能是直接保存的模型")

        print()
        print("="*80)

    except Exception as e:
        print(f"✗ 加载checkpoint失败")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='检查checkpoint文件内容')
    parser.add_argument('checkpoint', type=str, help='Checkpoint文件路径')

    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
