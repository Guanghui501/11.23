#!/usr/bin/env python
"""
训练方案1: 中期融合 + 细粒度注意力 + 门控融合（无全局注意力）

这是针对你的架构的最优配置：
- 中期融合在第2层注入文本信息
- 细粒度注意力在原子-词元级别交互
- 门控融合在最后阶段动态学习融合权重
- 跳过全局交叉注意力（避免过度融合）

推荐使用这个方案作为baseline！
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='训练方案1: 中期+细粒度+门控融合'
    )

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='jarvis',
                        choices=['jarvis', 'mp'],
                        help='数据集')
    parser.add_argument('--property', type=str, default='formation_energy',
                        help='预测属性')
    parser.add_argument('--root_dir', type=str, default='../dataset/',
                        help='数据集根目录')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 模型参数
    parser.add_argument('--alignn_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)

    # 融合参数
    parser.add_argument('--middle_fusion_layers', type=str, default='2',
                        help='中期融合层（逗号分隔）')
    parser.add_argument('--fine_grained_num_heads', type=int, default=8)
    parser.add_argument('--gated_fusion_type', type=str, default='dual_gate',
                        choices=['single_gate', 'dual_gate', 'attention'])

    # 输出
    parser.add_argument('--output_dir', type=str, default='experiments/solution1')
    parser.add_argument('--save_every', type=int, default=50,
                        help='每N个epoch保存一次模型')

    return parser.parse_args()


def main():
    args = get_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==================== 配置方案1 ====================
    print("\n" + "=" * 80)
    print("方案1: 中期融合 + 细粒度注意力 + 门控融合")
    print("=" * 80)

    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        hidden_features=args.hidden_features,
        output_features=1,

        # 中期融合
        use_middle_fusion=True,
        middle_fusion_layers=args.middle_fusion_layers,
        middle_fusion_hidden_dim=128,
        middle_fusion_num_heads=2,
        middle_fusion_dropout=0.1,

        # 细粒度注意力
        use_fine_grained_attention=True,
        fine_grained_hidden_dim=256,
        fine_grained_num_heads=args.fine_grained_num_heads,
        fine_grained_dropout=0.1,
        fine_grained_use_projection=True,

        # 跳过全局交叉注意力
        use_cross_modal_attention=False,

        # 门控融合
        fusion_strategy="gated",
        gated_fusion_type=args.gated_fusion_type,
        gated_fusion_hidden_dim=128,
        gated_fusion_dropout=0.1,

        classification=False,
    )

    # 打印配置
    print(f"\n配置详情:")
    print(f"  use_middle_fusion: {model_config.use_middle_fusion}")
    print(f"  middle_fusion_layers: {model_config.middle_fusion_layers}")
    print(f"  use_fine_grained_attention: {model_config.use_fine_grained_attention}")
    print(f"  fine_grained_num_heads: {model_config.fine_grained_num_heads}")
    print(f"  use_cross_modal_attention: {model_config.use_cross_modal_attention}")
    print(f"  fusion_strategy: {model_config.fusion_strategy}")
    print(f"  gated_fusion_type: {model_config.gated_fusion_type}")
    print("\n架构流程:")
    print("  Input → ALIGNN layers → Middle Fusion (layer 2)")
    print("        → GCN layers → Fine-grained Attention")
    print("        → Pooling → Gated Fusion → Output")
    print("=" * 80 + "\n")

    # 保存模型配置
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config.dict(), f, indent=2)

    # ==================== 初始化模型 ====================
    model = ALIGNN(model_config)
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # ==================== 准备数据 ====================
    print("Loading data...")
    train_loader, val_loader, test_loader = get_train_val_loaders(
        dataset=args.dataset,
        target=args.property,
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=args.batch_size,
        workers=0,
        pin_memory=True,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # ==================== 训练配置 ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True
    )

    criterion = nn.L1Loss()  # MAE loss

    # ==================== 训练循环 ====================
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    # 用于记录门控值
    gate_history = {
        'epoch': [],
        'gate_graph': [],
        'gate_text': []
    }

    for epoch in range(args.epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_gate_graph = []
        train_gate_text = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)

            # 前向传播（返回gate值用于监控）
            output = model(batch, return_attention=True)

            if isinstance(output, dict):
                predictions = output['predictions']

                # 记录门控值
                if 'gate_values' in output:
                    gate_graph = output['gate_values']['gate_graph'].mean().item()
                    gate_text = output['gate_values']['gate_text'].mean().item()
                    train_gate_graph.append(gate_graph)
                    train_gate_text.append(gate_text)
            else:
                predictions = output

            # 计算损失
            loss = criterion(predictions, batch.y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)

                if isinstance(output, dict):
                    predictions = output['predictions']
                else:
                    predictions = output

                loss = criterion(predictions, batch.y)
                val_loss += loss.item()

                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        val_loss /= len(val_loader)

        # 计算额外指标
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        val_r2 = 1 - np.sum((val_predictions - val_targets) ** 2) / np.sum((val_targets - val_targets.mean()) ** 2)

        # 记录门控值
        if train_gate_graph:
            avg_gate_graph = np.mean(train_gate_graph)
            avg_gate_text = np.mean(train_gate_text)
            gate_history['epoch'].append(epoch + 1)
            gate_history['gate_graph'].append(avg_gate_graph)
            gate_history['gate_text'].append(avg_gate_text)

            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
            print(f"  Gate values: Graph={avg_gate_graph:.3f}, Text={avg_gate_text:.3f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # ========== 保存模型 ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config.dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # 定期保存checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config.dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        print()

    # ==================== 保存训练历史 ====================
    with open(os.path.join(args.output_dir, 'gate_history.json'), 'w') as f:
        json.dump(gate_history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
