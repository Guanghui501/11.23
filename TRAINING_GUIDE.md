# 方案1和方案2训练指南

完整的训练教程：如何训练中期融合+细粒度注意力架构的优化方案

---

## 📋 快速开始

### 方案1训练（推荐）⭐

```bash
python train_with_cross_modal_attention.py \
  --dataset jarvis \
  --property formation_energy \
  --use_middle_fusion True \
  --middle_fusion_layers "2" \
  --use_fine_grained_attention True \
  --use_cross_modal False \
  --batch_size 64 \
  --epochs 300 \
  --output_dir experiments/solution1_gated
```

### 方案2训练

```bash
python train_with_cross_modal_attention.py \
  --dataset jarvis \
  --property formation_energy \
  --use_middle_fusion True \
  --middle_fusion_layers "2" \
  --use_fine_grained_attention True \
  --use_cross_modal True \
  --batch_size 64 \
  --epochs 300 \
  --output_dir experiments/solution2_unidirectional
```

**注意：** 由于当前训练脚本缺少新参数，你需要先更新训练脚本（见下文）。

---

## 🔧 方法1: 通过配置文件训练（推荐）

### 步骤1: 创建配置文件

#### 方案1配置 (`config_solution1.json`)

```json
{
  "name": "alignn",
  "alignn_layers": 4,
  "gcn_layers": 4,
  "hidden_features": 256,
  "atom_input_features": 92,
  "edge_input_features": 80,
  "triplet_input_features": 40,
  "embedding_features": 64,
  "output_features": 1,
  "graph_dropout": 0.0,

  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_hidden_dim": 128,
  "middle_fusion_num_heads": 2,
  "middle_fusion_dropout": 0.1,

  "use_fine_grained_attention": true,
  "fine_grained_hidden_dim": 256,
  "fine_grained_num_heads": 8,
  "fine_grained_dropout": 0.1,
  "fine_grained_use_projection": true,

  "use_cross_modal_attention": false,

  "fusion_strategy": "gated",
  "gated_fusion_type": "dual_gate",
  "gated_fusion_hidden_dim": 128,
  "gated_fusion_dropout": 0.1,

  "use_contrastive_loss": false,
  "link": "identity",
  "classification": false
}
```

#### 方案2配置 (`config_solution2.json`)

```json
{
  "name": "alignn",
  "alignn_layers": 4,
  "gcn_layers": 4,
  "hidden_features": 256,
  "atom_input_features": 92,
  "edge_input_features": 80,
  "triplet_input_features": 40,
  "embedding_features": 64,
  "output_features": 1,
  "graph_dropout": 0.0,

  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "middle_fusion_hidden_dim": 128,
  "middle_fusion_num_heads": 2,
  "middle_fusion_dropout": 0.1,

  "use_fine_grained_attention": true,
  "fine_grained_hidden_dim": 256,
  "fine_grained_num_heads": 8,
  "fine_grained_dropout": 0.1,
  "fine_grained_use_projection": true,

  "use_cross_modal_attention": true,
  "cross_modal_attention_type": "unidirectional",
  "cross_modal_hidden_dim": 256,
  "cross_modal_num_heads": 4,
  "cross_modal_dropout": 0.1,

  "fusion_strategy": "average",

  "use_contrastive_loss": false,
  "link": "identity",
  "classification": false
}
```

### 步骤2: 使用配置文件初始化模型

```python
import json
from models.alignn import ALIGNN, ALIGNNConfig

# 加载配置
with open('config_solution1.json', 'r') as f:
    config_dict = json.load(f)

# 创建配置对象
config = ALIGNNConfig(**config_dict)

# 初始化模型
model = ALIGNN(config)
```

### 步骤3: 完整训练脚本示例

创建 `train_solution1.py`:

```python
#!/usr/bin/env python
"""训练方案1: 中期融合 + 细粒度注意力 + 门控融合"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders  # 你的数据加载器
from train import train_dgl  # 你的训练函数

def main():
    # 1. 加载配置
    with open('config_solution1.json', 'r') as f:
        config_dict = json.load(f)

    model_config = ALIGNNConfig(**config_dict)

    # 2. 初始化模型
    model = ALIGNN(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("=" * 80)
    print("方案1: 中期融合 + 细粒度注意力 + 门控融合")
    print("=" * 80)
    print(f"use_middle_fusion: {model_config.use_middle_fusion}")
    print(f"use_fine_grained_attention: {model_config.use_fine_grained_attention}")
    print(f"use_cross_modal_attention: {model_config.use_cross_modal_attention}")
    print(f"fusion_strategy: {model_config.fusion_strategy}")
    print(f"gated_fusion_type: {model_config.gated_fusion_type}")
    print("=" * 80)

    # 3. 准备数据
    train_loader, val_loader, test_loader = get_train_val_loaders(
        dataset='jarvis',
        target='formation_energy',
        batch_size=64,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # 4. 训练配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    criterion = nn.L1Loss()  # MAE loss

    # 5. 训练
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(300):
        # 训练一个epoch
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            # 前向传播（支持返回gate值用于可解释性）
            output = model(batch, return_attention=True)

            if isinstance(output, dict):
                predictions = output['predictions']
                # 可选：记录门控值
                if 'gate_values' in output:
                    gate_graph = output['gate_values']['gate_graph'].mean()
                    gate_text = output['gate_values']['gate_text'].mean()
                    # print(f"Gate values: graph={gate_graph:.3f}, text={gate_text:.3f}")
            else:
                predictions = output

            # 计算损失
            loss = criterion(predictions, batch.y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0

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

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/300 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config_dict,
            }, 'best_model_solution1.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
```

---

## 🔧 方法2: 更新现有训练脚本

如果你想直接使用 `train_with_cross_modal_attention.py`，需要添加新参数。

### 添加到参数解析器（在 `get_parser()` 函数中）

在 `train_with_cross_modal_attention.py` 的参数定义部分添加：

```python
# 在第170行左右，对比学习参数之后添加：

# 融合策略参数 (NEW!)
parser.add_argument('--fusion_strategy', type=str, default='gated',
                    choices=['average', 'concat', 'gated'],
                    help='融合策略: average (平均), concat (拼接), gated (门控)')
parser.add_argument('--gated_fusion_type', type=str, default='dual_gate',
                    choices=['single_gate', 'dual_gate', 'attention'],
                    help='门控融合类型')
parser.add_argument('--gated_fusion_hidden_dim', type=int, default=128,
                    help='门控融合隐藏层维度')
parser.add_argument('--gated_fusion_dropout', type=float, default=0.1,
                    help='门控融合dropout率')

# 跨模态注意力类型 (NEW!)
parser.add_argument('--cross_modal_attention_type', type=str, default='bidirectional',
                    choices=['bidirectional', 'unidirectional'],
                    help='跨模态注意力类型: bidirectional (双向), unidirectional (单向)')
```

### 更新配置构建（在 `main()` 函数中）

找到创建 `ALIGNNConfig` 的位置，添加新参数：

```python
# 构建模型配置
model_config = ALIGNNConfig(
    name="alignn",
    alignn_layers=args.alignn_layers,
    gcn_layers=args.gcn_layers,
    hidden_features=args.hidden_features,
    output_features=1,
    graph_dropout=args.graph_dropout,

    # 中期融合
    use_middle_fusion=args.use_middle_fusion,
    middle_fusion_layers=args.middle_fusion_layers,
    middle_fusion_hidden_dim=args.middle_fusion_hidden_dim,
    middle_fusion_num_heads=args.middle_fusion_num_heads,
    middle_fusion_dropout=args.middle_fusion_dropout,

    # 细粒度注意力
    use_fine_grained_attention=args.use_fine_grained_attention,
    fine_grained_hidden_dim=args.fine_grained_hidden_dim,
    fine_grained_num_heads=args.fine_grained_num_heads,
    fine_grained_dropout=args.fine_grained_dropout,
    fine_grained_use_projection=args.fine_grained_use_projection,

    # 全局跨模态注意力 (NEW!)
    use_cross_modal_attention=args.use_cross_modal,
    cross_modal_attention_type=args.cross_modal_attention_type,  # NEW!
    cross_modal_hidden_dim=args.cross_modal_hidden_dim,
    cross_modal_num_heads=args.cross_modal_num_heads,
    cross_modal_dropout=args.cross_modal_dropout,

    # 融合策略 (NEW!)
    fusion_strategy=args.fusion_strategy,  # NEW!
    gated_fusion_type=args.gated_fusion_type,  # NEW!
    gated_fusion_hidden_dim=args.gated_fusion_hidden_dim,  # NEW!
    gated_fusion_dropout=args.gated_fusion_dropout,  # NEW!

    # 对比学习
    use_contrastive_loss=args.use_contrastive,
    contrastive_loss_weight=args.contrastive_weight,
    contrastive_temperature=args.contrastive_temperature,

    classification=bool(args.classification),
)
```

---

## 📊 完整训练命令示例

更新脚本后，使用以下命令训练：

### 方案1: 中期+细粒度+门控融合

```bash
python train_with_cross_modal_attention.py \
  --dataset jarvis \
  --property formation_energy \
  --batch_size 64 \
  --epochs 300 \
  --learning_rate 0.001 \
  --alignn_layers 4 \
  --gcn_layers 4 \
  --hidden_features 256 \
  --use_middle_fusion True \
  --middle_fusion_layers "2" \
  --middle_fusion_hidden_dim 128 \
  --use_fine_grained_attention True \
  --fine_grained_num_heads 8 \
  --fine_grained_hidden_dim 256 \
  --use_cross_modal False \
  --fusion_strategy gated \
  --gated_fusion_type dual_gate \
  --gated_fusion_hidden_dim 128 \
  --output_dir experiments/solution1
```

### 方案2: 中期+细粒度+单向注意力

```bash
python train_with_cross_modal_attention.py \
  --dataset jarvis \
  --property formation_energy \
  --batch_size 64 \
  --epochs 300 \
  --learning_rate 0.001 \
  --alignn_layers 4 \
  --gcn_layers 4 \
  --hidden_features 256 \
  --use_middle_fusion True \
  --middle_fusion_layers "2" \
  --middle_fusion_hidden_dim 128 \
  --use_fine_grained_attention True \
  --fine_grained_num_heads 8 \
  --fine_grained_hidden_dim 256 \
  --use_cross_modal True \
  --cross_modal_attention_type unidirectional \
  --cross_modal_num_heads 4 \
  --fusion_strategy average \
  --output_dir experiments/solution2
```

### 方案3: 双向注意力+门控融合（对比）

```bash
python train_with_cross_modal_attention.py \
  --dataset jarvis \
  --property formation_energy \
  --batch_size 64 \
  --epochs 300 \
  --use_middle_fusion True \
  --middle_fusion_layers "2" \
  --use_fine_grained_attention True \
  --use_cross_modal True \
  --cross_modal_attention_type bidirectional \
  --fusion_strategy gated \
  --gated_fusion_type dual_gate \
  --output_dir experiments/solution3
```

---

## 🔬 消融实验设计

### 实验1: 无中期融合

```bash
python train_with_cross_modal_attention.py \
  --use_middle_fusion False \
  --use_fine_grained_attention True \
  --use_cross_modal False \
  --fusion_strategy gated \
  --output_dir experiments/ablation_no_middle
```

### 实验2: 无细粒度注意力

```bash
python train_with_cross_modal_attention.py \
  --use_middle_fusion True \
  --use_fine_grained_attention False \
  --use_cross_modal False \
  --fusion_strategy gated \
  --output_dir experiments/ablation_no_fine
```

### 实验3: 不同门控类型对比

```bash
# Single gate
python train_with_cross_modal_attention.py \
  --gated_fusion_type single_gate \
  --output_dir experiments/gate_single

# Dual gate
python train_with_cross_modal_attention.py \
  --gated_fusion_type dual_gate \
  --output_dir experiments/gate_dual

# Attention-based
python train_with_cross_modal_attention.py \
  --gated_fusion_type attention \
  --output_dir experiments/gate_attention
```

---

## 📈 训练监控和可解释性

### 记录门控值

在训练循环中添加：

```python
if 'gate_values' in output:
    gate_values = output['gate_values']

    # 记录平均门控值
    if 'gate_graph' in gate_values:
        avg_gate_graph = gate_values['gate_graph'].mean().item()
        avg_gate_text = gate_values['gate_text'].mean().item()

        # 使用tensorboard或wandb记录
        writer.add_scalar('gate/graph', avg_gate_graph, global_step)
        writer.add_scalar('gate/text', avg_gate_text, global_step)
```

### 可视化注意力权重

```python
if 'attention_weights' in output:
    attn = output['attention_weights']

    # 保存注意力热图
    if epoch % 10 == 0:
        visualize_attention(attn, save_path=f'attn_epoch_{epoch}.png')
```

---

## 🎯 超参数调优建议

### 方案1推荐超参数

| 参数 | 推荐范围 | 默认值 |
|------|---------|--------|
| `middle_fusion_layers` | "1", "2", "2,3" | "2" |
| `fine_grained_num_heads` | 4, 8 | 8 |
| `gated_fusion_hidden_dim` | 64, 128, 256 | 128 |
| `gated_fusion_type` | dual_gate, attention | dual_gate |
| `learning_rate` | 0.0001-0.001 | 0.001 |
| `batch_size` | 32, 64, 128 | 64 |

### 方案2推荐超参数

| 参数 | 推荐范围 | 默认值 |
|------|---------|--------|
| `cross_modal_num_heads` | 2, 4, 8 | 4 |
| `cross_modal_hidden_dim` | 128, 256 | 256 |
| `fusion_strategy` | average, concat | average |

---

## 💾 模型保存和加载

### 保存模型（包含配置）

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config.__dict__,
    'train_loss': train_loss,
    'val_loss': val_loss,
}, 'checkpoint.pth')
```

### 加载模型

```python
# 加载checkpoint
checkpoint = torch.load('checkpoint.pth')

# 重建配置
config = ALIGNNConfig(**checkpoint['config'])

# 重建模型
model = ALIGNN(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器（如果继续训练）
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 🐛 常见问题排查

### Q1: 显存不足

**解决方法：**
```bash
# 减小batch size
--batch_size 32

# 减少注意力头数
--fine_grained_num_heads 4
--cross_modal_num_heads 2

# 减小隐藏层维度
--hidden_features 128
--fine_grained_hidden_dim 128
```

### Q2: 训练不收敛

**检查项：**
1. 学习率是否过大？尝试 `--learning_rate 0.0001`
2. Dropout是否过高？尝试 `--fine_grained_dropout 0.05`
3. 是否需要warmup？`--warmup_steps 1000`

### Q3: 验证集性能比训练集差很多

**可能原因：**
- 过拟合，增加dropout
- 数据量不足，使用数据增强
- 模型过于复杂，尝试方案1（更轻量）

### Q4: 门控值都接近0或1

**正常现象：**
- 接近0或1说明模型学到了清晰的模态选择策略
- 如果想要更平滑的分布，降低 `gated_fusion_hidden_dim`

---

## 📊 性能评估

### 评估脚本

创建 `evaluate.py`:

```python
import torch
from models.alignn import ALIGNN, ALIGNNConfig

# 加载模型
checkpoint = torch.load('best_model_solution1.pth')
config = ALIGNNConfig(**checkpoint['config'])
model = ALIGNN(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 在测试集上评估
test_mae = 0.0
test_rmse = 0.0

with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        predictions = output['predictions'] if isinstance(output, dict) else output

        # 计算指标
        mae = torch.abs(predictions - batch.y).mean()
        rmse = torch.sqrt(torch.mean((predictions - batch.y) ** 2))

        test_mae += mae.item()
        test_rmse += rmse.item()

test_mae /= len(test_loader)
test_rmse /= len(test_loader)

print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
```

---

## 🎉 总结

**推荐训练流程：**

1. **先训练方案1**（baseline）
   ```bash
   python train_solution1.py
   ```

2. **然后训练方案2**（对比）
   ```bash
   python train_solution2.py
   ```

3. **运行消融实验**（理解各组件贡献）

4. **分析可解释性**（门控值、注意力权重）

5. **超参数调优**（针对最佳方案）

**预期结果：**
- 方案1应该比baseline（无融合）提升5-15%
- 方案2可能略优于方案1，但计算开销更大
- 消融实验应该显示每个组件都有正向贡献

祝训练顺利！🚀
