# DynamicFusionModule 训练命令指南

## 🚀 快速开始

### 1. 验证集成（推荐先运行）
```bash
# 测试 DynamicFusionModule 是否正确集成
python test_integration.py
```

预期输出：
```
✅ All tests passed! Integration successful.
```

---

## 📋 训练命令

### 方式 1: 使用配置文件（推荐）

**标准训练（100 epochs）：**
```bash
python train.py --config config_dynamic_fusion.json
```

**快速测试（5 epochs）：**
```bash
python train.py --config config_dynamic_fusion.json --epochs 5
```

**小数据集测试：**
```bash
python train.py \
    --config config_dynamic_fusion.json \
    --n_train 100 \
    --n_val 20 \
    --n_test 20 \
    --epochs 10
```

---

### 方式 2: 命令行参数（更灵活）

**基础命令：**
```bash
python train.py \
    --config config.json \
    --model.use_middle_fusion True \
    --model.middle_fusion_layers "2" \
    --epochs 50 \
    --output_dir ./output_test
```

**完整参数示例：**
```bash
python train.py \
    --config config.json \
    --model.name alignn \
    --model.alignn_layers 4 \
    --model.gcn_layers 4 \
    --model.hidden_features 256 \
    --model.use_middle_fusion True \
    --model.middle_fusion_layers "2" \
    --model.middle_fusion_hidden_dim 128 \
    --model.middle_fusion_dropout 0.1 \
    --model.use_cross_modal_attention True \
    --model.cross_modal_num_heads 4 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --epochs 100 \
    --output_dir ./output_dynamic_fusion \
    --n_early_stopping 20
```

---

## 🎯 配置说明

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.use_middle_fusion` | `true` | 启用 DynamicFusionModule |
| `model.middle_fusion_layers` | `"2"` | 在哪些 ALIGNN 层应用融合 |
| `model.middle_fusion_hidden_dim` | `128` | 路由器隐藏层维度 |
| `model.middle_fusion_dropout` | `0.1` | Dropout 率 |
| `epochs` | `100` | 训练轮数 |
| `n_early_stopping` | `20` | 早停耐心值 |
| `output_dir` | `./output_dynamic_fusion` | 输出目录 |

### 多层融合示例

在第 2 和第 3 层都应用融合：
```bash
python train.py \
    --config config_dynamic_fusion.json \
    --model.middle_fusion_layers "2,3"
```

---

## 📊 训练过程监控

### 控制台输出

**启动时：**
```
✅ DynamicFusionModule weight monitoring enabled (logs every 5 epochs)
```

**每 5 个 epoch：**
```
================================================================================
DynamicFusionModule Weight Statistics (Epoch 50)
================================================================================

Fusion Module: layer_2
Updates: 15000

Router learned weights:
  w_graph: 0.6842
  w_text:  0.3158
  Sum:     1.0000

Effective weights (with double residual):
  Graph:  1.6842 (84.2%)
  Text:   0.3158 (15.8%)

✅ Graph dominant (ratio: 5.33x)
   This is expected for material property prediction.
================================================================================
```

### 输出文件

训练会生成以下文件在 `output_dir`：

| 文件 | 内容 |
|------|------|
| `best_val_model.pt` | 最佳验证集模型 |
| `best_test_model.pt` | 最佳测试集模型 |
| `fusion_weights.csv` | 权重演化记录 |
| `history_train.json` | 训练历史 |
| `history_val.json` | 验证历史 |
| `checkpoint_*_*.pt` | 训练检查点 |

---

## 🔍 结果分析

### 1. 查看权重演化

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取权重日志
df = pd.read_csv('output_dynamic_fusion/fusion_weights.csv')

# 绘制权重演化
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['layer_2_w_graph'], label='w_graph', linewidth=2)
plt.plot(df['epoch'], df['layer_2_w_text'], label='w_text', linewidth=2)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Weight', fontsize=12)
plt.title('Router Weight Evolution During Training', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('weight_evolution.png', dpi=300)
plt.show()

# 打印统计信息
print(f"Final w_graph: {df['layer_2_w_graph'].iloc[-1]:.4f}")
print(f"Final w_text:  {df['layer_2_w_text'].iloc[-1]:.4f}")
print(f"Final ratio:   {df['layer_2_eff_ratio'].iloc[-1]:.2f}x")
```

### 2. 查看训练历史

```python
import json

# 读取训练历史
with open('output_dynamic_fusion/history_val.json', 'r') as f:
    history = json.load(f)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png', dpi=300)
plt.show()
```

### 3. 加载最佳模型

```python
import torch
from models.alignn import ALIGNN

# 加载检查点
checkpoint = torch.load('output_dynamic_fusion/best_val_model.pt')

# 创建模型
model = ALIGNN(checkpoint['config'])
model.load_state_dict(checkpoint['model'])

# 查看权重统计
from monitor_fusion_weights import print_fusion_weights
print_fusion_weights(model)
```

---

## ⚠️ 故障排查

### 问题 1: ImportError

**错误：**
```
ImportError: cannot import name 'print_fusion_weights'
```

**解决：**
```bash
# 确保在项目根目录
ls monitor_fusion_weights.py
# 应该看到文件存在
```

### 问题 2: 权重不更新

**症状：**
```
update_count: 0
avg_w_graph: 0.0000
avg_w_text: 0.0000
```

**原因：** 模型处于评估模式

**解决：** 权重监控只在训练模式下更新，这是正常的。训练时会自动更新。

### 问题 3: 文本权重过高

**症状：**
```
⚠️ Warning: Text may have too much influence for physics tasks.
w_text > 0.5
```

**解决方案：**
1. 检查文本描述是否过于详细
2. 考虑限制文本权重上限
3. 增加 `middle_fusion_dropout`

---

## 🎯 性能基准

### 预期权重范围（材料性质预测）

| 指标 | 健康范围 | 优秀范围 | 警告 |
|------|---------|---------|------|
| w_graph | 0.5-0.9 | 0.6-0.8 | < 0.3 |
| w_text | 0.1-0.5 | 0.2-0.4 | > 0.7 |
| 图/文本比例 | 2-10x | 3-6x | < 2x |
| 有效图权重 | 1.5-1.9 | 1.6-1.8 | < 1.3 |

### 对比实验

**与原始模型对比：**
```bash
# 1. 训练原始模型（无融合）
python train.py --config config.json --output_dir ./output_baseline

# 2. 训练 DynamicFusion 模型
python train.py --config config_dynamic_fusion.json

# 3. 对比结果
python compare_fusion_mechanisms.py \
    --baseline output_baseline/best_val_model.pt \
    --fusion output_dynamic_fusion/best_val_model.pt
```

---

## 💡 高级用法

### 1. 多层融合策略

**早期和中期融合：**
```bash
python train.py \
    --config config_dynamic_fusion.json \
    --model.middle_fusion_layers "1,2"
```

**全层融合：**
```bash
python train.py \
    --config config_dynamic_fusion.json \
    --model.middle_fusion_layers "0,1,2,3"
```

### 2. 与对比学习联用

```bash
python train.py \
    --config config_dynamic_fusion.json \
    --model.use_contrastive_loss True \
    --model.contrastive_loss_weight 0.1
```

### 3. 分布式训练

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --config config_dynamic_fusion.json \
    --distributed True
```

---

## 📞 需要帮助？

**诊断脚本：**
```bash
# 运行完整诊断
python test_integration.py
python test_residual_impact.py
```

**查看日志：**
```bash
# 检查权重日志
cat output_dynamic_fusion/fusion_weights.csv

# 检查训练历史
cat output_dynamic_fusion/history_val.json
```

**快速测试：**
```bash
# 5 epoch 测试运行
python train.py \
    --config config_dynamic_fusion.json \
    --n_train 100 \
    --epochs 5 \
    --output_dir ./test_run
```

---

## 🎓 参考文档

- **实现细节**: `models/alignn.py` (第 121-257 行)
- **集成指南**: `INTEGRATION_CHECKLIST.md`
- **监控工具**: `monitor_fusion_weights.py`
- **原理分析**: `test_residual_impact.py`
