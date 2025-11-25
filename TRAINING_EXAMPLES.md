# DynamicFusionModule 训练示例
## 使用 train_with_cross_modal_attention.py

---

## 🎯 推荐：快速开始

### 1. 快速测试（5 epochs，小数据集）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --n_train 100 \
    --n_val 20 \
    --n_test 20 \
    --epochs 5 \
    --batch_size 16 \
    --output_dir ./output_test/
```

### 2. 标准训练（带 DynamicFusionModule）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_dropout 0.1 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --batch_size 32 \
    --epochs 100 \
    --early_stopping_patience 20 \
    --output_dir ./output_dynamic_fusion/
```

### 3. 使用一键脚本

```bash
chmod +x run_dynamic_fusion_training.sh
./run_dynamic_fusion_training.sh
```

---

## 📋 不同任务的训练命令

### JARVIS - Formation Energy（形成能）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 100 \
    --batch_size 32 \
    --output_dir ./output/jarvis_fe/
```

### JARVIS - Band Gap（带隙）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 100 \
    --batch_size 32 \
    --output_dir ./output/jarvis_bg/
```

### Material Project - Band Gap

```bash
python train_with_cross_modal_attention.py \
    --dataset mp \
    --property band_gap \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --cross_modal_num_heads 8 \
    --n_train 60000 \
    --n_val 5000 \
    --n_test 4132 \
    --epochs 100 \
    --batch_size 64 \
    --output_dir ./output/mp_bg/
```

### JARVIS - Bulk Modulus（体积模量）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property bulk_modulus_kv \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --epochs 100 \
    --batch_size 32 \
    --output_dir ./output/jarvis_bulk/
```

---

## 🔧 高级配置

### 多层融合（在第2和第3层）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3" \
    --epochs 100 \
    --output_dir ./output_multi_layer/
```

### 联合使用细粒度注意力

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --use_cross_modal True \
    --epochs 100 \
    --output_dir ./output_full_fusion/
```

### 添加对比学习

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --contrastive_temperature 0.1 \
    --epochs 100 \
    --output_dir ./output_contrastive/
```

### 大模型配置（更多层 + 更高维度）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --alignn_layers 6 \
    --gcn_layers 6 \
    --hidden_features 512 \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3,4" \
    --middle_fusion_hidden_dim 256 \
    --use_cross_modal True \
    --cross_modal_hidden_dim 512 \
    --cross_modal_num_heads 8 \
    --batch_size 16 \
    --epochs 100 \
    --output_dir ./output_large_model/
```

---

## 📊 训练时的输出

### 启动时
```
==========================================
CrysMMNet 训练 - 跨模态注意力机制
==========================================

中期融合配置:
  启用: True
  融合层: 2
  隐藏维度: 128
  注意力头数: 2
  Dropout率: 0.1

✅ DynamicFusionModule weight monitoring enabled (logs every 5 epochs)
```

### 每 5 个 epoch
```
================================================================================
DynamicFusionModule Weight Statistics (Epoch 50)
================================================================================

Fusion Module: layer_2
Updates: 15000

Router learned weights (from Softmax competition):
  w_graph: 0.6842
  w_text:  0.3158
  Sum:     1.0000 (should be ~1.0)

Effective weights (with double residual):
  Graph:  1.6842 (84.2%)
  Text:   0.3158 (15.8%)

Interpretation:
  ✅ Graph dominant (ratio: 5.33x)
     This is expected for material property prediction.
================================================================================
```

---

## 🎛️ 关键参数说明

### DynamicFusionModule 参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--use_middle_fusion` | False | **True** | 启用动态融合 |
| `--middle_fusion_layers` | "2" | "2" 或 "2,3" | 在哪些层应用融合 |
| `--middle_fusion_hidden_dim` | 128 | 128-256 | 路由器隐藏维度 |
| `--middle_fusion_dropout` | 0.1 | 0.1 | Dropout 率 |

### 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--epochs` | 100-200 | 训练轮数 |
| `--batch_size` | 32-64 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--early_stopping_patience` | 20 | 早停耐心值 |

### 数据集参数

| 参数 | 示例 | 说明 |
|------|------|------|
| `--dataset` | jarvis, mp | 数据集名称 |
| `--property` | formation_energy_peratom | 预测性质 |
| `--root_dir` | ../dataset/ | 数据集根目录 |
| `--n_train` | None | 训练样本数（None=全部） |

---

## 📈 结果分析

### 查看权重演化

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取权重日志
df = pd.read_csv('output_dynamic_fusion/formation_energy_peratom/fusion_weights.csv')

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['layer_2_w_graph'], label='w_graph', linewidth=2)
plt.plot(df['epoch'], df['layer_2_w_text'], label='w_text', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.title('DynamicFusionModule Weight Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('weight_evolution.png', dpi=300)
plt.show()

# 统计信息
print(f"Final w_graph: {df['layer_2_w_graph'].iloc[-1]:.4f}")
print(f"Final w_text:  {df['layer_2_w_text'].iloc[-1]:.4f}")
print(f"Final ratio:   {df['layer_2_eff_ratio'].iloc[-1]:.2f}x")
```

### 查看训练历史

```bash
# 查看配置
cat output_dynamic_fusion/formation_energy_peratom/config.json

# 查看验证集历史
cat output_dynamic_fusion/formation_energy_peratom/history_val.json

# 查看权重日志
cat output_dynamic_fusion/formation_energy_peratom/fusion_weights.csv
```

### 加载最佳模型

```python
import torch
from models.alignn import ALIGNN

# 加载检查点
checkpoint = torch.load('output_dynamic_fusion/formation_energy_peratom/best_val_model.pt')

# 创建模型
model = ALIGNN(checkpoint['config'])
model.load_state_dict(checkpoint['model'])

# 查看权重统计
from monitor_fusion_weights import print_fusion_weights
print_fusion_weights(model)
```

---

## ⚠️ 常见问题

### Q1: FileNotFoundError: CIF目录不存在

**问题**：
```
❌ 错误: CIF目录不存在: ../dataset/jarvis/formation_energy_peratom/cif/
```

**解决**：
```bash
# 检查当前目录
pwd

# 调整 --root_dir 参数
# 如果在项目根目录：
--root_dir ./dataset/

# 如果在 src 目录：
--root_dir ../dataset/

# 或使用绝对路径：
--root_dir /path/to/your/dataset/
```

### Q2: 权重日志文件不存在

**问题**：训练完成但没有 `fusion_weights.csv`

**原因**：
- 训练轮数 < 5（监控每5轮记录一次）
- `--use_middle_fusion` 未设置为 True

**解决**：
```bash
# 确保启用中期融合
--use_middle_fusion True

# 至少训练 5 个 epoch
--epochs 5
```

### Q3: 文本权重过高

**症状**：
```
⚠️ Warning: Text may have too much influence for physics tasks.
ratio < 2x
```

**解决**：
1. 检查文本描述是否过于详细
2. 增加 `--middle_fusion_dropout`
3. 减少 `--middle_fusion_hidden_dim`

---

## 🎯 性能基准

### 预期权重范围

| 指标 | 健康范围 | 优秀范围 |
|------|---------|---------|
| w_graph | 0.5-0.9 | 0.6-0.8 |
| w_text | 0.1-0.5 | 0.2-0.4 |
| 图/文本比例 | 2-10x | 3-6x |

### 对比实验

**基线（无融合）：**
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --use_middle_fusion False \
    --output_dir ./output_baseline/
```

**DynamicFusion：**
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --output_dir ./output_dynamic/
```

---

## 💡 最佳实践

### 1. 渐进式训练

```bash
# Step 1: 快速测试（验证配置）
python train_with_cross_modal_attention.py \
    --use_middle_fusion True \
    --n_train 100 --epochs 5 \
    --output_dir ./test/

# Step 2: 中等规模（调参）
python train_with_cross_modal_attention.py \
    --use_middle_fusion True \
    --n_train 1000 --epochs 20 \
    --output_dir ./tune/

# Step 3: 完整训练
python train_with_cross_modal_attention.py \
    --use_middle_fusion True \
    --epochs 100 \
    --output_dir ./final/
```

### 2. 超参数搜索

```bash
# 搜索不同的融合层
for layer in "1" "2" "3" "2,3"; do
    python train_with_cross_modal_attention.py \
        --use_middle_fusion True \
        --middle_fusion_layers "$layer" \
        --output_dir "./output_layer_$layer/"
done

# 搜索不同的隐藏维度
for dim in 64 128 256; do
    python train_with_cross_modal_attention.py \
        --use_middle_fusion True \
        --middle_fusion_hidden_dim $dim \
        --output_dir "./output_dim_$dim/"
done
```

### 3. 多种子实验

```bash
# 运行多个随机种子
for seed in 123 456 789; do
    python train_with_cross_modal_attention.py \
        --use_middle_fusion True \
        --random_seed $seed \
        --output_dir "./output_seed_$seed/"
done
```

---

## 📚 参考

- **实现代码**: `models/alignn.py` (第 121-257 行)
- **监控工具**: `monitor_fusion_weights.py`
- **集成指南**: `INTEGRATION_CHECKLIST.md`
- **通用命令**: `TRAINING_COMMANDS.md`
