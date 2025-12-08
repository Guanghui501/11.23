# 方案1和方案2训练快速入门

## 🚀 快速开始（3种方法）

### 方法1: 使用快速启动脚本（最简单）⭐

```bash
./quick_start.sh
```

然后根据提示选择方案1、方案2或两者都训练。

---

### 方法2: 直接运行训练脚本

#### 训练方案1（推荐）

```bash
python train_solution1.py \
  --dataset jarvis \
  --property formation_energy \
  --batch_size 64 \
  --epochs 300
```

#### 训练方案2

```bash
python train_solution2.py \
  --dataset jarvis \
  --property formation_energy \
  --batch_size 64 \
  --epochs 300
```

---

### 方法3: 自定义配置

创建配置文件 `my_config.json`:

```json
{
  "name": "alignn",
  "use_middle_fusion": true,
  "middle_fusion_layers": "2",
  "use_fine_grained_attention": true,
  "use_cross_modal_attention": false,
  "fusion_strategy": "gated",
  "gated_fusion_type": "dual_gate"
}
```

然后在代码中加载：

```python
from models.alignn import ALIGNN, ALIGNNConfig

with open('my_config.json') as f:
    config = ALIGNNConfig(**json.load(f))

model = ALIGNN(config)
```

---

## 📊 两个方案的区别

| 特性 | 方案1 | 方案2 |
|------|-------|-------|
| **中期融合** | ✅ | ✅ |
| **细粒度注意力** | ✅ | ✅ |
| **全局注意力** | ❌ 无 | ✅ 单向 |
| **融合方式** | 门控 | 平均 |
| **计算速度** | 快 | 中等 |
| **参数量** | 少 | 中等 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**方案1推荐理由:**
- 已有中期融合和细粒度注意力做了充分交互
- 门控融合轻量且有效
- 避免过度融合

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `train_solution1.py` | 方案1训练脚本 |
| `train_solution2.py` | 方案2训练脚本 |
| `quick_start.sh` | 快速启动脚本 |
| `TRAINING_GUIDE.md` | 详细训练指南 |
| `FUSION_SOLUTIONS_GUIDE.md` | 方案选择指南 |
| `test_gated_fusion.py` | 测试脚本 |

---

## 🔧 命令行参数

### 方案1参数

```bash
python train_solution1.py --help

常用参数:
  --dataset              数据集 (jarvis, mp)
  --property             属性 (formation_energy, band_gap)
  --batch_size           批次大小 (默认: 64)
  --epochs               训练轮数 (默认: 300)
  --learning_rate        学习率 (默认: 0.001)
  --middle_fusion_layers 中期融合层 (默认: "2")
  --fine_grained_num_heads 细粒度注意力头数 (默认: 8)
  --gated_fusion_type    门控类型 (dual_gate, single_gate, attention)
  --output_dir           输出目录
```

### 方案2参数

```bash
python train_solution2.py --help

与方案1类似，额外参数:
  --cross_modal_num_heads  全局注意力头数 (默认: 4)
```

---

## 📈 训练监控

### 查看训练进度

```bash
# 实时查看loss
tail -f experiments/solution1/train.log

# 查看门控值历史（方案1）
cat experiments/solution1/gate_history.json

# 查看注意力历史（方案2）
cat experiments/solution2/attn_history.json
```

### 使用TensorBoard（可选）

如果你的训练脚本支持TensorBoard:

```bash
tensorboard --logdir experiments/
```

---

## 💾 模型保存

训练脚本会自动保存：

- `best_model.pth` - 验证集最佳模型
- `checkpoint_epoch_N.pth` - 每50个epoch的checkpoint
- `model_config.json` - 模型配置
- `gate_history.json` / `attn_history.json` - 训练历史

### 加载模型

```python
import torch
from models.alignn import ALIGNN, ALIGNNConfig

# 加载checkpoint
checkpoint = torch.load('experiments/solution1/best_model.pth')

# 重建模型
config = ALIGNNConfig(**checkpoint['config'])
model = ALIGNN(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 评估模式
model.eval()
```

---

## 🔍 可解释性分析

### 方案1: 查看门控值

```python
output = model(batch, return_attention=True)

if 'gate_values' in output:
    gate_graph = output['gate_values']['gate_graph']  # [batch, 64]
    gate_text = output['gate_values']['gate_text']    # [batch, 64]

    print(f"Graph contribution: {gate_graph.mean():.3f}")
    print(f"Text contribution: {gate_text.mean():.3f}")
```

### 方案2: 查看注意力权重

```python
output = model(batch, return_attention=True)

if 'attention_weights' in output:
    attn = output['attention_weights']
    print(f"Attention strength: {attn.mean():.4f}")
```

---

## 🐛 常见问题

### Q1: 显存不足

```bash
# 减小batch size
--batch_size 32

# 减少注意力头数
--fine_grained_num_heads 4
```

### Q2: 找不到数据集

```bash
# 指定数据集路径
--root_dir /path/to/dataset
```

### Q3: ImportError

```bash
# 确保安装了依赖
pip install torch dgl jarvis-tools transformers
```

### Q4: 训练太慢

```bash
# 使用更少的epoch
--epochs 100

# 增大batch size (如果显存允许)
--batch_size 128
```

---

## 📊 性能基准

基于JARVIS formation_energy数据集的预期性能（仅供参考）：

| 方案 | MAE | RMSE | 训练时间 |
|------|-----|------|---------|
| 方案1 | ~0.08 | ~0.12 | 2-3小时 |
| 方案2 | ~0.07 | ~0.11 | 3-4小时 |
| Baseline (无融合) | ~0.12 | ~0.18 | 1-2小时 |

*实际性能取决于数据集、硬件和超参数*

---

## 🎯 推荐工作流

```
1. 快速测试（10 epochs）
   python train_solution1.py --epochs 10

2. 完整训练方案1（baseline）
   python train_solution1.py --epochs 300

3. 完整训练方案2（对比）
   python train_solution2.py --epochs 300

4. 分析结果
   - 对比验证集性能
   - 查看门控值/注意力权重
   - 可视化预测结果

5. 超参数调优（基于最佳方案）
   - 调整learning_rate
   - 调整fusion_hidden_dim
   - 调整num_heads
```

---

## 📚 更多资源

- **详细训练指南**: `TRAINING_GUIDE.md`
- **方案选择指南**: `FUSION_SOLUTIONS_GUIDE.md`
- **门控融合详解**: `GATED_FUSION_GUIDE.md`
- **测试脚本**: `test_gated_fusion.py`

---

## 🆘 获取帮助

如果遇到问题：

1. 查看 `TRAINING_GUIDE.md` 中的详细说明
2. 运行测试脚本检查环境: `python test_gated_fusion.py`
3. 查看训练日志: `cat experiments/solution1/train.log`
4. 检查配置文件: `cat experiments/solution1/model_config.json`

---

## ✅ 开始训练！

最简单的方式：

```bash
./quick_start.sh
```

或者直接：

```bash
python train_solution1.py
```

祝训练顺利！🚀
