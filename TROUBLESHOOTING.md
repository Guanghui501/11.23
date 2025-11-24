# 🔧 问题排查指南 (Troubleshooting Guide)

## 问题 1: KeyError: 'model_state_dict'

### 错误信息
```
KeyError: 'model_state_dict'
```

### 原因
你的检查点保存格式和代码期望的不同。常见的保存格式有：

1. **完整字典格式**（推荐）：
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'epoch': epoch,
       'loss': loss
   }, 'checkpoint.pt')
   ```

2. **仅模型权重**：
   ```python
   torch.save(model.state_dict(), 'checkpoint.pt')
   ```

3. **其他键名**：
   ```python
   torch.save({
       'model': model.state_dict(),  # 不是 'model_state_dict'
       'epoch': epoch
   }, 'checkpoint.pt')
   ```

### ✅ 解决方案

#### 方法 1: 使用智能加载工具（推荐）

我已经创建了 `utils_retrieval.py`，它能自动检测并处理所有格式：

```python
from utils_retrieval import load_model_checkpoint

# 自动处理任何格式
model, checkpoint_info = load_model_checkpoint(
    model,
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',
    verbose=True  # 打印加载信息
)
```

**优点**：
- ✅ 自动检测格式
- ✅ 友好的错误信息
- ✅ 支持多种键名
- ✅ 返回额外的检查点信息

#### 方法 2: 先检查检查点格式

```bash
# 查看你的检查点包含什么
python check_checkpoint.py checkpoints/best_model.pt
```

**输出示例**：
```
🔍 检查点信息: checkpoints/best_model.pt
================================================================================
📦 检查点是字典，包含以下键:
  - epoch: int = 100
  - best_val_loss: float = 0.123
  - model: dict with 142 items
      - atom_embedding.layer.0.weight
      - atom_embedding.layer.0.bias
      - ...

🔎 检测到的可能的模型权重键:
  ✅ 'model' 存在
  ❌ 'model_state_dict' 不存在
  ❌ 'state_dict' 不存在
  ❌ 'net' 不存在
================================================================================
```

然后根据输出修改代码：
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])  # 使用正确的键名
```

#### 方法 3: 手动处理

```python
import torch

checkpoint = torch.load('checkpoint.pt', map_location='cuda')

# 尝试不同的键名
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # 可能整个 checkpoint 就是 state_dict
        state_dict = checkpoint
else:
    # checkpoint 直接是 state_dict
    state_dict = checkpoint

model.load_state_dict(state_dict)
```

---

## 问题 2: 数据集变大后如何调整超参数？

### 背景
当数据集规模增加（如从 1k → 10k+ 样本）时，需要调整超参数以：
- 充分利用更多数据
- 加快训练速度
- 避免过拟合

### ✅ 推荐调整

| 超参数 | 小数据集 (1k) | 大数据集 (10k+) | 调整理由 |
|--------|--------------|----------------|---------|
| **Learning Rate** | 1e-4 | 2e-4 至 3e-4 | 更多数据使梯度估计更稳定 |
| **Batch Size** | 32-64 | 128-256 | 充分利用 GPU，提高效率 |
| **Epochs** | 200 | 50-100 | 每个 epoch 已见更多样本 |
| **Dropout** | 0.1 | 0.0-0.05 | 更多数据本身就是正则化 |
| **Weight Decay** | 1e-4 | 1e-5 | 减少正则化强度 |
| **Warmup Steps** | 100 | 500-1000 | 给更大的 LR 缓冲期 |

### 配置示例

**小数据集 (~1k 样本)**：
```python
config = {
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 200,
    'graph_dropout': 0.1,
    'weight_decay': 1e-4,
    'warmup_steps': 100,
}
```

**大数据集 (10k+ 样本)**：
```python
config = {
    'learning_rate': 2e-4,      # 提高 2x
    'batch_size': 128,          # 提高 4x
    'epochs': 100,              # 减少 2x
    'graph_dropout': 0.05,      # 减少 2x
    'weight_decay': 1e-5,       # 减少 10x
    'warmup_steps': 500,        # 提高 5x
}
```

### 调整策略

1. **先调整 Batch Size**
   - 尽可能增大（GPU 内存允许）
   - 从 32 → 64 → 128 → 256
   - 每次翻倍测试

2. **再调整 Learning Rate**
   - 规则：Batch Size 翻倍 → LR 提高 √2
   - 例：Batch 32→128 (4x) → LR 1e-4→2e-4 (2x)

3. **减少训练轮数**
   - 计算总样本数：`total_samples = epochs × dataset_size`
   - 保持总样本数相近即可
   - 例：1k 样本 × 200 epochs = 200k 总样本
   - 10k 样本 × 20 epochs = 200k 总样本

4. **降低正则化**
   - Dropout: 0.1 → 0.05 → 0.0
   - Weight Decay: 1e-4 → 1e-5
   - 更多数据不需要强正则化

---

## 问题 3: R@1 很低 (<30%)

### 可能原因

1. **没有启用融合机制**
   ```python
   config.use_middle_fusion = False  # ❌
   config.use_cross_modal_attention = False  # ❌
   ```

2. **对比学习损失未启用**
   ```python
   config.use_contrastive_loss = False  # ❌
   ```

3. **训练不充分**
   - Epochs 太少
   - Learning rate 太小

### ✅ 解决方案

```python
config = ALIGNNConfig(
    # 🔥 启用所有融合机制
    use_middle_fusion=True,
    middle_fusion_layers="2,3",

    use_fine_grained_attention=True,
    fine_grained_num_heads=8,

    use_cross_modal_attention=True,
    cross_modal_num_heads=4,

    # 🔥 对比学习
    use_contrastive_loss=True,
    contrastive_loss_weight=0.1,
    contrastive_temperature=0.1,
)
```

---

## 问题 4: 评估太慢

### 原因
- 数据集太大
- 每次都完整评估

### ✅ 解决方案

#### 方案 1: 减少样本数
```bash
python evaluate_retrieval.py \
    --checkpoint best_model.pt \
    --max_samples 500  # 只评估 500 个样本
```

#### 方案 2: 训练时快速检查
```python
from demo_retrieval import quick_retrieval_check

# 每 5 个 epoch 快速检查
if epoch % 5 == 0:
    metrics = quick_retrieval_check(
        model, val_loader,
        num_samples=100  # 只用 100 个样本
    )
    print(f"R@1: {metrics['avg_R@1']*100:.2f}%")
```

#### 方案 3: 只计算 R@1
```bash
python evaluate_retrieval.py \
    --checkpoint best_model.pt \
    --k_values 1  # 只计算 R@1，不算 R@5, R@10
    --no_visualize  # 不生成图表
```

---

## 问题 5: Graph→Text 和 Text→Graph 性能差异大

### 示例
```
Graph→Text R@1: 85%
Text→Graph R@1: 45%  # 差距太大！
```

### 原因
- 模态不平衡
- 一个模态的特征维度或表达能力更强

### ✅ 解决方案

1. **确保投影维度相同**
   ```python
   graph_projection_dim = 64
   text_projection_dim = 64  # 必须相同
   ```

2. **调整对比学习温度**
   ```python
   contrastive_temperature = 0.1  # 降低温度使分布更平衡
   ```

3. **检查特征范数**
   ```python
   print(f"Graph feature norm: {graph_features.norm(dim=1).mean()}")
   print(f"Text feature norm: {text_features.norm(dim=1).mean()}")
   # 应该接近（因为都做了 L2 归一化）
   ```

---

## 问题 6: 导入错误

### 错误信息
```
ImportError: cannot import name 'get_train_val_loaders' from 'data'
```

### 原因
- 代码示例中的导入路径可能与你的项目结构不同

### ✅ 解决方案

根据你的项目结构修改导入：

```python
# 如果你的数据加载器在不同位置
from your_project.data_loader import get_loaders  # 修改这里

# 或者直接在脚本中创建 dataloader
from torch.utils.data import DataLoader
from your_dataset import YourDataset

dataset = YourDataset(...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
```

---

## 快速参考

### 检查检查点格式
```bash
python check_checkpoint.py checkpoints/best_model.pt
```

### 运行完整评估
```bash
./run_retrieval_evaluation.sh
```

### 快速评估（训练中）
```python
from demo_retrieval import quick_retrieval_check
metrics = quick_retrieval_check(model, val_loader, num_samples=100)
```

### 模型对比
```bash
./run_ablation_retrieval.sh
```

---

## 获取帮助

1. **查看详细文档**：`RETRIEVAL_README.md`
2. **快速开始**：`QUICKSTART_RETRIEVAL.md`
3. **检查示例**：`demo_retrieval.py`
4. **检查检查点**：`python check_checkpoint.py <path>`

如果问题依然存在，请提供：
- 错误的完整堆栈跟踪
- 检查点文件的结构（使用 `check_checkpoint.py`）
- 你的模型配置

祝使用顺利！🚀
