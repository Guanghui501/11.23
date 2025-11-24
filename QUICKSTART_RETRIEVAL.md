# 🚀 Retrieval 评估快速开始

## 一键运行

### 方式 1: 使用 Shell 脚本（推荐）

```bash
# 1. 修改脚本中的检查点路径
vim run_retrieval_evaluation.sh
# 修改 CHECKPOINT="checkpoints/best_model.pt"

# 2. 运行评估
./run_retrieval_evaluation.sh
```

### 方式 2: 直接使用 Python

```bash
python evaluate_retrieval.py \
    --checkpoint checkpoints/best_model.pt \
    --split val \
    --max_samples 1000
```

### 方式 3: 使用简化示例

```bash
# 修改 demo_retrieval.py 中的路径后运行
python demo_retrieval.py
```

---

## 消融实验（比较多个模型）

```bash
# 1. 修改脚本中的模型路径
vim run_ablation_retrieval.sh

# 2. 运行批量评估
./run_ablation_retrieval.sh

# 3. 查看汇总结果
cat retrieval_ablation_results/summary.txt
```

---

## 在训练脚本中集成

在你的 `train.py` 中添加：

```python
from demo_retrieval import quick_retrieval_check

# 训练循环中
for epoch in range(num_epochs):
    train_one_epoch(...)

    # 每 5 个 epoch 检查检索性能
    if epoch % 5 == 0:
        model.eval()
        metrics = quick_retrieval_check(model, val_loader, num_samples=100)
        print(f"Epoch {epoch} - Retrieval R@1: {metrics['avg_R@1']*100:.2f}%")

        # 记录到 tensorboard
        writer.add_scalar('Retrieval/R@1', metrics['avg_R@1'], epoch)

        model.train()
```

---

## 性能优化建议

### 数据集变大了？调整这些超参数：

| 超参数 | 小数据集 | 大数据集 | 原因 |
|--------|---------|---------|------|
| **Learning Rate** | 1e-4 | 2e-4 | 更稳定的梯度 |
| **Batch Size** | 32 | 128 | 充分利用 GPU |
| **Epochs** | 200 | 100 | 已见足够样本 |
| **Dropout** | 0.1 | 0.0-0.05 | 数据自带正则化 |
| **Weight Decay** | 1e-4 | 1e-5 | 减少正则化 |

### 提高 R@1 的配置：

```python
config = ALIGNNConfig(
    # 🔥 最关键：启用所有融合机制
    use_middle_fusion=True,
    middle_fusion_layers="2,3",

    use_fine_grained_attention=True,
    fine_grained_num_heads=8,

    use_cross_modal_attention=True,
    cross_modal_num_heads=4,

    # 🔥 对比学习损失
    use_contrastive_loss=True,
    contrastive_loss_weight=0.1,
    contrastive_temperature=0.1,

    # 正则化（根据数据集大小调整）
    graph_dropout=0.0,  # 大数据集用 0.0，小数据集用 0.1
)
```

---

## 预期性能

| R@1 范围 | 评级 | 说明 |
|---------|------|------|
| **85%+** | 🏆 优秀 | 生产可用，对齐能力强 |
| **70-85%** | 👍 良好 | 继续优化可达优秀 |
| **50-70%** | 😐 一般 | 融合机制部分起效 |
| **<50%** | ❌ 较差 | 检查配置和训练 |

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `evaluate_retrieval.py` | 完整的检索评估脚本 |
| `demo_retrieval.py` | 简化示例 + 模型对比 |
| `run_retrieval_evaluation.sh` | 一键评估脚本 |
| `run_ablation_retrieval.sh` | 消融实验批量评估 |
| `RETRIEVAL_README.md` | 详细文档 |
| `QUICKSTART_RETRIEVAL.md` | 本文件（快速开始）|

---

## 常见问题速查

| 问题 | 快速解决 |
|------|---------|
| R@1 < 30% | 启用 `use_middle_fusion=True` |
| 评估太慢 | 设置 `--max_samples 500` |
| 想对比模型 | 运行 `./run_ablation_retrieval.sh` |
| 训练时监控 | 使用 `quick_retrieval_check()` |

---

## 下一步

1. ✅ 运行基线评估
2. ✅ 启用融合机制
3. ✅ 添加对比学习
4. ✅ 调整超参数
5. ✅ 达到 80%+ R@1

**Good Luck!** 🚀
