# Multimodal Retrieval Evaluation (R@1)

本文档介绍如何使用 `evaluate_retrieval.py` 脚本评估多模态模型的对齐能力。

## 概述

R@1 (Retrieval at Rank 1) 是评估多模态对齐能力的**金标准指标**。该评估脚本计算以下指标：

### 检索指标

- **R@k (Recall at k)**: 在前k个检索结果中找到正确匹配的比例
  - R@1: 最严格的指标，要求第一个检索结果就是正确的
  - R@5, R@10: 更宽松的指标，允许在前5或10个结果中找到正确匹配

- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
  - 考虑了正确匹配的排名位置
  - 值越高越好（100%为最佳）

### 检索方向

评估脚本计算**双向检索**指标：

1. **Graph→Text (G2T)**: 给定图结构，检索对应的文本描述
2. **Text→Graph (T2G)**: 给定文本描述，检索对应的图结构
3. **Average**: 两个方向的平均值（最常用的整体指标）

## 快速开始

### 基本用法

```bash
# 评估测试集
python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split test \
    --visualize

# 评估验证集
python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --split val \
    --visualize
```

### 完整参数

```bash
python evaluate_retrieval.py \
    --model_path outputs/best_val_model.pt \
    --config_path outputs/config.json \
    --split test \
    --output_dir outputs/retrieval_results \
    --batch_size 32 \
    --device cuda \
    --top_k 10 \
    --visualize
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | **必需** | 模型检查点路径 (.pt文件) |
| `--config_path` | str | None | 配置文件路径（如不提供，从模型目录查找） |
| `--split` | str | test | 评估数据集划分 (train/val/test) |
| `--output_dir` | str | None | 输出目录（默认：模型目录/retrieval_results） |
| `--batch_size` | int | 32 | 批次大小 |
| `--device` | str | cuda | 设备 (cuda/cpu) |
| `--top_k` | int | 10 | 保存前k个检索结果 |
| `--visualize` | flag | False | 生成可视化图表 |

## 输出文件

评估完成后，将在输出目录生成以下文件：

### 1. 指标文件

**`retrieval_metrics_{split}.json`**
```json
{
  "G2T_R@1": 75.5,
  "G2T_R@5": 92.3,
  "G2T_R@10": 96.8,
  "G2T_MRR": 82.1,
  "T2G_R@1": 73.2,
  "T2G_R@5": 90.1,
  "T2G_R@10": 95.4,
  "T2G_MRR": 80.5,
  "Avg_R@1": 74.35,
  "Avg_R@5": 91.2,
  "Avg_R@10": 96.1,
  "Avg_MRR": 81.3
}
```

### 2. 详细检索结果

**`graph_to_text_retrieval.json`**
- 每个图样本的top-k检索结果
- 包含检索的样本ID、目标值、相似度分数
- 正确匹配的排名

**`text_to_graph_retrieval.json`**
- 每个文本样本的top-k检索结果
- 结构同上

### 3. 可视化图表 (使用 `--visualize`)

**`similarity_matrix_{split}.png`**
- 图-文本相似度矩阵热图
- 对角线应该最亮（正确匹配）

**`g2t_rank_dist_{split}.png`**
- Graph→Text检索的排名分布直方图
- 显示大多数样本在第几名找到正确匹配

**`t2g_rank_dist_{split}.png`**
- Text→Graph检索的排名分布直方图

## 评估结果解读

### 优秀的模型应该表现为：

✅ **R@1 > 70%**: 大多数情况下第一个检索结果就是正确的
✅ **R@5 > 90%**: 几乎所有情况前5个结果包含正确答案
✅ **MRR > 80%**: 正确答案平均排在很靠前的位置
✅ **相似度矩阵**: 对角线明显高于其他位置（正样本相似度高）
✅ **排名分布**: 大部分样本排名在1-3之间

### 需要改进的迹象：

⚠️ **R@1 < 50%**: 模型的区分能力较弱
⚠️ **G2T ≠ T2G**: 两个方向差异大，说明模态对齐不平衡
⚠️ **相似度矩阵**: 对角线不突出，说明正负样本难以区分
⚠️ **排名分布**: 排名分散，说明检索不稳定

## 使用场景

### 1. 模型评估

评估训练好的模型的多模态对齐质量：
```bash
python evaluate_retrieval.py \
    --model_path outputs/exp1/best_val_model.pt \
    --split test \
    --visualize
```

### 2. 模型对比

比较不同模型的检索性能：
```bash
# 评估模型1
python evaluate_retrieval.py \
    --model_path outputs/baseline/best_val_model.pt \
    --output_dir outputs/baseline/retrieval \
    --split test

# 评估模型2
python evaluate_retrieval.py \
    --model_path outputs/improved/best_val_model.pt \
    --output_dir outputs/improved/retrieval \
    --split test
```

然后比较两个输出目录中的 `retrieval_metrics_test.json`

### 3. 错误分析

分析检索失败的案例：
```python
import json

# 加载检索结果
with open('outputs/retrieval_results/graph_to_text_retrieval.json') as f:
    results = json.load(f)

# 找出检索失败的案例（排名>1）
failed_cases = [r for r in results if r['correct_rank'] > 1]

# 按排名排序，找出最严重的失败
failed_cases.sort(key=lambda x: x['correct_rank'], reverse=True)

# 查看前10个最差案例
for case in failed_cases[:10]:
    print(f"ID: {case['graph_id']}")
    print(f"Correct rank: {case['correct_rank']}")
    print(f"Top-5 retrieved: {case['top_k_ids'][:5]}")
    print()
```

## 技术细节

### 特征提取

脚本使用模型的 `return_features=True` 模式提取特征：
- Graph features: 图神经网络编码后的特征
- Text features: 文本编码器（MatSciBERT）编码后的特征

### 相似度计算

使用**余弦相似度**（L2归一化后的点积）：
```python
similarity = cosine_similarity(graph_features, text_features)
```

### 排名计算

对于每个query（图或文本），根据相似度对所有candidates排序，找到正确匹配的排名。

### 内存优化

- 批量提取特征（可调整 `--batch_size`）
- 在CPU上累积特征，避免GPU内存溢出
- 可视化时采样（大规模数据集）

## 常见问题

### Q: 为什么需要 `return_features=True`？

A: 检索评估需要分别访问图特征和文本特征，而不是最终融合后的预测。模型的 `forward()` 方法支持这个参数来返回中间特征。

### Q: 评估需要多长时间？

A: 取决于数据集大小和GPU。通常：
- 小数据集（1000样本）：1-2分钟
- 中等数据集（10000样本）：5-10分钟
- 大数据集（100000样本）：30-60分钟

### Q: 可以在CPU上运行吗？

A: 可以，但会慢很多。使用 `--device cpu`。

### Q: 如何提高R@1分数？

A: 考虑以下策略：
1. 使用对比学习损失（ContrastiveLoss）
2. 增加对比学习权重
3. 调整温度参数
4. 使用更大的批次大小（增加负样本多样性）
5. 添加hard negative mining

## 引用

如果这个评估脚本对你的研究有帮助，请引用相关的论文和仓库。

## 相关资源

- [CLIP Paper](https://arxiv.org/abs/2103.00020): 开创性的图文对比学习工作
- [ALIGN Paper](https://arxiv.org/abs/2102.05918): 大规模视觉-语言对齐
- [对比学习综述](https://arxiv.org/abs/2011.00362): 深入理解对比学习

---

**祝你的多模态模型取得优秀的检索性能！** 🚀
