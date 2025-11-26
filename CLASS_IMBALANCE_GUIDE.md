# 二分类任务中的类别不平衡问题指南

## 🎯 快速回答

**测试集类别分布不平衡有影响吗？** → **是的，有显著影响！**

---

## ⚠️ 主要影响

### 1. 评估指标失真

假设测试集分布：
```
类别0: 900个样本 (90%)
类别1: 100个样本 (10%)
```

**情况A**: 模型全部预测为类别0
```
准确率 = 90% ✅  ← 看起来很好！
召回率（类别1）= 0% ❌  ← 完全失败！
```

**结论**: 准确率会严重误导，不能真实反映模型性能。

---

### 2. 模型优化偏向

训练时如果不处理不平衡：

| 模型行为 | 原因 | 后果 |
|---------|------|------|
| 倾向预测多数类 | 损失函数被多数类主导 | 少数类样本被忽略 |
| 决策边界偏移 | 优化整体准确率 | 少数类分类错误率高 |
| 特征学习不充分 | 少数类样本少 | 泛化能力差 |

---

### 3. 对不同指标的影响

| 指标 | 受影响程度 | 是否可靠 | 说明 |
|------|----------|---------|------|
| **准确率** | 🔴 严重 | ❌ | 被多数类主导，容易虚高 |
| **精确率** | 🟡 中等 | ⚠️ | 对少数类可能不准 |
| **召回率** | 🔴 严重 | ❌ | 少数类召回率通常很低 |
| **F1分数** | 🟢 轻微 | ✅ | 平衡了精确率和召回率 |
| **ROC-AUC** | 🟢 轻微 | ✅ | 对不平衡相对鲁棒 |
| **PR-AUC** | 🟢 轻微 | ✅ | 更适合不平衡数据 |

---

## 🔍 检查你的数据分布

### 方法1: 使用检查脚本

```bash
python check_class_distribution.py /path/to/your/id_prop.csv
```

**输出示例**:
```
📊 数据集总样本数: 1000

类别分布:
------------------------------------------------------------
  类别 0:    900 样本 (90.00%) ████████████████████████████████████████████████
  类别 1:    100 样本 (10.00%) █████

不平衡分析:
------------------------------------------------------------
  多数类 (类别0): 900 样本
  少数类 (类别1): 100 样本
  不平衡比率: 9.00:1

严重程度: 🟡 中度不平衡
建议: 建议使用类别权重或调整损失函数

💡 推荐配置:
------------------------------------------------------------
  pos_weight (用于BCEWithLogitsLoss): 9.0000
  class_weight={0: 1.0, 1: 9.0000}
```

### 方法2: 快速统计

```python
import pandas as pd

df = pd.read_csv('your_data.csv')
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))
```

---

## 🛠️ 解决方案

### 策略1: 使用加权损失函数 ⭐ 推荐

**原理**: 给少数类更高的权重，让模型更关注少数类。

#### a) BCEWithLogitsLoss + pos_weight

```python
# 计算pos_weight
num_class_0 = 900
num_class_1 = 100
pos_weight = num_class_0 / num_class_1  # 9.0

# 创建损失函数
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
```

**优点**:
- ✅ 简单有效
- ✅ 不改变数据分布
- ✅ 计算开销小

**训练脚本中使用**:
```bash
python train_with_cross_modal_attention.py \
    --classification 1 \
    --pos_weight 9.0 \
    ...
```

#### b) 自定义加权BCE损失

```python
class WeightedBCELoss(nn.Module):
    def __init__(self, weight_pos=1.0):
        super().__init__()
        self.weight_pos = weight_pos

    def forward(self, pred, target):
        loss = -(self.weight_pos * target * torch.log(pred + 1e-8) +
                 (1 - target) * torch.log(1 - pred + 1e-8))
        return loss.mean()

# 使用
criterion = WeightedBCELoss(weight_pos=9.0)
```

---

### 策略2: 数据重采样

#### a) 过采样（Oversampling）

**原理**: 复制少数类样本

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE

# 方法1: 随机过采样
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 方法2: SMOTE（生成合成样本）
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**优点**:
- ✅ 增加少数类样本数量
- ✅ SMOTE可以生成多样性

**缺点**:
- ⚠️ 可能过拟合少数类
- ⚠️ 增加训练时间

#### b) 欠采样（Undersampling）

**原理**: 减少多数类样本

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

**优点**:
- ✅ 平衡数据集
- ✅ 减少训练时间

**缺点**:
- ⚠️ 丢失多数类信息

---

### 策略3: 调整决策阈值

**原理**: 不使用默认的0.5阈值，根据验证集调整。

```python
from sklearn.metrics import precision_recall_curve

# 在验证集上找最佳阈值
precisions, recalls, thresholds = precision_recall_curve(y_val, pred_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"最佳阈值: {best_threshold:.4f}")

# 使用最佳阈值预测
y_pred = (pred_probs >= best_threshold).astype(int)
```

---

### 策略4: 集成方法

**原理**: 训练多个模型处理不同的数据子集。

```python
# BalancedBaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

model = BalancedBaggingClassifier(
    base_estimator=your_model,
    n_estimators=10,
    random_state=42
)
```

---

## 📊 评估指标选择

### ❌ 不推荐的指标

| 指标 | 问题 |
|------|------|
| **准确率** | 被多数类主导，容易误导 |

### ✅ 推荐的指标

#### 1. F1分数（最重要）

```python
from sklearn.metrics import f1_score

# Macro F1: 每个类别F1的平均（给少数类更多权重）
f1_macro = f1_score(y_true, y_pred, average='macro')

# Weighted F1: 按样本数加权
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

**推荐使用 Macro F1**，对不平衡数据更敏感。

#### 2. 精确率和召回率

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred)  # 正确预测的正样本比例
recall = recall_score(y_true, y_pred)        # 正样本被正确识别的比例
```

**少数类的召回率**特别重要！

#### 3. ROC-AUC

```python
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, pred_probs)
```

**优点**: 对不平衡相对鲁棒

#### 4. PR-AUC（推荐）

```python
from sklearn.metrics import average_precision_score

pr_auc = average_precision_score(y_true, pred_probs)
```

**优点**: 比ROC-AUC更适合不平衡数据

---

## 🚀 实战：修改你的训练脚本

### 步骤1: 检查数据分布

```bash
python check_class_distribution.py /public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap/id_prop.csv
```

**记下输出的 `pos_weight` 值！**

### 步骤2: 使用加权损失训练

假设 `pos_weight = 9.0`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 128 \
    --epochs 100 \
    --classification 1 \
    --pos_weight 9.0 \
    --use_fine_grained_attention True \
    --use_only_graph_for_prediction True \
    --output_dir ./output_classification_balanced \
    --random_seed 42
```

### 步骤3: 使用正确的评估指标

训练完成后，评估时关注：

```python
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 详细报告（包含每个类别的precision/recall/f1）
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# ROC-AUC
print(f"ROC-AUC: {roc_auc_score(y_test, pred_probs):.4f}")

# PR-AUC（推荐）
print(f"PR-AUC: {average_precision_score(y_test, pred_probs):.4f}")
```

---

## 📈 不平衡严重程度评估

| 不平衡比率 | 严重程度 | 必须采取的措施 |
|-----------|---------|--------------|
| < 3:1 | 🟢 轻度 | 可选：类别权重 |
| 3:1 - 10:1 | 🟡 中度 | **必须**：类别权重或重采样 |
| > 10:1 | 🔴 严重 | **必须**：类别权重 + 重采样 + 调整阈值 |

---

## 💡 最佳实践总结

### 训练集不平衡

1. ✅ **使用加权损失函数**（pos_weight）
2. ✅ 考虑过采样（SMOTE）
3. ✅ 使用Focal Loss（对难分样本加权）
4. ⚠️ 避免欠采样（除非数据量很大）

### 验证集/测试集不平衡

1. ✅ **使用F1分数、ROC-AUC、PR-AUC评估**
2. ✅ 分别报告每个类别的精确率和召回率
3. ✅ 在验证集上调整决策阈值
4. ❌ **不要只看准确率**

### 综合策略

```
训练时: 加权损失 + 过采样（SMOTE）
评估时: F1-macro + PR-AUC + 混淆矩阵
调优时: 网格搜索最佳阈值
```

---

## 🔧 代码示例

### 完整的不平衡分类流程

```python
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import numpy as np

# 1. 计算类别权重
def compute_pos_weight(y_train):
    """计算BCEWithLogitsLoss的pos_weight"""
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = num_neg / num_pos
    return pos_weight

# 2. 创建加权损失
pos_weight = compute_pos_weight(y_train)
print(f"Pos weight: {pos_weight:.4f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

# 3. 训练（示例）
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        logits = model(batch)
        loss = criterion(logits, batch_labels)
        # ... backward and optimize

# 4. 评估
model.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        logits = model(batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# 5. 报告结果
print("\n分类报告:")
print(classification_report(all_labels, all_preds,
                          target_names=['Class 0', 'Class 1']))

print(f"\nROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
print(f"PR-AUC: {average_precision_score(all_labels, all_probs):.4f}")

# 6. 混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

---

## 📚 参考资料

### 学术论文
- **Focal Loss**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **SMOTE**: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)

### 工具库
- **imbalanced-learn**: https://imbalanced-learn.org/
- **PyTorch 加权损失**: https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss

---

## 🎯 快速决策树

```
测试集类别不平衡？
    ↓
    ├─ 是 → 计算不平衡比率
    │       ↓
    │       ├─ < 3:1 → 可以只用F1分数评估
    │       ├─ 3:1 - 10:1 → 使用pos_weight + F1/PR-AUC评估
    │       └─ > 10:1 → pos_weight + 重采样 + 阈值调整 + PR-AUC
    │
    └─ 否 → 使用标准流程（准确率可信）
```

---

## ⚙️ 下一步行动

1. **检查数据分布**:
   ```bash
   python check_class_distribution.py your_data.csv
   ```

2. **选择策略** (根据不平衡比率)

3. **修改训练脚本** (添加pos_weight)

4. **使用正确的评估指标** (F1-macro, PR-AUC)

5. **对比实验** (有权重 vs 无权重)

---

需要我帮你修改训练脚本来支持类别权重吗？
