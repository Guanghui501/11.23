# Masking vs Deletion: 关键区别

## 🎯 核心问题

你发现100%遮挡后两个模型的MAE不同：
- model1+2 (无中期融合): MAE = 0.5358
- SAGE-Net (有中期融合): MAE = 0.7470

**你的疑问**: 如果文本完全遮挡了，为什么不一样？

**答案**: 因为"遮挡"不等于"删除"！

---

## 📊 方法对比

### 方法1: **遮挡 (Masking)** - 当前使用的方法

```python
原始文本: "Copper oxide cubic structure with space group Fm-3m"

100%遮挡后: "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
```

**关键特点**:
- ✗ 文本**不是空的**，而是变成了一串 `[MASK]` 标记
- ✗ MatSciBERT 仍会处理这些 `[MASK]` 标记
- ✗ `[MASK]` 的嵌入是在预训练时学到的，**不是零向量**
- ✗ 这些嵌入代表"平均材料科学文本"，仍包含信息

**问题**:
```python
# SAGE-Net的固定中期融合
middle_feat = 0.5 * graph_feat + 0.5 * text_feat
#                                      ↑
#                                这不是零！是[MASK]嵌入！

# 结果：图特征被污染 → MAE 0.7470
```

---

### 方法2: **删除 (Deletion)** - 你想要的方法

```python
原始文本: "Copper oxide cubic structure with space group Fm-3m"

100%删除后: ""  # 空字符串！
50%删除后: "Copper cubic structure Fm-3m"  # 只保留部分tokens
```

**关键特点**:
- ✓ 文本**真的是空的**
- ✓ MatSciBERT 处理空字符串时产生默认嵌入
- ✓ 可以测试**真正的**纯图特征性能
- ✓ 没有噪声嵌入混入图特征

**预期**:
```python
# SAGE-Net的固定中期融合
middle_feat = 0.5 * graph_feat + 0.5 * empty_text_feat
#                                      ↑
#                                   空文本的默认嵌入

# 预期：比[MASK]嵌入干净 → MAE应该会更好！
```

---

## 🔬 为什么删除应该更好？

### 假设 1: `[MASK]` 嵌入是有害的

如果在100%删除时MAE比100%遮挡更好：
```
SAGE-Net (有中期融合):
- 100%遮挡 (文本='[MASK] [MASK] ...'): MAE = 0.7470
- 100%删除 (文本=''):                 MAE = ??? (应该更低!)
```

**这证明**:
- `[MASK]` 嵌入包含噪声信息
- 固定中期融合强制将这些噪声混入图特征
- 这就是为什么SAGE-Net在100%遮挡时崩溃！

### 假设 2: 空文本的处理方式不同

```python
# MatSciBERT处理空字符串
text = ""
tokens = tokenizer(text)  # 可能得到 [CLS] [SEP] 或 [PAD]
embeddings = model(tokens)  # 得到特殊token的嵌入

# 这些特殊token的嵌入可能比[MASK]嵌入更"中性"
# → 对模型预测的干扰更小
```

---

## 📈 实验设计

### 实验目的

对比**遮挡**和**删除**两种方法，回答：
1. `[MASK]` 嵌入是否有害？
2. 空文本的表现如何？
3. 固定中期融合是否真的在混入噪声？

### 实验步骤

```bash
# 一键运行
chmod +x compare_masking_vs_deletion.sh
./compare_masking_vs_deletion.sh
```

**执行流程**:
1. 生成**遮挡**数据集（tokens → `[MASK]`）
2. 生成**删除**数据集（直接移除tokens）
3. 用两个模型评估**遮挡**数据集
4. 用两个模型评估**删除**数据集
5. 对比结果

### 预期结果

#### 场景 A: 删除比遮挡更好

```
Strategy: random_token, 100%

              遮挡方法                     删除方法
              (文本='[MASK] ...')         (文本='')
─────────────────────────────────────────────────────────────
model1+2      0.5358                      0.5000  ✓ 更好!
SAGE-Net      0.7470                      0.6500  ✓ 更好!
```

**解释**: `[MASK]` 嵌入是有害的！删除更干净。

**意义**:
- 证明固定中期融合混入了噪声嵌入
- 证明需要自适应融合（Gated Cross-Attention）

---

#### 场景 B: 遮挡比删除更好

```
Strategy: random_token, 100%

              遮挡方法                     删除方法
              (文本='[MASK] ...')         (文本='')
─────────────────────────────────────────────────────────────
model1+2      0.5358  ✓ 更好!             0.5800
SAGE-Net      0.7470  ✓ 更好!             0.8000
```

**解释**: `[MASK]` 嵌入虽然质量不高，但仍比空文本有用。

**意义**:
- `[MASK]` 嵌入包含一些有用的"平均信息"
- 问题不是嵌入本身，而是融合权重无法适应质量变化

---

#### 场景 C: model1+2 vs SAGE-Net 行为不同

```
Strategy: random_token, 100%

              遮挡方法         删除方法         差异
─────────────────────────────────────────────────────────
model1+2      0.5358          0.5000          删除更好
SAGE-Net      0.7470          0.7500          遮挡更好!
```

**解释**:
- model1+2（无中期融合）：删除更好 → 能有效忽略噪声
- SAGE-Net（有中期融合）：遮挡更好 → 至少`[MASK]`比空文本有信息

**意义**:
- 固定中期融合无法处理空文本
- 需要质量感知的自适应机制

---

## 💡 关键洞察

### 遮挡方法的问题

```
原始文本 → [MASK] [MASK] [MASK] → MatSciBERT → [MASK]嵌入
                                                    ↓
                                           固定中期融合 (0.5 * graph + 0.5 * text)
                                                    ↓
                                           污染的特征 → 差的预测
```

**问题**: `[MASK]` 嵌入不是"无信息"，而是"平均信息" + "噪声"

### 删除方法的优势

```
原始文本 → "" (删除) → MatSciBERT → 空文本嵌入 (更中性)
                                        ↓
                               固定中期融合 (0.5 * graph + 0.5 * empty)
                                        ↓
                               更干净的特征 → 更好的预测 (期望)
```

**优势**: 空文本的嵌入更"中性"，干扰更小

---

## 🎯 对你研究的意义

### 如果删除更好 → 强化你的论文故事

1. **问题定义更清晰**:
   ```
   "我们发现固定中期融合会混入有害的[MASK]嵌入，
    导致100%遮挡时性能下降192%。
    当我们改用删除方法（真正的空文本）时，
    性能提升了XX%，证明问题出在噪声嵌入的强制混合。"
   ```

2. **解决方案更有针对性**:
   ```
   "因此，我们提出Gated Cross-Attention，
    它可以检测文本质量，并在遇到低质量文本时
    自动降低融合权重，避免污染图特征。"
   ```

3. **预期结果**:
   ```
   Gated Cross-Attention应该：
   - 在干净文本时：高融合权重 → MAE ≈ 0.2550
   - 在[MASK]文本时：低融合权重 → MAE ≈ 0.5000 (接近删除方法)
   - 在空文本时：极低权重 → MAE ≈ 0.5000
   ```

---

## 📝 使用方法

### 快速开始

```bash
# 1. 使脚本可执行
chmod +x compare_masking_vs_deletion.sh
chmod +x pregenerate_deletion_dataset.py
chmod +x compare_masking_deletion_results.py

# 2. 一键运行完整对比
./compare_masking_vs_deletion.sh

# 3. 查看结果
cat masking_vs_deletion_comparison/summary.txt
```

### 手动运行

```bash
# 步骤1: 生成删除数据集
python pregenerate_deletion_dataset.py \
    --input_data ./corrected_test_set/test.pkl \
    --output_dir ./deletion_datasets \
    --strategies random_token random_word keep_keywords \
    --ratios 0.0 0.5 1.0 \
    --seed 42

# 步骤2: 评估模型（使用删除数据）
python evaluate_with_premasked_data.py \
    --checkpoint ./checkpoints/model.pt \
    --masked_data ./deletion_datasets/random_token_1.0.pkl \
    --output ./results_deletion.json

# 步骤3: 对比遮挡vs删除结果
python compare_masking_deletion_results.py \
    --masking_model1 "./results_masking_model1_*.json" \
    --masking_model2 "./results_masking_model2_*.json" \
    --deletion_model1 "./results_deletion_model1_*.json" \
    --deletion_model2 "./results_deletion_model2_*.json" \
    --output_dir ./comparison_plots
```

---

## 🔍 诊断检查

### 验证删除确实产生空字符串

```python
from pregenerate_deletion_dataset import DeterministicTextDeleter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
deleter = DeterministicTextDeleter(tokenizer, strategy='random_token')

text = "Copper oxide cubic structure"
print(f"原始: {text}")

# 50%删除
deleted_50 = deleter.delete_text_deterministic(text, 0.5, "sample_1")
print(f"50%删除: '{deleted_50}'")

# 100%删除
deleted_100 = deleter.delete_text_deterministic(text, 1.0, "sample_1")
print(f"100%删除: '{deleted_100}'")
print(f"是否为空: {deleted_100 == ''}")
```

预期输出:
```
原始: Copper oxide cubic structure
50%删除: 'Copper structure'
100%删除: ''
是否为空: True
```

---

## 📊 预期图表

运行后会生成以下图表（每个策略一组）：

### 图1: Model1 遮挡vs删除对比
- X轴: 删除/遮挡比率 (0-100%)
- Y轴: MAE
- 两条线: 遮挡方法 vs 删除方法

### 图2: Model2 遮挡vs删除对比
- 同上，但针对SAGE-Net

### 图3: 100%时的柱状图对比
- 直观对比100%遮挡 vs 100%删除的MAE

### 图4: 删除方法的改进百分比
- 显示删除相比遮挡的改进
- 正值 = 删除更好
- 负值 = 遮挡更好

---

## 🎯 Bottom Line

### 问题
"为什么100%遮挡后MAE不一样？"

### 答案的两层
1. **表面原因**: 100%遮挡产生`[MASK]`标记，不是空字符串
2. **深层原因**: 固定中期融合强制混合噪声`[MASK]`嵌入和图特征

### 解决方案
- **短期**: 使用删除方法测试真正的图特征性能
- **长期**: 实现Gated Cross-Attention，自适应调整融合权重

### 实验价值
通过对比遮挡vs删除：
- ✓ 量化`[MASK]`嵌入的影响
- ✓ 验证固定融合的问题
- ✓ 为Gated Cross-Attention提供实证支持

**这个实验将为你的论文提供强有力的实证证据！** 🚀
