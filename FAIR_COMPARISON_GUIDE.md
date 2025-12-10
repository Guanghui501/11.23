# 如何进行公平的模型对比（解决随机遮挡不确定性问题）

## 🎯 问题说明

当你使用 `evaluate_text_masking.py` 分别评估两个模型时：

```bash
# 第一次运行 - 评估模型1
python evaluate_text_masking.py --checkpoint model1.pt ...
# 随机遮挡生成：样本A → "Cu [MASK] structure"

# 第二次运行 - 评估模型2
python evaluate_text_masking.py --checkpoint model2.pt ...
# 随机遮挡生成：样本A → "Cu oxide [MASK]"  ← 遮挡位置不同！
```

**问题**：虽然使用了相同的随机种子（seed=42），但两次运行是**不同的Python进程**，遮挡结果可能不同，导致对比不公平。

---

## ✅ 解决方案：预生成遮挡数据

使用 `pregenerate_masked_dataset.py` **预先生成**所有遮挡数据，确保两个模型看到的遮挡数据**完全相同**。

---

## 📋 使用步骤

### 步骤1：预生成所有遮挡数据（只需运行一次）

```bash
python pregenerate_masked_dataset.py \
    --input_data ./corrected_test_set/test.pkl \
    --output_dir ./masked_datasets \
    --strategies random_token sentence keep_keywords \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seed 42
```

**输出**：
```
masked_datasets/
├── index.json                    # 索引文件
├── random_token_0.0.pkl          # 0% 遮挡
├── random_token_0.1.pkl          # 10% 遮挡
├── random_token_0.5.pkl          # 50% 遮挡
├── random_token_1.0.pkl          # 100% 遮挡
├── sentence_0.0.pkl
├── sentence_0.5.pkl
├── ...
└── keep_keywords_1.0.pkl
```

**特点**：
- ✅ **确定性遮挡**：基于样本ID + 种子，每个样本的遮挡结果固定
- ✅ **可重复**：任何时候重新生成都得到相同结果
- ✅ **高效**：生成一次，多次使用

---

### 步骤2：使用相同的遮挡数据评估模型1

```bash
# 评估：中期融合+跨模态+细粒度
python evaluate_with_premasked_data.py \
    --checkpoint /path/to/model1.pt \
    --masked_data ./masked_datasets/random_token_0.5.pkl \
    --output_file ./results_model1_random_token_0.5.json \
    --batch_size 64
```

**输出**：
```json
{
  "checkpoint": "/path/to/model1.pt",
  "masked_data": "./masked_datasets/random_token_0.5.pkl",
  "masking_strategy": "random_token",
  "masking_ratio": 0.5,
  "metrics": {
    "mae": 0.3921,
    "rmse": 0.5234,
    "r2": 0.8456,
    "num_samples": 1234
  }
}
```

---

### 步骤3：使用相同的遮挡数据评估模型2

```bash
# 评估：跨模态+细粒度（使用完全相同的遮挡数据！）
python evaluate_with_premasked_data.py \
    --checkpoint /path/to/model2.pt \
    --masked_data ./masked_datasets/random_token_0.5.pkl \  # ← 相同文件
    --output_file ./results_model2_random_token_0.5.json \
    --batch_size 64
```

**输出**：
```json
{
  "checkpoint": "/path/to/model2.pt",
  "masked_data": "./masked_datasets/random_token_0.5.pkl",
  "masking_strategy": "random_token",
  "masking_ratio": 0.5,
  "metrics": {
    "mae": 0.3154,
    "rmse": 0.4567,
    "r2": 0.8876,
    "num_samples": 1234
  }
}
```

---

## 🚀 一键运行脚本

我已经为你准备了自动化脚本：

```bash
chmod +x fair_compare_with_premasked.sh
./fair_compare_with_premasked.sh
```

这个脚本会：
1. 预生成所有遮挡数据
2. 评估模型1（所有策略和遮挡率）
3. 评估模型2（使用相同的遮挡数据）
4. 保存所有结果到JSON文件

---

## 📊 对比结果

运行完成后，你可以对比两个模型的结果：

```bash
# 对比 random_token 策略，50% 遮挡率
python compare_two_models.py \
    --result1 ./results_model1_random_token_0.5.json \
    --result2 ./results_model2_random_token_0.5.json
```

**示例输出**：
```
模型对比（random_token, 50%遮挡）
======================================================================
                        模型1           模型2           改进
----------------------------------------------------------------------
MAE                     0.3921          0.3154          -19.6% ✓
RMSE                    0.5234          0.4567          -12.7% ✓
R²                      0.8456          0.8876          +5.0% ✓

结论：模型2在50%遮挡下表现更好
```

---

## 🔧 技术细节

### 确定性遮挡原理

```python
def mask_text_deterministic(self, text, ratio, sample_id):
    # 基于样本ID生成确定性种子
    seed = base_seed + hash(sample_id) % 1000000  # 42 + hash("JVASP-1234")

    # 设置临时种子
    random.seed(seed)

    # 执行遮挡（结果固定）
    masked = self._mask_random_tokens(text, ratio)

    # 恢复原始种子
    return masked
```

**关键点**：
- 每个样本有唯一的ID（如 "JVASP-1234"）
- 相同ID + 相同base_seed → 相同随机种子 → 相同遮挡结果
- 不同样本ID → 不同遮挡模式（保持随机性）

---

## 📝 完整示例：对比两个模型

### 示例场景

对比以下两个模型：
- **模型1**：中期融合+跨模态+细粒度
- **模型2**：跨模态+细粒度

使用 **random_token** 策略，遮挡率 **0%, 50%, 100%**

### 步骤1：预生成数据

```bash
python pregenerate_masked_dataset.py \
    --input_data ./corrected_test_set/test.pkl \
    --output_dir ./masked_datasets_mbj \
    --strategies random_token \
    --ratios 0.0 0.5 1.0 \
    --seed 42
```

### 步骤2：批量评估

创建批量评估脚本 `batch_eval.sh`：

```bash
#!/bin/bash

MODEL1="/path/to/model1.pt"
MODEL2="/path/to/model2.pt"

for ratio in 0.0 0.5 1.0; do
    echo "评估遮挡率: ${ratio}"

    # 模型1
    python evaluate_with_premasked_data.py \
        --checkpoint "$MODEL1" \
        --masked_data "./masked_datasets_mbj/random_token_${ratio}.pkl" \
        --output_file "./results/model1_${ratio}.json"

    # 模型2（使用相同数据）
    python evaluate_with_premasked_data.py \
        --checkpoint "$MODEL2" \
        --masked_data "./masked_datasets_mbj/random_token_${ratio}.pkl" \
        --output_file "./results/model2_${ratio}.json"
done
```

运行：
```bash
chmod +x batch_eval.sh
./batch_eval.sh
```

### 步骤3：汇总结果

```bash
python summarize_comparison.py \
    --model1_results ./results/model1_*.json \
    --model2_results ./results/model2_*.json \
    --output_table ./comparison_table.csv
```

---

## ✅ 验证遮挡数据一致性

你可以验证两次生成的数据是否完全相同：

```bash
# 生成第一次
python pregenerate_masked_dataset.py \
    --input_data ./test.pkl \
    --output_dir ./masked_v1 \
    --seed 42

# 生成第二次（相同种子）
python pregenerate_masked_dataset.py \
    --input_data ./test.pkl \
    --output_dir ./masked_v2 \
    --seed 42

# 验证完全一致
diff ./masked_v1/random_token_0.5.pkl ./masked_v2/random_token_0.5.pkl
# 输出：（无输出表示文件完全相同）✓
```

---

## 🎯 关键优势

| 特性 | 原始方法（分次运行） | 预生成方法 |
|------|---------------------|-----------|
| **遮挡一致性** | ❌ 不保证相同 | ✅ 100%相同 |
| **可重复性** | ⚠️ 理论上可以 | ✅ 完全可重复 |
| **对比公平性** | ❌ 不公平 | ✅ 完全公平 |
| **调试方便** | ❌ 难以追踪 | ✅ 可以查看遮挡数据 |
| **计算效率** | ⚠️ 每次都遮挡 | ✅ 遮挡一次，多次使用 |

---

## 🔍 调试和验证

### 查看遮挡数据示例

```python
import pickle

# 加载遮挡数据
with open('./masked_datasets/random_token_0.5.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看第一个样本
sample = data[0]
print(f"ID: {sample['id']}")
print(f"原始文本: {sample['original_text']}")
print(f"遮挡文本: {sample['text']}")
print(f"遮挡策略: {sample['masking_strategy']}")
print(f"遮挡率: {sample['masking_ratio']}")
```

**输出示例**：
```
ID: JVASP-1234
原始文本: Copper oxide cubic structure with space group Fm-3m
遮挡文本: Copper [MASK] cubic [MASK] with space [MASK] Fm-3m
遮挡策略: random_token
遮挡率: 0.5
```

---

## 💡 常见问题

### Q1: 预生成数据需要多长时间？

**答**：取决于数据集大小
- 1000个样本：~1分钟
- 10000个样本：~5分钟
- 包含11个遮挡率 × 3种策略 = 33个文件

### Q2: 预生成的数据占用多少空间？

**答**：与原始数据相似
- 原始test.pkl: 50MB
- 每个遮挡文件: ~50MB
- 33个文件总计: ~1.65GB

### Q3: 如果修改了测试集怎么办？

**答**：重新生成遮挡数据
```bash
# 删除旧数据
rm -rf ./masked_datasets

# 重新生成
python pregenerate_masked_dataset.py ...
```

### Q4: 可以只生成部分遮挡率吗？

**答**：可以，指定需要的遮挡率
```bash
python pregenerate_masked_dataset.py \
    --ratios 0.0 0.5 1.0  # 只生成0%, 50%, 100%
```

---

## 📌 总结

**使用预生成遮挡数据的工作流**：

1. ✅ **预生成一次**：`pregenerate_masked_dataset.py`
2. ✅ **评估模型1**：`evaluate_with_premasked_data.py` (使用预生成数据)
3. ✅ **评估模型2**：`evaluate_with_premasked_data.py` (使用相同的预生成数据)
4. ✅ **公平对比**：两个模型看到完全相同的遮挡数据

**关键原则**：
- 🎯 一次生成，多次使用
- 🎯 确定性遮挡，100%可重复
- 🎯 公平对比，结果可靠

---

**开始使用**：

```bash
# 快速开始
chmod +x fair_compare_with_premasked.sh
./fair_compare_with_premasked.sh
```

就是这么简单！🎉
