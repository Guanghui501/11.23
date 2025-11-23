# Attention Weight 可视化问题诊断指南

## 问题描述

所有原子（Ba_0, Ba_1, Ba_2, Ba_3, Hf_4, Li_5）显示相同的 Top Words，这不符合预期。

## 已执行的修改总结

### 1. 核心修改：从 `.max()` 改为 `.mean()` (Commit b35d3ba)

**位置**：`interpretability_enhanced.py` 中的 `_merge_tokens_and_weights()` 方法

**原因**：`.max()` 会导致具有更多 WordPiece 子token的词（如 "liba4hf" → ["li", "##ba", "##4", "##hf"]）在归一化后主导注意力权重。

**变更**：
```python
# 之前 (使用 max)
merged_weights = np.array([weights[indices].max() for indices in token_mapping])

# 现在 (使用 mean)
merged_weights = np.array([weights[indices].mean() for indices in token_mapping])
```

### 2. 可视化改进 (Commits 00d120e, c996b48)

- 移除了表格显示
- 将文本信息集成到柱状图内部
- 简化布局从3行到2行

### 3. Bug修复

- **文本分词** (Commits 0c6ac8f, ab68607): 修复了文本tokenization的类型处理
- **文本到句子映射** (Commit 321b93e): 改进了语义区域分析中的算法

## 代码库结构问题

⚠️ **重要发现**：发现两个版本的 `interpretability_enhanced.py`：

1. **`/home/user/11.23/interpretability_enhanced.py`** (2127行, 87KB) - 我们修改的版本
2. **`/home/user/11.23/models/interpretability_enhanced.py`** (914行, 32KB) - 旧版本

### 确认使用的版本

运行以下命令确认导入的是哪个文件：
```python
import interpretability_enhanced
print(interpretability_enhanced.__file__)
```

如果导入的是 `models/` 下的文件，那么我们的修改不会生效。

## 诊断步骤

### 步骤 1: 运行测试脚本

```bash
python3 test_attention_merge.py
```

此脚本会验证：
- 基本的注意力权重处理逻辑
- WordPiece token 合并是否保留了每个原子的差异

### 步骤 2: 禁用 WordPiece 合并和停用词过滤

在您的demo脚本中，尝试：

```python
analysis = analyzer.visualize_fine_grained_attention(
    attention_weights=fg_attn,
    atoms_object=atoms_object,
    text_tokens=tokens,
    save_path=save_path,
    top_k_atoms=10,
    top_k_words=15,
    show_all_heads=False,
    merge_wordpiece=False,  # 禁用合并
    filter_stopwords=False   # 禁用停用词过滤
)
```

### 步骤 3: 检查原始注意力权重

在模型输出后添加诊断代码：

```python
# 在获取 fg_attn 之后
atom_to_text = fg_attn['atom_to_text']  # [batch, heads, num_atoms, seq_len]
print(f"原始注意力形状: {atom_to_text.shape}")

# 取第一个batch，对heads求平均
atom_to_text_avg = atom_to_text[0].mean(dim=0).cpu().numpy()  # [num_atoms, seq_len]

# 检查前3个原子的注意力分布是否不同
for i in range(min(3, atom_to_text_avg.shape[0])):
    top_5_indices = atom_to_text_avg[i].argsort()[-5:][::-1]
    print(f"Atom {i} top 5 token indices: {top_5_indices}")
    print(f"Atom {i} top 5 weights: {atom_to_text_avg[i, top_5_indices]}")

# 检查是否所有原子相同
if np.allclose(atom_to_text_avg[0], atom_to_text_avg[1]):
    print("⚠️  警告：原子0和原子1的注意力完全相同！")
else:
    print("✅ 原子0和原子1的注意力不同")
```

### 步骤 4: 回退到原始版本测试

```bash
# 回退到最初的上传版本
git checkout 87191ba -- interpretability_enhanced.py

# 运行您的分析
python demo_fine_grained_attention.py --model_path ... --cif_path ... --text "..."

# 如果仍然有问题，说明问题不在 interpretability_enhanced.py
```

## 可能的根本原因

### 1. 模型本身的问题

**症状**：即使回退代码也无法复现原来的结果

**可能原因**：
- Checkpoint 文件已更改
- 模型的 fine-grained attention 没有正确训练
- 所有注意力头学到了相同的模式

**验证方法**：
```python
# 检查模型是否有 fine-grained attention
print(f"use_fine_grained_attention: {model.use_fine_grained_attention}")

# 检查不同 attention head 是否有差异
for head in range(num_heads):
    head_attn = atom_to_text[0, head]  # [num_atoms, seq_len]
    print(f"Head {head} entropy: {scipy.stats.entropy(head_attn.flatten()):.4f}")
```

### 2. 数据输入问题

**症状**：文本 tokenization 结果异常

**验证方法**：
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('m3rg-iitd/matscibert')

text = "您的材料描述文本"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
```

### 3. Python 缓存问题

**解决方法**：
```bash
# 清除所有 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 重新运行
python demo_fine_grained_attention.py ...
```

### 4. 导入路径问题

**症状**：修改代码后没有效果

**验证方法**：
```python
import sys
print("Python path:")
for p in sys.path:
    print(f"  {p}")

import interpretability_enhanced
print(f"\nLoaded from: {interpretability_enhanced.__file__}")
```

## 当前代码状态

### 最新改动 (Commit d0b7907)

1. ✅ 移除了所有 debug 打印输出
2. ✅ 在 `text_to_atom` 合并中统一使用 `.mean()` (之前误用了 `.max()`)
3. ✅ 清理了 Python 缓存文件
4. ✅ 添加了测试脚本 `test_attention_merge.py`

### 核心逻辑

**WordPiece 合并** (`_merge_tokens_and_weights`):
- 对每个原子独立合并 sub-tokens
- 使用 `.mean()` 保持注意力分布
- Shape: `[num_atoms, seq_len]` → `[num_atoms, merged_seq_len]`

**Top Words 分析** (`visualize_fine_grained_attention`):
```python
for i, element in enumerate(elements):
    all_words = [(text_tokens[idx], float(atom_to_text_avg[i, idx]))
                for idx in range(len(text_tokens[:seq_len]))]
    all_words.sort(key=lambda x: x[1], reverse=True)
    # 过滤停用词并返回 top-k
```

## 下一步建议

1. **首先**：运行 `test_attention_merge.py` 验证逻辑正确性
2. **然后**：在demo中添加诊断代码，检查原始注意力权重
3. **最后**：如果问题依然存在，尝试：
   - 禁用 `merge_wordpiece` 和 `filter_stopwords`
   - 检查模型 checkpoint 是否正确
   - 验证文本输入是否正常

## 联系信息

如果问题仍未解决，请提供以下信息：
- `test_attention_merge.py` 的输出
- 原始注意力权重的诊断信息（步骤3）
- 模型配置（`use_fine_grained_attention`, `cross_modal_num_heads` 等）
- 一个具体的输入样例（CIF文件 + 文本描述）
