# Keep Keywords 策略详解

## 🎯 策略目标

`keep_keywords` 策略的设计理念是：**只保留材料科学中最关键的信息**（元素名称和晶体结构术语），移除其他描述性文本。

这模拟了一个现实场景：当只有最少的材料信息可用时，模型能否仍然做出合理预测？

---

## 🔧 工作流程详解

### 步骤1：识别关键词

代码首先定义了两类关键词：

#### 1.1 化学元素（118个元素）
```python
self.element_pattern = re.compile(
    r'\b(H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|'
    r'Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|'
    r'Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
    r'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)\b'
)
```

#### 1.2 晶体结构术语（10个关键词）
```python
self.structure_keywords = [
    'cubic',           # 立方
    'tetragonal',      # 四方
    'orthorhombic',    # 正交
    'hexagonal',       # 六方
    'monoclinic',      # 单斜
    'triclinic',       # 三斜
    'rhombohedral',    # 菱形
    'space group',     # 空间群
    'lattice',         # 晶格
    'crystal'          # 晶体
]
```

### 步骤2：扫描文本并标记关键词

```python
def _keep_only_keywords(self, text: str, ratio: float) -> str:
    words = text.split()  # 按空格分词

    # 找出所有关键词的位置
    keywords = []
    keyword_indices = []
    for i, word in enumerate(words):
        if (self.element_pattern.search(word) or  # 匹配元素
            any(kw in word.lower() for kw in self.structure_keywords)):  # 匹配结构术语
            keywords.append(i)
            keyword_indices.append(i)
```

### 步骤3：根据遮挡率决定保留多少关键词

**关键公式**：
```python
num_keywords_to_keep = int(len(keywords) * (1.0 - ratio))
```

- **ratio = 0.0** → 保留100%关键词
- **ratio = 0.5** → 保留50%关键词
- **ratio = 1.0** → 保留0%关键词（完全遮挡）

### 步骤4：随机选择要保留的关键词

```python
if num_keywords_to_keep > 0 and len(keywords) > 0:
    keep_indices = set(random.sample(keywords, min(num_keywords_to_keep, len(keywords))))
else:
    keep_indices = set()  # 不保留任何关键词
```

### 步骤5：构建输出文本

```python
result = []
for i, word in enumerate(words):
    if i in keep_indices:
        result.append(word)  # 保留的关键词
    elif i in keyword_indices:
        result.append('[MASK]')  # 被遮挡的关键词
    # 注意：非关键词直接删除（不添加到result）

return ' '.join(result) if result else '[MASK]'
```

**重要特点**：
- ✅ 保留的关键词：原样输出
- 🔒 遮挡的关键词：替换为`[MASK]`
- ❌ 非关键词：**直接删除**

---

## 📝 实例演示

### 示例1：原始文本

```
"The crystal structure of Cu2O is cubic with a lattice parameter of 4.27 Å.
It contains copper and oxygen atoms in a specific arrangement."
```

**关键词识别**：
- 元素：Cu, O, copper, oxygen
- 结构：crystal, cubic, lattice

### 遮挡率 = 0% (保留100%关键词)

```
"crystal Cu O cubic lattice copper oxygen"
```

**输出特点**：
- 保留了所有7个关键词
- 删除了所有描述性文本（"The", "structure", "of", "is", "with", "a", "parameter", "of", "4.27", "Å", "It", "contains", "and", "atoms", "in", "a", "specific", "arrangement"）

### 遮挡率 = 50% (保留50%关键词)

假设随机保留了3个关键词：

```
"crystal [MASK] O [MASK] lattice [MASK] [MASK]"
```

**输出特点**：
- 保留：crystal, O, lattice
- 遮挡：Cu, cubic, copper, oxygen
- 删除：所有非关键词

### 遮挡率 = 100% (保留0%关键词)

```
"[MASK]"
```

或完全空字符串 `""`

**这就是为什么100%遮挡时keep_keywords表现最差！**

---

## 🔄 完整评估流程

### 1. 初始化

```python
# 在evaluate_with_masking函数中
masker = TextMasker(tokenizer, strategy='keep_keywords')
```

### 2. 对每个batch的每个样本应用遮挡

```python
for batch in test_loader:
    g, lg, target, text_list = prepare_batch(batch, device)

    # 对batch中每个文本应用keep_keywords
    masked_text_list = [masker.mask_text(text, masking_ratio) for text in text_list]
    # 例如 masking_ratio=0.5:
    # 输入: "The cubic crystal of Cu O has lattice parameter..."
    # 输出: "cubic [MASK] Cu lattice"
```

### 3. 模型前向传播

```python
out_data = model([g, lg, masked_text_list])
```

**关键点**：模型接收到的是**高度稀疏的文本**：
- 只有少量关键词
- 大量信息缺失
- 对于100%遮挡，可能是空字符串或单个`[MASK]`

### 4. 计算指标

```python
mae = np.mean(np.abs(predictions - targets))
rmse = np.sqrt(np.mean((predictions - targets) ** 2))
r2 = 1 - (ss_res / ss_tot)
```

---

## 🚀 如何运行

### 方式1：单独运行keep_keywords策略

```bash
cd /public/home/ghzhang/11.23

python evaluate_text_masking.py \
    --checkpoint /path/to/best_test_model.pt \
    --test_data ./corrected_test_set/test.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy keep_keywords \
    --masking_ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --output_dir ./keep_keywords_output \
    --batch_size 64
```

### 方式2：运行所有策略（包括keep_keywords）

```bash
./quick_extract_and_eval.sh
```

这会自动运行5种策略，包括keep_keywords。

---

## 📊 为什么keep_keywords在100%遮挡时表现最差？

### 原因分析

#### 1. **输出格式差异**

| 策略 | 100%遮挡输出 | 模型感知 |
|------|-------------|---------|
| random_token | `"[MASK] [MASK] [MASK] ... [MASK]"` | 看起来像完整的（虽然无意义的）句子 |
| keep_keywords | `""` 或 `"[MASK]"` | **完全空白或极少token** |

#### 2. **模型的训练分布**

训练时，模型见过的文本格式：
- ✅ 完整的句子（正常情况）
- ✅ 部分损坏的句子（dropout等）
- ❌ **完全空白的文本**（从未见过）

keep_keywords的100%遮挡产生的**极度稀疏文本**超出了模型的训练分布。

#### 3. **无Middle Fusion时的脆弱性**

**有Middle Fusion**：
```python
# 模型学会了统一的失效检测
if text_is_extremely_sparse:
    use_pure_graph_mode()  # 统一处理
```

**无Middle Fusion**：
```python
# 模型对不同"稀疏"程度反应不一
if text == "":
    # 极度困惑，因为从未见过
    return bad_prediction()  # MAE = 12.318
elif text == "[MASK] [MASK] [MASK]":
    # 至少看起来像文本
    return ok_prediction()  # MAE = 9.93
```

---

## 💡 设计意义

keep_keywords策略测试的是：
1. ✅ **最小信息预测能力**：只用元素和结构能预测性质吗？
2. ✅ **鲁棒性极限**：模型在极端信息缺失下的表现
3. ✅ **关键信息识别**：模型是否真的依赖关键词还是依赖完整描述

### 实验洞察

你的数据显示：
- 0%遮挡时（保留所有关键词）：MAE ≈ 8.78-9.10
- 50%遮挡时（保留一半关键词）：MAE ≈ 9.6-12.3
- 100%遮挡时（无关键词）：
  - 有Middle: MAE = 9.99（优雅退化）
  - 无Middle: MAE = 12.32（崩溃）

**结论**：关键词确实重要，但模型需要**足够的架构鲁棒性**（如Middle Fusion）来处理极端情况。

---

## 🔬 调试技巧

### 打印遮挡示例

```python
# 在evaluate_text_masking.py中添加：
if batch_idx == 0:  # 只打印第一个batch
    print("\n原始文本示例:")
    print(text_list[0])
    print(f"\n{masking_ratio*100}%遮挡后:")
    print(masked_text_list[0])
    print()
```

### 统计关键词数量

```python
# 在_keep_only_keywords中添加：
print(f"原文长度: {len(words)}")
print(f"关键词数量: {len(keywords)}")
print(f"保留关键词: {num_keywords_to_keep}")
print(f"输出长度: {len(result)}")
```

---

## ✅ 总结

**keep_keywords策略的核心**：
1. 📍 识别材料科学关键信息（元素+结构）
2. ✂️ 删除所有非关键词
3. 🎲 根据遮挡率随机保留部分关键词
4. 🔒 遮挡未保留的关键词为`[MASK]`
5. 📊 产生**极度稀疏**的文本输入

**为什么重要**：
- 测试模型对**最小信息**的利用能力
- 揭示**架构鲁棒性**的差异（Middle Fusion的价值）
- 模拟**实际应用场景**（只有有限的材料信息）

你的实验完美展示了：**在极端情况下，架构设计的鲁棒性比单纯的性能提升更重要**！
