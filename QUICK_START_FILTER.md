# filter_global_information.py 快速开始

## 🚀 最简单的用法（3步）

### 步骤 1: 导入函数

```python
from filter_descriptions_simple import remove_local_information
```

### 步骤 2: 准备描述

```python
description = "LiBa4Hf crystallizes in the cubic F-43m space group. All Ba(1)-Hf(1) bond lengths are 4.25 Å."
```

### 步骤 3: 过滤

```python
filtered = remove_local_information(description, mode='aggressive')
print(filtered)
# 输出: "LiBa4Hf crystallizes in the cubic F-43m space group."
```

✅ **完成！键长 "4.25 Å" 已被去除**

---

## 📋 三种使用场景

### 场景 A: 处理单个描述（最常用）

```python
from filter_descriptions_simple import remove_local_information

# 您的材料描述
desc = """Ba(1) is bonded to six equivalent Ba(1) atoms.
There are three shorter (3.60 Å) and three longer (3.66 Å) bond lengths."""

# 过滤
filtered = remove_local_information(desc, mode='aggressive')

print("原始:", desc)
print("过滤:", filtered)
```

**输出**:
```
原始: Ba(1) is bonded to six equivalent Ba(1) atoms. There are three shorter (3.60 Å) and three longer (3.66 Å) bond lengths.
过滤: Ba(1) is bonded to six equivalent Ba(1) atoms.
```

---

### 场景 B: 提取全局摘要

```python
from filter_descriptions_simple import extract_global_summary

desc = "AlAs is Zincblende structured and crystallizes in the cubic F-43m space group."

summary = extract_global_summary(desc)
print(summary)
# 输出: "AlAs has Zincblende structure crystallizes in cubic system space group F-43m."
```

---

### 场景 C: 批量处理列表

```python
from filter_descriptions_simple import remove_local_information

descriptions = [
    "Material 1 description with bond length 2.48 Å...",
    "Material 2 description with bond length 3.21 Å...",
    "Material 3 description with bond length 4.25 Å..."
]

# 批量过滤
filtered_list = [
    remove_local_information(d, mode='aggressive')
    for d in descriptions
]

for i, (orig, filt) in enumerate(zip(descriptions, filtered_list)):
    print(f"{i+1}. 原始: {len(orig)} 字符 → 过滤: {len(filt)} 字符")
```

---

## ⚙️ 三种过滤模式

### 1. Aggressive（激进 - 推荐）

```python
filtered = remove_local_information(desc, mode='aggressive')
```

**去除**: 所有包含键长、键角的句子

**示例**:
```
原始: "All Ba(1)-Hf(1) bond lengths are 4.25 Å. The angles are 90°."
过滤: ""  (整句删除)
```

**适用**: 注意力可解释性分析

---

### 2. Moderate（中等）

```python
filtered = remove_local_information(desc, mode='moderate')
```

**去除**: 键长键角句子，保留配位描述

**示例**:
```
原始: "Li(1) is bonded in 12-coordinate geometry. Bond lengths are 4.31 Å."
过滤: "Li(1) is bonded in 12-coordinate geometry."
```

**适用**: 保留更多结构信息

---

### 3. Conservative（保守）

```python
filtered = remove_local_information(desc, mode='conservative')
```

**去除**: 只替换数值为 X

**示例**:
```
原始: "All Ba(1)-Hf(1) bond lengths are 4.25 Å."
过滤: "All Ba(1)-Hf(1) bond lengths are X."
```

**适用**: 保持句子完整性

---

## 💻 实际应用示例

### 示例 1: 在注意力分析中使用

```python
from filter_descriptions_simple import remove_local_information
from demo_robust_attention import run_complete_analysis

# 准备描述
original_description = "LiBa4Hf crystallizes... bond lengths are 4.25 Å..."

# 过滤描述
filtered_description = remove_local_information(
    original_description,
    mode='aggressive'
)

# 使用过滤后的描述进行分析
results = run_complete_analysis(
    model=model,
    g=g,
    lg=lg,
    text=filtered_description,  # 使用过滤后的描述
    atoms_object=atoms,
    save_dir='./results'
)
```

---

### 示例 2: 在数据预处理中使用

```python
from filter_descriptions_simple import remove_local_information

def preprocess_materials_data(data_list):
    """
    预处理材料数据
    """
    processed_data = []

    for item in data_list:
        # 过滤描述
        filtered_desc = remove_local_information(
            item['description'],
            mode='aggressive'
        )

        processed_data.append({
            'formula': item['formula'],
            'structure': item['structure'],
            'description': filtered_desc  # 使用过滤后的描述
        })

    return processed_data

# 使用
data = [
    {'formula': 'LiBa4Hf', 'description': 'LiBa4Hf crystallizes...', ...},
    {'formula': 'AlAs', 'description': 'AlAs is Zincblende...', ...}
]

processed = preprocess_materials_data(data)
```

---

### 示例 3: 命令行快速测试

```bash
# 运行演示脚本
python demo_filter_usage.py

# 查看所有示例和对比
```

---

## 📊 处理CSV文件（需要pandas）

如果您有 pandas，可以处理整个CSV文件：

```python
from filter_global_information import process_descriptions

# 处理CSV文件
df = process_descriptions(
    csv_file='your_materials.csv',
    output_file='your_materials_filtered.csv',
    mode='aggressive',
    include_global_summary=True
)

# 输出文件包含：
# - description: 原始描述
# - description_filtered: 过滤后描述
# - global_summary: 全局摘要
```

**输入CSV格式**:
```csv
id,formula,bandgap,description,source
1,LiBa4Hf,0.0,"LiBa4Hf crystallizes... bond lengths are 4.25 Å.",file.csv
2,AlAs,2.276,"AlAs is Zincblende... bond lengths are 2.48 Å.",file.csv
```

**输出CSV格式**:
```csv
id,formula,bandgap,description,source,description_filtered,global_summary
1,LiBa4Hf,0.0,"LiBa4Hf crystallizes...","LiBa4Hf crystallizes... (无键长)","LiBa4Hf crystallizes in cubic system..."
2,AlAs,2.276,"AlAs is Zincblende...","AlAs is Zincblende... (无键长)","AlAs has Zincblende structure..."
```

---

## 🎯 常见问题

### Q1: 我没有pandas，能用吗？

**A**: 可以！使用 `filter_descriptions_simple.py`（无依赖）

```python
from filter_descriptions_simple import remove_local_information
filtered = remove_local_information(your_description, mode='aggressive')
```

### Q2: 哪个模式最好？

**A**: 对于注意力可解释性分析，推荐 **aggressive** 模式

- 去除最多噪音
- 注意力更集中在关键词
- 与 Middle Fusion 配合效果最好

### Q3: 会丢失重要信息吗？

**A**: 会丢失局部细节（键长数值），但保留结构特征

- ✅ 保留: 空间群、晶系、配位几何、成键拓扑
- ❌ 去除: 具体键长、键角数值

对于可解释性分析，这是合理的权衡。

### Q4: 如何验证过滤效果？

**A**: 对比原始和过滤后的描述

```python
desc = "Your description..."
filtered = remove_local_information(desc, mode='aggressive')

print(f"原始长度: {len(desc)}")
print(f"过滤长度: {len(filtered)}")
print(f"减少: {100*(1-len(filtered)/len(desc)):.1f}%")
print(f"\n原始:\n{desc}")
print(f"\n过滤:\n{filtered}")
```

---

## 📚 相关文档

- `GLOBAL_INFORMATION_FILTERING_GUIDE.md` - 完整使用指南
- `demo_filter_usage.py` - 可运行的示例代码
- `filter_descriptions_simple.py` - 简化版脚本（推荐）
- `filter_global_information.py` - 完整版脚本（需要pandas）

---

## ✅ 快速检查清单

在使用前确认：

- [ ] 已导入函数：`from filter_descriptions_simple import remove_local_information`
- [ ] 选择模式：`mode='aggressive'`（推荐）
- [ ] 测试单个描述确认效果
- [ ] 在分析管道中集成过滤步骤

---

## 🎓 推荐工作流

```
步骤 1: 准备数据
  ↓
步骤 2: 过滤描述 (filter_global_information.py)
  ↓
步骤 3: 使用过滤后的描述训练模型 (Middle Fusion + Fine-Grained Attention)
  ↓
步骤 4: 分析注意力热图 (demo_robust_attention.py)
  ↓
结果: 清晰的、有意义的注意力分布！
```

---

**开始使用**：

```bash
# 运行演示
python demo_filter_usage.py

# 查看您的第一个过滤结果！
```
