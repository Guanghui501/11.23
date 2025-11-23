# filter_global_information.py 使用总结

## 🎯 核心功能

**去除材料描述中的局部信息（键长、键角数值），保留全局和半全局结构特征**

---

## 🚀 快速上手（复制粘贴即可用）

### 最简单的用法

```python
from filter_descriptions_simple import remove_local_information

# 您的描述
desc = "LiBa4Hf crystallizes in cubic F-43m. Bond lengths are 4.25 Å."

# 过滤
filtered = remove_local_information(desc, mode='aggressive')

print(filtered)
# 输出: "LiBa4Hf crystallizes in cubic F-43m."
```

---

## 📝 三种测试方法

### 方法 1: 快速测试（推荐先试这个）

```bash
python test_filter.py quick
```

**输出**: 3个示例的对比结果

### 方法 2: 交互式测试

```bash
python test_filter.py interactive
```

**功能**: 输入您自己的描述，实时查看过滤效果

### 方法 3: 查看所有示例

```bash
python demo_filter_usage.py
```

**功能**: 展示5种使用场景和代码示例

---

## 💻 在代码中使用

### 场景 A: 处理单个描述

```python
from filter_descriptions_simple import remove_local_information

description = "Your material description with bond length 2.48 Å..."
filtered = remove_local_information(description, mode='aggressive')

# 使用过滤后的描述
model_output = model(structure, filtered)
```

### 场景 B: 批量处理

```python
from filter_descriptions_simple import remove_local_information

# 您的描述列表
descriptions = ["desc1...", "desc2...", "desc3..."]

# 批量过滤
filtered_descriptions = [
    remove_local_information(d, mode='aggressive')
    for d in descriptions
]
```

### 场景 C: 在注意力分析中使用

```python
from filter_descriptions_simple import remove_local_information
import demo_robust_attention

# 过滤描述
filtered_text = remove_local_information(original_text, mode='aggressive')

# 使用过滤后的描述进行注意力分析
results = demo_robust_attention.run_complete_analysis(
    model=model,
    g=graph,
    lg=line_graph,
    text=filtered_text,  # 使用过滤后的文本
    atoms_object=atoms,
    save_dir='./results'
)
```

---

## ⚙️ 三种模式对比

| 模式 | 去除内容 | 压缩率 | 适用场景 |
|------|---------|--------|---------|
| **aggressive** | 所有键长、键角句子 | ~30-40% | 注意力分析（推荐） |
| **moderate** | 键长键角，保留配位 | ~20-30% | 平衡信息和噪音 |
| **conservative** | 只替换数值为X | ~10-15% | 保持句子结构 |

### 示例对比

**原始**:
```
"Ba(1) is bonded to six Ba(1) atoms. There are three shorter (3.60 Å) and
three longer (3.66 Å) bond lengths. All bond lengths are 4.25 Å."
```

**Aggressive**（推荐）:
```
"Ba(1) is bonded to six Ba(1) atoms."
```

**Moderate**:
```
"Ba(1) is bonded to six Ba(1) atoms."
```

**Conservative**:
```
"Ba(1) is bonded to six Ba(1) atoms. There are three shorter (X) and
three longer (X) bond lengths. All bond lengths are X."
```

---

## 📊 效果验证

### 验证脚本

```python
from filter_descriptions_simple import remove_local_information

desc = "Your description..."
filtered = remove_local_information(desc, mode='aggressive')

print(f"原始长度: {len(desc)} 字符")
print(f"过滤长度: {len(filtered)} 字符")
print(f"减少: {100*(1-len(filtered)/len(desc)):.1f}%")

print(f"\n原始:\n{desc}")
print(f"\n过滤:\n{filtered}")
```

### 预期效果

- ✅ 去除所有键长数值（如 "2.48 Å", "3.60 Å"）
- ✅ 去除键角数值（如 "40-54°"）
- ✅ 保留空间群、晶系、结构类型
- ✅ 保留配位几何、成键拓扑
- ✅ 描述长度减少 10-50%

---

## 🔧 常见用法

### 1. 在数据加载时过滤

```python
def load_data(file_path):
    from filter_descriptions_simple import remove_local_information
    import json

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 过滤所有描述
    for item in data:
        item['description'] = remove_local_information(
            item['description'],
            mode='aggressive'
        )

    return data
```

### 2. 在模型前向传播前过滤

```python
def forward_with_filtering(model, structure, description):
    from filter_descriptions_simple import remove_local_information

    # 过滤描述
    filtered = remove_local_information(description, mode='aggressive')

    # 前向传播
    output = model(structure, filtered)

    return output
```

### 3. 创建过滤后的数据集

```python
from filter_descriptions_simple import remove_local_information

# 读取原始数据
with open('materials_data.json', 'r') as f:
    data = json.load(f)

# 过滤并保存
filtered_data = []
for item in data:
    filtered_data.append({
        'formula': item['formula'],
        'structure': item['structure'],
        'description_original': item['description'],
        'description_filtered': remove_local_information(
            item['description'],
            mode='aggressive'
        )
    })

# 保存
with open('materials_data_filtered.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)
```

---

## 📋 文件清单

| 文件 | 用途 | 推荐度 |
|------|------|--------|
| `filter_descriptions_simple.py` | 核心过滤函数（无依赖） | ⭐⭐⭐⭐⭐ |
| `test_filter.py` | 交互式测试工具 | ⭐⭐⭐⭐⭐ |
| `demo_filter_usage.py` | 使用示例展示 | ⭐⭐⭐⭐ |
| `filter_global_information.py` | 完整版（需要pandas） | ⭐⭐⭐ |
| `QUICK_START_FILTER.md` | 快速开始指南 | ⭐⭐⭐⭐⭐ |
| `GLOBAL_INFORMATION_FILTERING_GUIDE.md` | 完整指南 | ⭐⭐⭐⭐ |

---

## 🎓 学习路径

### 新手入门（5分钟）

1. 运行快速测试看效果
   ```bash
   python test_filter.py quick
   ```

2. 在Python中试试
   ```python
   from filter_descriptions_simple import remove_local_information
   filtered = remove_local_information("Your desc...", mode='aggressive')
   print(filtered)
   ```

3. 阅读 `QUICK_START_FILTER.md`

### 实际应用（15分钟）

1. 运行交互式测试，用您自己的数据
   ```bash
   python test_filter.py interactive
   ```

2. 查看使用示例
   ```bash
   python demo_filter_usage.py
   ```

3. 在您的代码中集成

### 深入理解（30分钟）

1. 阅读 `GLOBAL_INFORMATION_FILTERING_GUIDE.md`
2. 了解三种模式的差异
3. 查看源码理解实现原理

---

## ❓ 常见问题

### Q: 我应该用哪个脚本？

**A**: 用 `filter_descriptions_simple.py`（推荐）
- 无依赖，最简单
- 功能完整
- 适合99%的使用场景

### Q: 如何选择模式？

**A**: 对于注意力分析，用 **aggressive**
- 去除最多噪音
- 注意力更集中
- 配合 Middle Fusion 效果最好

### Q: 会丢失重要信息吗？

**A**: 只丢失局部数值，不影响全局理解
- ✅ 保留: 结构类型、空间群、配位方式
- ❌ 去除: 键长数值、键角数值

### Q: 如何测试效果？

**A**: 运行测试脚本查看对比
```bash
python test_filter.py quick
```

---

## ✅ 检查清单

使用前确认：

- [ ] 已安装 Python 3.6+
- [ ] 能导入 `filter_descriptions_simple`
- [ ] 运行 `test_filter.py quick` 查看效果
- [ ] 选择合适的模式（推荐 aggressive）
- [ ] 在小样本上测试确认效果

---

## 🎯 推荐工作流

```
1. 准备材料描述数据
   ↓
2. 使用 filter_global_information.py 过滤局部信息
   filtered = remove_local_information(desc, mode='aggressive')
   ↓
3. 使用过滤后的描述训练/分析模型
   output = model(structure, filtered_description)
   ↓
4. 生成注意力热图
   demo_robust_attention.py
   ↓
5. 结果: 清晰的、集中的注意力分布！
```

---

## 📞 需要帮助？

1. **快速测试**: `python test_filter.py quick`
2. **查看示例**: `python demo_filter_usage.py`
3. **阅读指南**: `QUICK_START_FILTER.md`
4. **完整文档**: `GLOBAL_INFORMATION_FILTERING_GUIDE.md`

---

## 🌟 开始使用

**最简单的方法**：

```python
from filter_descriptions_simple import remove_local_information

# 您的描述
desc = "Material description with bond length 2.48 Å..."

# 过滤
filtered = remove_local_information(desc, mode='aggressive')

# 完成！
print(filtered)
```

**就是这么简单！** 🎉
