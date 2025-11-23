# 简洁清理工具使用指南

## 🎯 功能

**删除包含以下关键词的句子**：
- ✅ `Å` (埃符号)
- ✅ `bond length` / `bond lengths`
- ✅ `shorter`
- ✅ `longer`
- ✅ `tilt angles`

## 🚀 使用方法

### 基本用法

```bash
python clean_simple.py 输入文件.csv 输出文件.csv
```

### 指定列名

```bash
python clean_simple.py 输入文件.csv 输出文件.csv 列名
```

## 📝 示例

### 示例 1: 基本使用

```bash
python clean_simple.py desc_mbj_bandgap0.csv desc_cleaned.csv
```

**假设**: 您的描述列叫 `Description`（默认）

### 示例 2: 指定列名

```bash
python clean_simple.py data.csv cleaned.csv description
```

**适用于**: 列名是小写 `description`

### 示例 3: 处理特定列

```bash
python clean_simple.py data.csv output.csv text_description
```

**适用于**: 列名是 `text_description`

## 📊 清理效果

### 输入示例

```
"LiBa4Hf crystallizes in the cubic F-43m space group. Ba(1) is bonded
to six equivalent Ba(1) atoms. There are three shorter (3.60 Å) and
three longer (3.66 Å) bond lengths. All Ba(1)–Hf(1) bond lengths are 4.25 Å."
```

### 输出示例

```
"LiBa4Hf crystallizes in the cubic F-43m space group. Ba(1) is bonded
to six equivalent Ba(1) atoms."
```

**删除的句子**：
- ❌ "There are three shorter (3.60 Å) and three longer (3.66 Å) bond lengths." ← 包含 `shorter`, `longer`, `Å`
- ❌ "All Ba(1)–Hf(1) bond lengths are 4.25 Å." ← 包含 `bond lengths`, `Å`

## 🔍 工作原理

### 简单直接

1. **分割句子**：按句号 `.` 分割文本
2. **检查关键词**：检查每个句子是否包含关键词
3. **删除句子**：包含关键词的句子直接删除
4. **重组文本**：保留的句子用 `. ` 连接

### 代码逻辑

```python
# 关键词列表
keywords = ['Å', '?', 'bond length', 'shorter', 'longer', 'tilt angle']

# 检查每个句子
for sentence in sentences:
    if any(keyword in sentence for keyword in keywords):
        # 删除这个句子
        continue
    else:
        # 保留这个句子
        keep_sentence(sentence)
```

## 📈 测试结果

使用测试数据 `test_data.csv`：

```
================================================================================
总行数: 3

统计信息:
  原始平均长度: 257 字符
  清理后平均长度: 188 字符
  平均减少: 26.9%

前3个示例:

1. LiBa4Hf:
   LiBa4Hf crystallizes in the cubic F-43m space group. The structure
   consists of four Li clusters inside a Ba4Hf framework. Ba(1) is
   bonded to six equivalent Ba(1) and three equivalent Hf(1) atoms.

2. AlAs:
   AlAs is Zincblende, Sphalerite structured and crystallizes in the
   cubic F-43m space group. Al(1) is bonded to four equivalent As(1)
   atoms to form corner-sharing AlAs4 tetrahedra.

3. NaI:
   NaI is Halite, Rock Salt structured and crystallizes in the cubic
   Fm-3m space group. Na(1) is bonded to six equivalent I(1) atoms to
   form a mixture of corner and edge-sharing NaI6 octahedra.

✅ 完成!
================================================================================
```

## ✅ 优势

### 1. 简单直接
- 只删除包含关键词的句子
- 不使用复杂的正则表达式
- 逻辑清晰易懂

### 2. 彻底清理
- 包含 `Å` 的句子 → 删除
- 包含 `bond length` 的句子 → 删除
- 包含 `shorter/longer` 的句子 → 删除
- 包含 `tilt angle` 的句子 → 删除

### 3. 无残留
- 不会留下 ") and three longer" 等片段
- 整句删除，干净利落

### 4. 直接替换
- 不添加新列
- 直接覆盖原始 Description 列
- 输出简洁

## ⚠️ 注意事项

### 列名大小写

确保列名正确：

```bash
# 如果列名是 Description（大写 D）
python clean_simple.py data.csv output.csv Description

# 如果列名是 description（小写 d）
python clean_simple.py data.csv output.csv description
```

### 查看可用列名

如果不确定列名，运行脚本会提示：

```
❌ 错误: 列 'Description' 不存在
可用列: id, formula, bandgap, description
```

### 编码问题

脚本使用 UTF-8 编码，支持 `Å` 和中文。

## 🔧 对比其他工具

| 工具 | 方法 | 复杂度 | 残留风险 |
|------|------|--------|---------|
| `clean_descriptions.py` | 正则表达式 | 高 | 中 |
| `ultra_clean.py` | 15轮清理 | 很高 | 低 |
| **`clean_simple.py`** | **整句删除** | **低** | **无** |

## 💡 推荐使用场景

### ✅ 推荐

- 简单直接的清理需求
- 不想要复杂的正则表达式
- 确保无残留
- 快速处理

### ❌ 不推荐

- 需要保留部分句子内容
- 需要更精细的控制

## 🎓 完整示例

### 场景：处理您的数据

```bash
# 步骤 1: 查看帮助
python clean_simple.py

# 步骤 2: 处理文件
python clean_simple.py desc_mbj_bandgap0.csv desc_cleaned.csv Description

# 步骤 3: 查看结果
head -5 desc_cleaned.csv

# 步骤 4: 验证（应该没有结果）
grep "bond length" desc_cleaned.csv
grep "Å" desc_cleaned.csv
grep "shorter" desc_cleaned.csv
```

## 📋 快速参考

```bash
# 基本用法
python clean_simple.py input.csv output.csv

# 指定列名
python clean_simple.py input.csv output.csv 列名

# 查看帮助
python clean_simple.py
```

## 🎯 核心特点

- ✅ **简单**：逻辑清晰，易于理解
- ✅ **彻底**：整句删除，无残留
- ✅ **快速**：高效处理
- ✅ **可靠**：结果稳定
- ✅ **简洁**：直接替换，不添加列

---

**立即使用**：

```bash
python clean_simple.py 您的文件.csv 输出文件.csv
```
