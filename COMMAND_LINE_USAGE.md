# 命令行清理工具使用指南

## 🚀 快速开始

### 最简单的使用方法

```bash
python clean_descriptions.py -i input.csv -o output.csv
```

就这么简单！

---

## 📋 所有参数说明

### 必需参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--input` | `-i` | 输入CSV文件 | `-i data.csv` |
| `--output` | `-o` | 输出CSV文件 | `-o cleaned.csv` |

### 可选参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--column` | `-c` | `description` | 要处理的列名 |
| `--output-column` | 无 | `{列名}_cleaned` | 输出列名 |
| `--mode` | `-m` | `aggressive` | 过滤模式 |
| `--verbose` | `-v` | 关闭 | 显示详细信息 |
| `--help` | `-h` | - | 显示帮助信息 |
| `--version` | 无 | - | 显示版本号 |

---

## 💡 使用示例

### 示例 1: 基本使用

```bash
python clean_descriptions.py -i desc_mbj_bandgap0.csv -o desc_cleaned.csv
```

**效果**：
- 读取 `desc_mbj_bandgap0.csv`
- 处理 `description` 列
- 创建 `description_cleaned` 列
- 保存到 `desc_cleaned.csv`

---

### 示例 2: 详细输出模式

```bash
python clean_descriptions.py -i input.csv -o output.csv -v
```

**输出示例**：
```
================================================================================
 材料描述清理工具 v2.0
================================================================================
✓ 使用 pandas 处理（快速模式）

输入文件: input.csv
输出文件: output.csv

处理中...
  输入列: description
  输出列: description_cleaned
  模式: aggressive

统计信息:
  处理行数: 100
  原始平均长度: 450 字符
  清理后平均长度: 312 字符
  平均减少: 30.7%

前3个示例:

  1. LiBa4Hf:
     原始: LiBa4Hf crystallizes in the cubic F-43m space group... bond lengths...
     清理: LiBa4Hf crystallizes in the cubic F-43m space group...

  2. AlAs:
     原始: AlAs is Zincblende structured... All Al(1)–As(1) bond lengths are...
     清理: AlAs is Zincblende structured...

✅ 成功! 清理后的文件已保存到: output.csv
```

---

### 示例 3: 指定列名

如果您的列名不是 `description`：

```bash
python clean_descriptions.py -i data.csv -o clean.csv -c text_description
```

或者处理已经过滤过的列：

```bash
python clean_descriptions.py -i data.csv -o final.csv -c description_filtered
```

---

### 示例 4: 自定义输出列名

```bash
python clean_descriptions.py -i input.csv -o output.csv --output-column final_description
```

**结果**：输出CSV中会有 `final_description` 列

---

### 示例 5: 选择过滤模式

#### Aggressive 模式（默认，推荐）

```bash
python clean_descriptions.py -i input.csv -o output.csv -m aggressive
```

**效果**：
- 去除所有键长句子
- 去除所有键角句子
- 去除残留数值
- 最彻底的清理

#### Moderate 模式

```bash
python clean_descriptions.py -i input.csv -o output.csv -m moderate
```

**效果**：
- 去除键长键角句子
- 保留配位几何描述
- 适度清理

#### Conservative 模式

```bash
python clean_descriptions.py -i input.csv -o output.csv -m conservative
```

**效果**：
- 只替换数值为 X
- 保留句子结构
- 最小改动

---

### 示例 6: 处理您的实际数据

假设您有文件 `desc_mbj_bandgap0_aggressive.csv`，想清理 `description_filtered` 列：

```bash
python clean_descriptions.py \
    -i desc_mbj_bandgap0_aggressive.csv \
    -o desc_mbj_bandgap0_final.csv \
    -c description_filtered \
    --output-column description_cleaned \
    -v
```

**结果**：
- 读取 `desc_mbj_bandgap0_aggressive.csv`
- 处理 `description_filtered` 列（去除残留的 "49 Å", "31 Å" 等）
- 创建 `description_cleaned` 列（完全清理）
- 保存到 `desc_mbj_bandgap0_final.csv`
- 显示详细统计信息

---

## 📊 输出格式

### 输入CSV示例

```csv
id,formula,description
1,LiBa4Hf,"LiBa4Hf crystallizes... All bond lengths are 4.25 Å."
2,AlAs,"AlAs is Zincblende... All bond lengths are 2.48 Å."
```

### 输出CSV示例

```csv
id,formula,description,description_cleaned
1,LiBa4Hf,"LiBa4Hf crystallizes... All bond lengths are 4.25 Å.","LiBa4Hf crystallizes..."
2,AlAs,"AlAs is Zincblende... All bond lengths are 2.48 Å.","AlAs is Zincblende..."
```

**注意**：
- 原始列保持不变
- 新增 `description_cleaned` 列
- 可以对比原始和清理后的内容

---

## 🔧 高级用法

### 组合多个参数

```bash
python clean_descriptions.py \
    --input /path/to/data.csv \
    --output /path/to/cleaned.csv \
    --column description_text \
    --output-column clean_desc \
    --mode aggressive \
    --verbose
```

### 处理大文件（推荐安装pandas）

```bash
# 如果有pandas，会自动使用快速模式
pip install pandas

# 然后正常使用
python clean_descriptions.py -i large_file.csv -o output.csv -v
```

**速度对比**：
- 有pandas: ~1000行/秒
- 无pandas: ~100行/秒

---

## ⚠️ 常见错误和解决方案

### 错误 1: 文件不存在

```
❌ 错误: 输入文件不存在: input.csv
```

**解决**：
- 检查文件路径是否正确
- 使用绝对路径：`/home/user/data.csv`
- 或确保文件在当前目录

### 错误 2: 列名不存在

```
❌ 错误: 列 'description' 不存在
   可用列: id, formula, text, bandgap
```

**解决**：
- 使用 `-c` 指定正确的列名
- 例如：`-c text`

### 错误 3: 权限错误

```
❌ 错误: Permission denied: output.csv
```

**解决**：
- 检查输出目录是否有写权限
- 更换输出路径

---

## 🎯 推荐工作流

### 工作流 A: 从原始数据开始

```bash
# 步骤1: 清理原始描述
python clean_descriptions.py \
    -i desc_original.csv \
    -o desc_cleaned.csv \
    -m aggressive \
    -v

# 步骤2: 查看结果
head -20 desc_cleaned.csv

# 步骤3: 用于注意力分析
python demo_robust_attention.py --text-from-csv desc_cleaned.csv
```

### 工作流 B: 清理已有的过滤结果

```bash
# 如果您已经用旧版过滤器处理过，还有残留
python clean_descriptions.py \
    -i desc_with_residuals.csv \
    -o desc_final.csv \
    -c description_filtered \
    --output-column description_cleaned \
    -v
```

### 工作流 C: 批量处理多个文件

```bash
# 创建批处理脚本
for file in data_*.csv; do
    output="cleaned_${file}"
    python clean_descriptions.py -i "$file" -o "$output" -v
done
```

---

## 📈 性能提示

### 提升处理速度

1. **安装pandas**（10倍速度提升）
   ```bash
   pip install pandas
   ```

2. **处理大文件时关闭详细输出**
   ```bash
   python clean_descriptions.py -i large.csv -o output.csv
   # 不使用 -v
   ```

3. **使用SSD存储**
   - 输入输出文件都放在SSD上

---

## 🎓 完整示例脚本

创建一个处理脚本 `batch_clean.sh`：

```bash
#!/bin/bash

# 批量清理所有CSV文件

echo "开始批量清理..."

# 设置参数
INPUT_DIR="./data"
OUTPUT_DIR="./cleaned"
COLUMN="description"
MODE="aggressive"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 处理所有CSV文件
for file in "$INPUT_DIR"/*.csv; do
    filename=$(basename "$file")
    output="$OUTPUT_DIR/cleaned_$filename"

    echo "处理: $filename"

    python clean_descriptions.py \
        -i "$file" \
        -o "$output" \
        -c "$COLUMN" \
        -m "$MODE" \
        -v

    echo "完成: $output"
    echo ""
done

echo "所有文件处理完成!"
```

使用：
```bash
chmod +x batch_clean.sh
./batch_clean.sh
```

---

## 📚 总结

### 最常用的命令

```bash
# 1. 基本使用
python clean_descriptions.py -i input.csv -o output.csv

# 2. 详细输出
python clean_descriptions.py -i input.csv -o output.csv -v

# 3. 指定列名
python clean_descriptions.py -i input.csv -o output.csv -c your_column

# 4. 查看帮助
python clean_descriptions.py --help
```

### 参数速查表

```
-i, --input          输入文件 [必需]
-o, --output         输出文件 [必需]
-c, --column         列名 [默认: description]
-m, --mode           模式 [aggressive/moderate/conservative]
-v, --verbose        详细输出
-h, --help           帮助信息
```

---

**现在就开始使用吧！**

```bash
python clean_descriptions.py -i your_data.csv -o cleaned_data.csv -v
```
