# 命令行工具快速参考

## 🚀 一行命令搞定

```bash
python clean_descriptions.py -i input.csv -o output.csv
```

---

## 📝 常用命令

### 1. 基本使用
```bash
python clean_descriptions.py -i data.csv -o cleaned.csv
```

### 2. 查看详细信息
```bash
python clean_descriptions.py -i data.csv -o cleaned.csv -v
```

### 3. 处理特定列
```bash
python clean_descriptions.py -i data.csv -o cleaned.csv -c description_filtered
```

### 4. 查看帮助
```bash
python clean_descriptions.py --help
```

---

## ⚙️ 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i` | 输入文件 | `-i input.csv` |
| `-o` | 输出文件 | `-o output.csv` |
| `-c` | 列名 | `-c description` |
| `-m` | 模式 | `-m aggressive` |
| `-v` | 详细输出 | `-v` |

---

## 🎯 三种模式

| 模式 | 效果 | 推荐度 |
|------|------|--------|
| `aggressive` | 完全去除键长键角 | ⭐⭐⭐⭐⭐ |
| `moderate` | 保留配位几何 | ⭐⭐⭐ |
| `conservative` | 只隐藏数值 | ⭐⭐ |

---

## 💡 实际示例

### 处理您的数据

```bash
# 如果您有 desc_mbj_bandgap0.csv
python clean_descriptions.py \
    -i desc_mbj_bandgap0.csv \
    -o desc_cleaned.csv \
    -v

# 如果列名是 description_filtered
python clean_descriptions.py \
    -i desc_mbj_bandgap0_aggressive.csv \
    -o desc_final.csv \
    -c description_filtered \
    -v
```

### 批量处理

```bash
# 处理所有CSV文件
for file in *.csv; do
    python clean_descriptions.py -i "$file" -o "cleaned_$file"
done
```

---

## 📊 输出示例

### 输入（有键长）
```
"LiBa4Hf crystallizes... All Ba(1)–Hf(1) bond lengths are 4.25 Å."
```

### 输出（无键长）
```
"LiBa4Hf crystallizes in the cubic F-43m space group..."
```

---

## ✅ 检查结果

```bash
# 查看输出文件
head -10 output.csv

# 检查列名
head -1 output.csv

# 统计行数
wc -l output.csv
```

---

## 🔧 故障排除

### 问题：找不到文件
```bash
# 使用绝对路径
python clean_descriptions.py -i /home/user/data.csv -o /home/user/output.csv
```

### 问题：列名错误
```bash
# 先查看文件列名
head -1 your_file.csv

# 然后指定正确的列名
python clean_descriptions.py -i your_file.csv -o output.csv -c your_column_name
```

### 问题：处理速度慢
```bash
# 安装pandas加速
pip install pandas

# 然后正常使用
python clean_descriptions.py -i large_file.csv -o output.csv
```

---

## 📚 完整文档

- `COMMAND_LINE_USAGE.md` - 详细使用指南
- `clean_descriptions.py --help` - 命令行帮助

---

**立即开始**：

```bash
python clean_descriptions.py -i your_data.csv -o cleaned_data.csv -v
```
