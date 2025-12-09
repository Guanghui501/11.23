# 数据预处理指南

## 📋 概述

数据预处理将原始CIF文件和CSV描述文件转换为预处理的pickle文件，可以**大幅加快**后续训练和评估的加载速度。

**加载速度对比**：
- 原始数据加载：~10-30分钟（每次训练/评估）
- 预处理数据加载：~10-30秒（提速100倍+）

**一次预处理，多次使用！**

## 🎯 为什么需要预处理？

在训练或评估时，每次都需要：
1. 从CIF文件读取晶体结构
2. 构建DGL图（atom graph 和 line graph）
3. 归一化文本描述

这些操作**非常耗时**。预处理后，这些都只需做一次！

## 📦 脚本说明

### 1. **preprocess_dataset.py** - 通用预处理脚本
支持任意数据集和属性

### 2. **preprocess_mbj_bandgap.sh** - MBJ Band Gap 快捷脚本
专门为你的MBJ Band Gap模型预配置

## 🚀 快速开始（MBJ Band Gap）

### 方法1: 使用快捷脚本（推荐）

```bash
# 1. 编辑脚本，修改路径
vim preprocess_mbj_bandgap.sh

# 修改这两个路径：
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
OUTPUT_DIR="/public/home/ghzhang/preprocessed_data"

# 2. 运行脚本
chmod +x preprocess_mbj_bandgap.sh
./preprocess_mbj_bandgap.sh
```

### 方法2: 直接使用Python脚本

```bash
python preprocess_dataset.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --output_dir /public/home/ghzhang/preprocessed_data \
    --cutoff 8.0 \
    --max_neighbors 12 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

## 📂 输入文件要求

预处理脚本需要以下文件结构：

```
ROOT_DIR/
  jarvis/
    mbj_bandgap/
      cif/
        JVASP-1.cif
        JVASP-2.cif
        JVASP-3.cif
        ...
      description.csv  ← CSV文件，包含ID、组成、目标值、文本描述等
```

**description.csv 格式**（JARVIS数据集）：
```csv
id,composition,target,description,extra
JVASP-1,Al2O3,2.5,Aluminum oxide with corundum structure...,...
JVASP-2,Si,1.1,Silicon in diamond cubic structure...,...
...
```

## 📊 输出文件结构

预处理完成后会生成：

```
OUTPUT_DIR/
  jarvis/
    mbj_bandgap/
      train.pkl       ← 训练集（80%）
      val.pkl         ← 验证集（10%）
      test.pkl        ← 测试集（10%）
      README.txt      ← 数据说明
```

**每个.pkl文件包含**：
```python
[
    {
        "id": "JVASP-1",
        "graph": (dgl.DGLGraph,),  # Atom graph
        "line_graph": dgl.DGLGraph,  # Line graph
        "text": "normalized text description",
        "target": 2.5
    },
    ...
]
```

## 🔧 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集名称 | `jarvis`, `mp`, `class` |
| `--property` | 属性名称 | `mbj_bandgap`, `formation_energy`, `band_gap` |
| `--root_dir` | 数据集根目录 | `/public/home/ghzhang/crysmmnet-main/dataset` |
| `--output_dir` | 预处理数据输出目录 | `/public/home/ghzhang/preprocessed_data` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cutoff` | 邻居搜索截断半径（Å） | 8.0 |
| `--max_neighbors` | 最大邻居数 | 12 |
| `--use_canonize` | 是否使用规范化坐标 | True |
| `--train_ratio` | 训练集比例 | 0.8 |
| `--val_ratio` | 验证集比例 | 0.1 |
| `--test_ratio` | 测试集比例 | 0.1 |
| `--seed` | 随机种子 | 42 |

## 📝 使用示例

### 示例1: JARVIS - MBJ Band Gap

```bash
python preprocess_dataset.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --output_dir /public/home/ghzhang/preprocessed_data
```

### 示例2: JARVIS - Formation Energy

```bash
python preprocess_dataset.py \
    --dataset jarvis \
    --property formation_energy \
    --root_dir /path/to/dataset \
    --output_dir /path/to/preprocessed_data
```

### 示例3: Material Project - Band Gap

```bash
python preprocess_dataset.py \
    --dataset mp \
    --property band_gap \
    --root_dir /path/to/dataset \
    --output_dir /path/to/preprocessed_data \
    --cutoff 8.0 \
    --max_neighbors 12
```

### 示例4: 自定义数据分割

```bash
python preprocess_dataset.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /path/to/dataset \
    --output_dir /path/to/preprocessed_data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

## ⏱️ 预期处理时间

取决于数据集大小：

| 数据集大小 | 预期时间 |
|-----------|----------|
| ~1,000 样本 | 5-10 分钟 |
| ~5,000 样本 | 10-20 分钟 |
| ~10,000 样本 | 20-40 分钟 |
| ~50,000 样本 | 1-2 小时 |

**提示**：可以在后台运行
```bash
nohup ./preprocess_mbj_bandgap.sh > preprocess.log 2>&1 &
tail -f preprocess.log  # 查看进度
```

## 🔍 如何使用预处理数据？

预处理完成后，在训练和评估时使用预处理数据：

### 训练时使用

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_preprocessed True \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data
```

**关键参数**：
- `--use_preprocessed True` ← 启用预处理数据加载
- `--preprocessed_dir <路径>` ← 指定预处理数据目录

### 评估时使用

```bash
python evaluate_text_masking.py \
    --checkpoint /path/to/best_model.pt \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap
```

评估脚本**要求**使用预处理数据。

## ❓ 常见问题

### Q1: "找不到vocab_mappings.txt"错误

**错误信息**：
```
FileNotFoundError: 无法找到 vocab_mappings.txt 文件
```

**解决方案**：
确保`vocab_mappings.txt`在以下位置之一：
- 当前目录
- `crysmmnet-main/src/` 目录

```bash
# 检查文件
find . -name "vocab_mappings.txt"

# 如果不存在，从仓库复制
cp /path/to/crysmmnet-main/src/vocab_mappings.txt ./
```

### Q2: "CIF目录不存在"错误

**错误信息**：
```
FileNotFoundError: CIF目录不存在: /path/to/jarvis/mbj_bandgap/cif/
```

**解决方案**：
1. 检查`--root_dir`路径是否正确
2. 确认数据集已下载并解压
3. 检查目录结构是否正确

```bash
# 验证目录结构
ls -la /public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap/
```

### Q3: 预处理过程中部分样本失败

**提示信息**：
```
✓ 成功加载: 4500 样本
⚠ 跳过: 50 样本
```

**说明**：
- 正常现象，部分CIF文件可能损坏或格式不正确
- 只要成功样本数量足够，可以继续使用
- 如果失败率>10%，请检查数据质量

### Q4: 内存不足

**错误信息**：
```
MemoryError: Unable to allocate ...
```

**解决方案**：
1. 分批处理数据（修改脚本）
2. 使用更大内存的机器
3. 减少`--max_neighbors`参数

### Q5: 如何验证预处理数据？

```python
import pickle

# 加载并检查
with open('/path/to/preprocessed_data/jarvis/mbj_bandgap/test.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"测试集样本数: {len(test_data)}")
print(f"第一个样本: {test_data[0].keys()}")

# 检查一个样本
sample = test_data[0]
print(f"ID: {sample['id']}")
print(f"Graph nodes: {sample['graph'][0].number_of_nodes()}")
print(f"Graph edges: {sample['graph'][0].number_of_edges()}")
print(f"Line graph nodes: {sample['line_graph'].number_of_nodes()}")
print(f"Text length: {len(sample['text'])}")
print(f"Target: {sample['target']}")
```

### Q6: 需要重新预处理吗？

**需要重新预处理的情况**：
- 更改了`--cutoff`或`--max_neighbors`参数
- 更改了数据集或属性
- 原始数据集有更新

**不需要重新预处理的情况**：
- 更改训练参数（学习率、batch size等）
- 更改模型结构
- 更改融合策略

## 🎯 完整工作流程

### 步骤1: 预处理数据（只需一次）

```bash
# 使用快捷脚本
./preprocess_mbj_bandgap.sh

# 或使用Python脚本
python preprocess_dataset.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --output_dir /public/home/ghzhang/preprocessed_data
```

**预计时间**: 10-30 分钟

### 步骤2: 验证预处理数据

```bash
# 检查文件是否生成
ls -lh /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/

# 应该看到：
# train.pkl  (最大)
# val.pkl    (中等)
# test.pkl   (中等)
# README.txt
```

### 步骤3: 使用预处理数据

#### 3a. 训练模型
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_preprocessed True \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --epochs 100 \
    --batch_size 64
```

#### 3b. 文本遮挡评估
```bash
# 快速测试
./quick_test_masking.sh

# 完整评估
./run_masking_eval_mbj_bandgap.sh
```

## 💡 最佳实践

1. **预处理一次，多次使用**
   - 预处理数据可以用于不同的实验和模型

2. **保存预处理数据**
   - 预处理数据可以保存并分享给团队成员
   - 建议备份预处理数据

3. **使用固定的随机种子**
   - 确保数据分割的可重复性
   - 不同实验使用相同的`--seed`

4. **验证数据质量**
   - 预处理后，检查样本数量是否合理
   - 查看README.txt了解数据统计

5. **磁盘空间**
   - 预处理数据通常是原始数据的2-5倍大小
   - 确保有足够的磁盘空间

## 📚 相关文档

- **文本遮挡评估指南**: `TEXT_MASKING_EVALUATION_GUIDE.md`
- **MBJ Band Gap评估指南**: `README_MBJ_BANDGAP_MASKING_EVAL.md`
- **训练指南**: `TRAINING_GUIDE.md`

## 🆘 获取帮助

如果遇到问题：

1. **检查错误信息**：脚本会输出详细的错误信息
2. **查看日志**：预处理过程会显示详细进度
3. **验证输入**：确认数据集路径和文件存在
4. **查看示例**：参考本指南中的示例

预处理成功的标志：
```
✓ 成功加载: XXXX 样本
✓ 成功构建: XXXX 个图
✓ train 集: /path/to/train.pkl
✓ val 集: /path/to/val.pkl
✓ test 集: /path/to/test.pkl
```

祝预处理顺利！ 🎉
