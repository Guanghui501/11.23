# MBJ Bandgap Optuna 超参数调优指南

专门用于优化 MBJ Bandgap 预测模型的 Optuna 超参数调优框架。

## 🎯 目标

使用 Optuna 自动搜索最佳超参数组合，以最小化 MBJ Bandgap 预测的验证集 MAE（平均绝对误差）。

## 📋 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [运行优化](#运行优化)
4. [参数说明](#参数说明)
5. [结果分析](#结果分析)
6. [完整训练](#完整训练)

## 🚀 快速开始

### 方式 1: 使用快速启动脚本（推荐）

```bash
# 基本用法（50次试验）
./run_mbj_optuna.sh

# 自定义参数
./run_mbj_optuna.sh 100 100 4 ../dataset/ mbj_results 20

# 参数说明:
# 100 = 试验次数
# 100 = 每次试验的训练轮数
# 4 = 并行作业数
# ../dataset/ = 数据集目录
# mbj_results = 输出目录
# 20 = 早停轮数
```

### 方式 2: 直接运行 Python 脚本

```bash
# 基本用法
python train_mbj_with_optuna.py --n_trials 50

# 并行优化（4个进程）
python train_mbj_with_optuna.py --n_trials 100 --n_jobs 4

# 自定义数据路径
python train_mbj_with_optuna.py \
    --root_dir ../dataset/ \
    --n_trials 50 \
    --n_epochs 100 \
    --output_dir mbj_optuna_results
```

## 📂 数据准备

### 数据集结构

确保你的数据集按以下结构组织：

```
dataset/
└── jarvis/
    └── mbj_bandgap/
        ├── cif/                # CIF 晶体结构文件
        │   ├── JVASP-1.cif
        │   ├── JVASP-2.cif
        │   └── ...
        └── description.csv     # 包含 ID、Description 和 target 列
```

### description.csv 格式

CSV 文件必须包含以下列：

| ID | Description | target |
|----|-------------|--------|
| JVASP-1 | Crystal structure description... | 1.234 |
| JVASP-2 | Another crystal description... | 2.345 |
| ... | ... | ... |

- **ID**: 结构唯一标识符（对应 CIF 文件名，不含 .cif 后缀）
- **Description**: 晶体结构的文本描述
- **target**: MBJ bandgap 值（eV）

## ⚙️ 运行优化

### 基本优化

```bash
# 50次试验，每次100轮
python train_mbj_with_optuna.py \
    --n_trials 50 \
    --n_epochs 100 \
    --output_dir mbj_optuna_results
```

### 并行优化（加速）

```bash
# 使用4个并行进程
python train_mbj_with_optuna.py \
    --n_trials 100 \
    --n_epochs 100 \
    --n_jobs 4 \
    --output_dir mbj_optuna_results
```

### 继续之前的优化

```bash
# 从之前的 study 继续
python train_mbj_with_optuna.py \
    --n_trials 50 \
    --load_study mbj_optuna_results/optuna_study.db \
    --study_name mbj_bandgap_optuna_20240101_120000
```

### 长时间优化（设置超时）

```bash
# 运行6小时（21600秒）
python train_mbj_with_optuna.py \
    --n_trials 1000 \
    --timeout 21600 \
    --output_dir mbj_optuna_results
```

## 📊 参数说明

### 可调超参数（19+ 个）

#### 1. 模型架构参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `alignn_layers` | [2, 6] | ALIGNN 图卷积层数 |
| `gcn_layers` | [2, 6] | GCN 图卷积层数 |
| `hidden_features` | {128, 256, 512} | 隐藏层特征维度 |
| `embedding_features` | {32, 64, 128} | 嵌入层特征维度 |
| `edge_input_features` | {40, 80, 120} | 边特征输入维度 |
| `triplet_input_features` | {20, 40, 60} | 三元组特征输入维度 |

#### 2. 训练参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `learning_rate` | [1e-4, 1e-2] (log) | 学习率 |
| `weight_decay` | [1e-6, 1e-3] (log) | 权重衰减（L2正则化） |
| `batch_size` | {16, 32, 64} | 批次大小 |
| `graph_dropout` | [0.0, 0.5] | 图卷积层 dropout 率 |

#### 3. 跨模态注意力参数（晚期融合）

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `use_cross_modal_attention` | {True, False} | 是否使用跨模态注意力 |
| `cross_modal_hidden_dim` | {128, 256, 512} | 跨模态注意力隐藏层维度 |
| `cross_modal_num_heads` | {2, 4, 8} | 跨模态注意力头数 |
| `cross_modal_dropout` | [0.0, 0.3] | 跨模态注意力 dropout |

#### 4. 细粒度注意力参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `use_fine_grained_attention` | {True, False} | 是否使用细粒度注意力 |
| `fine_grained_num_heads` | {4, 8, 16} | 细粒度注意力头数 |
| `fine_grained_dropout` | [0.0, 0.3] | 细粒度注意力 dropout |

#### 5. 中期融合参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `use_middle_fusion` | {True, False} | 是否使用中期融合 |
| `middle_fusion_layers` | 动态 | 融合层位置（根据 alignn_layers 调整） |
| `middle_fusion_hidden_dim` | {64, 128, 256} | 中期融合隐藏层维度 |
| `middle_fusion_num_heads` | {1, 2, 4} | 中期融合注意力头数 |
| `middle_fusion_dropout` | [0.0, 0.3] | 中期融合 dropout |

### 命令行参数

```bash
--root_dir          数据集根目录（默认: ../dataset/）
--n_trials          试验次数（默认: 50）
--n_epochs          每次试验的训练轮数（默认: 100）
--early_stopping    早停轮数（默认: 20）
--output_dir        输出目录（默认: mbj_optuna_results）
--study_name        Optuna study 名称（可选）
--n_jobs            并行作业数（默认: 1，-1 表示所有 CPU）
--timeout           优化超时时间（秒，可选）
--load_study        加载已有 study 数据库（可选）
```

## 📈 结果分析

### 输出文件

优化完成后，输出目录包含：

1. **best_params_mbj.json** - 最佳超参数
2. **all_trials_mbj.csv** - 所有试验结果
3. **optuna_study.db** - Optuna study 数据库
4. **mbj_optimization_history.html** - 优化历史图
5. **mbj_param_importances.html** - 参数重要性图
6. **mbj_parallel_coordinate.html** - 并行坐标图

### 查看最佳参数

```bash
# 查看 JSON 文件
cat mbj_optuna_results/best_params_mbj.json

# 或使用 Python 解析
python -c "
import json
with open('mbj_optuna_results/best_params_mbj.json') as f:
    data = json.load(f)
    print(f\"最佳 MAE: {data['best_value']:.6f} eV\")
    print('参数:')
    for k, v in data['best_params'].items():
        print(f'  {k}: {v}')
"
```

### 分析所有试验

```bash
# 使用 pandas 分析
python << EOF
import pandas as pd
df = pd.read_csv('mbj_optuna_results/all_trials_mbj.csv')
completed = df[df['state'] == 'COMPLETE']

print(f"完成的试验: {len(completed)}")
print(f"最佳 MAE: {completed['value'].min():.6f} eV")
print(f"最差 MAE: {completed['value'].max():.6f} eV")
print(f"平均 MAE: {completed['value'].mean():.6f} eV")
print(f"标准差: {completed['value'].std():.6f} eV")

# 显示最佳10个试验
print("\n最佳10个试验:")
print(completed.nsmallest(10, 'value')[['number', 'value']])
EOF
```

### 可视化分析

在浏览器中打开生成的 HTML 文件：

```bash
# 优化历史 - 查看 MAE 随试验的变化
firefox mbj_optuna_results/mbj_optimization_history.html

# 参数重要性 - 了解哪些参数最关键
firefox mbj_optuna_results/mbj_param_importances.html

# 并行坐标图 - 理解参数组合
firefox mbj_optuna_results/mbj_parallel_coordinate.html
```

## 🎓 完整训练

找到最佳参数后，使用这些参数进行完整训练（更多 epoch）：

```bash
python train_with_best_params.py \
    --best_params mbj_optuna_results/best_params_mbj.json \
    --epochs 500 \
    --dataset user_data \
    --target target \
    --output_dir mbj_best_model \
    --early_stopping_patience 50
```

### 训练输出

完整训练会生成：

- `config.json` - 训练配置
- `training_history.json` - 训练历史
- `final_results.json` - 最终结果
- `checkpoints/` - 模型检查点
- `tb_logs/` - TensorBoard 日志

### 监控训练

```bash
# 使用 TensorBoard 监控
tensorboard --logdir mbj_best_model/tb_logs
```

## 💡 最佳实践

### 1. 分阶段优化

```bash
# 阶段1: 快速探索（50次试验，100轮）
python train_mbj_with_optuna.py --n_trials 50 --n_epochs 100

# 阶段2: 精细搜索（100次试验，200轮）
python train_mbj_with_optuna.py --n_trials 100 --n_epochs 200 \
    --load_study mbj_optuna_results/optuna_study.db

# 阶段3: 最终优化（50次试验，300轮）
python train_mbj_with_optuna.py --n_trials 50 --n_epochs 300 \
    --load_study mbj_optuna_results/optuna_study.db
```

### 2. 并行加速

- 使用多进程并行优化：`--n_jobs 4`
- 适合多核 CPU 或多 GPU 环境
- 注意内存占用

### 3. 早停设置

- 较快试验：`--early_stopping 10`
- 标准试验：`--early_stopping 20`
- 谨慎试验：`--early_stopping 50`

### 4. 试验次数建议

- 快速测试：20-50 次
- 标准优化：50-100 次
- 深度优化：100-200 次
- 超级优化：200+ 次

## 🔧 故障排除

### 问题 1: 内存不足

**解决方案:**
- 减小 batch_size 选项
- 减少并行作业数 `--n_jobs`
- 使用更少的 hidden_features

### 问题 2: 所有试验被剪枝

**解决方案:**
- 增加 `n_startup_trials`（修改脚本）
- 增加每次试验的轮数 `--n_epochs`
- 检查数据质量

### 问题 3: 训练太慢

**解决方案:**
- 减少每次试验的轮数 `--n_epochs 50`
- 使用并行优化 `--n_jobs 4`
- 减小模型大小

### 问题 4: 数据加载失败

**解决方案:**
- 检查数据集路径 `--root_dir`
- 确认 description.csv 格式正确
- 验证 CIF 文件存在

## 📚 参考

- [Optuna 官方文档](https://optuna.readthedocs.io/)
- [ALIGNN 论文](https://www.nature.com/articles/s41524-021-00650-1)
- [完整 Optuna 调优指南](./OPTUNA_TUNING_README.md)

## 🎯 预期结果

基于历史优化经验，MBJ Bandgap 预测的典型结果：

- **优秀**: MAE < 0.15 eV
- **良好**: MAE < 0.20 eV
- **可接受**: MAE < 0.30 eV

实际结果取决于：
- 数据集质量和大小
- 文本描述的信息量
- 超参数搜索空间
- 训练轮数

---

**祝您调参顺利！** 🎉

有问题请参考 [OPTUNA_TUNING_README.md](./OPTUNA_TUNING_README.md) 获取更多帮助。
