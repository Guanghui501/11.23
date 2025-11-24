# Optuna 超参数调优指南

本指南介绍如何使用 Optuna 进行 ALIGNN 模型的超参数调优。

## ✨ 新功能：中期融合参数调优

现在支持优化**中期融合（Mid-level Fusion）**参数！中期融合在 ALIGNN 图编码的中间层插入文本信息，允许：
- 在不同层级进行多模态特征融合
- 动态选择最佳融合层位置（如第2层、第1和第3层组合等）
- 优化融合机制的架构参数（隐藏维度、注意力头数、dropout等）

总计可调超参数：**19+ 个**，涵盖模型架构、训练、注意力机制和中期融合设置。

## 📋 目录

1. [安装依赖](#安装依赖)
2. [快速开始](#快速开始)
3. [详细说明](#详细说明)
4. [可调参数](#可调参数)
5. [示例](#示例)
6. [结果分析](#结果分析)

## 🔧 安装依赖

```bash
pip install optuna plotly kaleido
```

## 🚀 快速开始

### 步骤 1: 运行 Optuna 超参数搜索

```bash
# 基本用法（50 次试验）
python train_optuna.py --n_trials 50 --output_dir optuna_results

# 更多试验次数
python train_optuna.py --n_trials 100 --output_dir optuna_results

# 并行运行（使用 4 个进程）
python train_optuna.py --n_trials 100 --n_jobs 4 --output_dir optuna_results
```

### 步骤 2: 使用最佳参数训练完整模型

```bash
python train_with_best_params.py \
    --best_params optuna_results/best_params.json \
    --epochs 500 \
    --output_dir best_model_output
```

## 📖 详细说明

### train_optuna.py

使用 Optuna 自动搜索最佳超参数的脚本。

**参数说明:**

```bash
--n_trials          # Optuna 试验次数（默认: 50）
--n_epochs          # 每次试验的训练轮数（默认: 100）
--dataset           # 数据集名称（默认: user_data）
--target            # 目标属性（默认: target）
--output_dir        # 输出目录（默认: optuna_results）
--study_name        # Optuna study 名称（可选）
--n_jobs            # 并行作业数（默认: 1，-1 表示使用所有 CPU）
--timeout           # 优化超时时间（秒，可选）
--load_study        # 加载已有的 study 数据库路径（可选）
```

**输出文件:**

- `best_params.json` - 最佳超参数
- `all_trials.csv` - 所有试验的结果
- `optuna_study.db` - Optuna study 数据库
- `optimization_history.html` - 优化历史可视化
- `param_importances.html` - 参数重要性可视化
- `parallel_coordinate.html` - 并行坐标图

### train_with_best_params.py

使用 Optuna 找到的最佳参数训练完整模型。

**参数说明:**

```bash
--best_params               # 最佳参数 JSON 文件路径（必需）
--epochs                    # 训练轮数（默认: 500）
--dataset                   # 数据集名称（默认: user_data）
--target                    # 目标属性（默认: target）
--output_dir                # 输出目录（默认: best_model_output）
--no_early_stopping         # 禁用早停
--early_stopping_patience   # 早停轮数（默认: 50）
```

**输出文件:**

- `config.json` - 训练配置
- `training_history.json` - 训练历史
- `final_results.json` - 最终结果
- `checkpoints/` - 模型检查点
- `tb_logs/` - TensorBoard 日志

## 🎯 可调参数

### 模型架构参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `alignn_layers` | [2, 6] | ALIGNN 层数 |
| `gcn_layers` | [2, 6] | GCN 层数 |
| `hidden_features` | {128, 256, 512} | 隐藏层特征数 |
| `embedding_features` | {32, 64, 128} | 嵌入特征数 |
| `edge_input_features` | {40, 80, 120} | 边输入特征数 |
| `triplet_input_features` | {20, 40, 60} | 三元组输入特征数 |

### 训练参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `learning_rate` | [1e-4, 1e-2] (log) | 学习率 |
| `weight_decay` | [1e-6, 1e-3] (log) | 权重衰减 |
| `batch_size` | {16, 32, 64, 128} | 批次大小 |
| `graph_dropout` | [0.0, 0.5] | 图dropout率 |

### 注意力机制参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `use_cross_modal_attention` | {True, False} | 是否使用跨模态注意力 |
| `cross_modal_hidden_dim` | {128, 256, 512} | 跨模态注意力隐藏层维度 |
| `cross_modal_num_heads` | {2, 4, 8} | 跨模态注意力头数 |
| `cross_modal_dropout` | [0.0, 0.3] | 跨模态注意力dropout |
| `use_fine_grained_attention` | {True, False} | 是否使用细粒度注意力 |
| `fine_grained_num_heads` | {4, 8, 16} | 细粒度注意力头数 |
| `fine_grained_dropout` | [0.0, 0.3] | 细粒度注意力dropout |

### 中期融合参数

| 参数 | 搜索范围 | 说明 |
|------|---------|------|
| `use_middle_fusion` | {True, False} | 是否使用中期融合 |
| `middle_fusion_layers` | {"2", "1,3", "2,3", "1,2,3"} | 插入融合的层索引（根据模型层数动态调整） |
| `middle_fusion_hidden_dim` | {64, 128, 256} | 中期融合隐藏层维度 |
| `middle_fusion_num_heads` | {1, 2, 4} | 中期融合注意力头数 |
| `middle_fusion_dropout` | [0.0, 0.3] | 中期融合dropout |

**注意**: 中期融合在 ALIGNN 层的中间位置插入文本-图特征融合，允许文本信息调制节点表示。层索引的选择会根据 `alignn_layers` 参数自动调整：
- 如果 `alignn_layers >= 4`: 可选 "2", "1,3", "2,3", "1,2,3"
- 如果 `alignn_layers >= 3`: 可选 "1", "2", "1,2"
- 如果 `alignn_layers < 3`: 可选 "1"

## 💡 示例

### 示例 1: 基本使用

```bash
# 1. 运行 50 次试验
python train_optuna.py --n_trials 50 --output_dir my_optuna_results

# 2. 查看最佳参数
cat my_optuna_results/best_params.json

# 3. 使用最佳参数训练
python train_with_best_params.py \
    --best_params my_optuna_results/best_params.json \
    --epochs 500 \
    --output_dir my_best_model
```

### 示例 2: 自定义数据集

```bash
# 1. 对自定义数据集进行调优
python train_optuna.py \
    --n_trials 100 \
    --dataset user_data \
    --target band_gap \
    --output_dir bandgap_optuna

# 2. 使用最佳参数训练
python train_with_best_params.py \
    --best_params bandgap_optuna/best_params.json \
    --dataset user_data \
    --target band_gap \
    --epochs 1000 \
    --output_dir bandgap_best_model
```

### 示例 3: 并行搜索

```bash
# 使用 4 个并行作业加速搜索
python train_optuna.py \
    --n_trials 200 \
    --n_jobs 4 \
    --output_dir parallel_optuna
```

### 示例 4: 继续之前的搜索

```bash
# 加载之前的 study 并继续搜索
python train_optuna.py \
    --n_trials 50 \
    --load_study optuna_results/optuna_study.db \
    --study_name alignn_optuna_20240101_120000 \
    --output_dir optuna_results
```

### 示例 5: 设置超时

```bash
# 设置 6 小时的超时（21600 秒）
python train_optuna.py \
    --n_trials 1000 \
    --timeout 21600 \
    --output_dir optuna_timeout
```

## 📊 结果分析

### 查看可视化结果

训练完成后，在输出目录中会生成以下 HTML 文件：

1. **optimization_history.html** - 显示优化过程中验证 MAE 的变化
2. **param_importances.html** - 显示各个超参数的重要性
3. **parallel_coordinate.html** - 并行坐标图，展示参数组合

在浏览器中打开这些文件：

```bash
# 在浏览器中打开
firefox optuna_results/optimization_history.html
firefox optuna_results/param_importances.html
firefox optuna_results/parallel_coordinate.html
```

### 分析所有试验

所有试验的详细结果保存在 CSV 文件中：

```bash
# 查看所有试验
cat optuna_results/all_trials.csv

# 使用 pandas 分析
python -c "
import pandas as pd
df = pd.read_csv('optuna_results/all_trials.csv')
print(df.describe())
print('\n最佳 10 个试验:')
print(df.nsmallest(10, 'value'))
"
```

### TensorBoard 监控

使用最佳参数训练时，可以用 TensorBoard 监控：

```bash
tensorboard --logdir best_model_output/tb_logs
```

## 🔍 高级技巧

### 自定义搜索空间

如果需要调整搜索空间，编辑 `train_optuna.py` 中的 `objective` 函数：

```python
# 示例：更改学习率搜索范围
learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

# 示例：添加新的离散参数
new_param = trial.suggest_categorical("new_param", [10, 20, 30])
```

### 使用不同的剪枝策略

在 `train_optuna.py` 中修改 pruner：

```python
# MedianPruner（默认）
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=1,
)

# HyperbandPruner（更激进）
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=100,
    reduction_factor=3,
)

# SuccessiveHalvingPruner
pruner = optuna.pruners.SuccessiveHalvingPruner()
```

### 多目标优化

如果想同时优化多个指标（如 MAE 和推理速度），可以修改为多目标优化：

```python
study = optuna.create_study(
    directions=["minimize", "minimize"],  # [MAE, inference_time]
    pruner=pruner,
)
```

## ⚠️ 注意事项

1. **计算资源**: 超参数搜索需要大量计算资源，建议：
   - 先用较少试验次数（如 20-50）快速测试
   - 使用 GPU 加速训练
   - 使用并行搜索（`--n_jobs`）

2. **早停**: 在 Optuna 搜索时使用较少的 epoch（如 100），最终训练时使用更多 epoch（如 500-1000）

3. **剪枝**: Optuna 会自动剪枝表现不佳的试验，这是正常的

4. **数据集**: 确保数据集已准备好并放在正确位置

5. **内存**: 如果遇到内存不足，可以：
   - 减小 `batch_size`
   - 减小 `hidden_features`
   - 使用更小的模型

## 📚 参考资源

- [Optuna 官方文档](https://optuna.readthedocs.io/)
- [Optuna 教程](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [ALIGNN 论文](https://www.nature.com/articles/s41524-021-00650-1)

## 🐛 故障排除

### 问题 1: CUDA out of memory

**解决方案:**
- 减小 batch_size
- 减小模型大小（hidden_features, alignn_layers）
- 使用梯度累积

### 问题 2: 所有试验都被剪枝

**解决方案:**
- 增加 `n_startup_trials`
- 增加 `n_warmup_steps`
- 调整剪枝策略

### 问题 3: 搜索时间过长

**解决方案:**
- 减少每次试验的 epoch 数（`--n_epochs`）
- 使用并行搜索（`--n_jobs`）
- 设置超时时间（`--timeout`）

### 问题 4: 无法生成可视化图表

**解决方案:**
```bash
pip install plotly kaleido
```

---

**祝您调参顺利！** 🎉
