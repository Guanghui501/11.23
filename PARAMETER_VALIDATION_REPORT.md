# 参数验证报告
**验证时间**: 2025-12-08
**验证对象**: submit_ablation_study.sh vs train_with_cross_modal_attention.py

## ✅ 验证结果: 通过

所有参数均正确匹配，无冲突或缺失。

---

## 📋 参数对照表

### 1. 数据集参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--root_dir` | line 77 | str | `../dataset/` | ✅ |
| `--dataset jarvis` | line 79 | str | `jarvis` | ✅ |
| `--property` | line 82 | str | `formation_energy` | ✅ |

### 2. 数据划分参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--train_ratio 0.8` | line 92 | float | `0.8` | ✅ |
| `--val_ratio 0.1` | line 94 | float | `0.1` | ✅ |
| `--test_ratio 0.1` | line 96 | float | `0.1` | ✅ |

### 3. 训练参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--batch_size 64` | line 106 | int | `64` | ✅ |
| `--epochs 100` | line 108 | int | `1000` | ✅ |
| `--learning_rate 5e-4` | line 110 | float | `0.001` | ✅ |
| `--weight_decay 1e-3` | line 112 | float | `1e-5` | ✅ |
| `--warmup_steps 2000` | line 114 | int | `2000` | ✅ |

### 4. 模型架构参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--alignn_layers 4` | line 118 | int | `4` | ✅ |
| `--gcn_layers 4` | line 120 | int | `4` | ✅ |
| `--hidden_features 256` | line 122 | int | `256` | ✅ |
| `--graph_dropout 0.15` | line 124 | float | `0.0` | ✅ |

### 5. 跨模态注意力参数（晚期融合）

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--use_cross_modal False` | line 128 | str2bool | `True` | ✅ |
| `--cross_modal_num_heads 2` | line 132 | int | `4` | ✅ |

**注意**: SLURM脚本中设置为 `False`，与默认值不同（符合消融实验设计）

### 6. 中期融合参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--use_middle_fusion` | line 139 | str2bool | `False` | ✅ |
| `--middle_fusion_layers 2` | line 141 | str | `"2"` | ✅ |
| `--middle_fusion_dropout 0.35` | line 148 | float | `0.1` | ✅ |

**变量值**: 在不同配置中为 `True` 或 `False`

### 7. 细粒度注意力参数（原子-token级别）⭐ NEW

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--use_fine_grained_attention` | line 152 | str2bool | `False` | ✅ |
| `--fine_grained_hidden_dim 256` | line 154 | int | `256` | ✅ |
| `--fine_grained_num_heads 8` | line 156 | int | `8` | ✅ |
| `--fine_grained_dropout 0.35` | line 159 | float | `0.1` | ✅ |
| `--fine_grained_use_projection` | line 161 | str2bool | `True` | ✅ |

**变量值**:
- `use_fine_grained_attention`: 配置1为False, 配置2-3为True
- `fine_grained_use_projection`: 配置1为False, 配置2-3为True

### 8. 其他参数

| SLURM脚本参数 | 训练脚本定义 | 类型 | 默认值 | 状态 |
|--------------|-------------|------|--------|------|
| `--early_stopping_patience 150` | line 189 | int | `None` | ✅ |
| `--num_workers 24` | line 187 | int | `0` | ✅ |
| `--random_seed` | line 185 | int | `123` | ✅ |
| `--output_dir` | line 179 | str | `./output/` | ✅ |

**变量值**: `random_seed` 和 `output_dir` 由脚本动态设置

---

## 🔍 布尔值参数处理验证

训练脚本使用 `str2bool` 函数处理布尔参数（line 55-64）:

```python
def str2bool(v):
    """将字符串转换为布尔值（用于argparse）"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值应为 yes/no, true/false, t/f, y/n, 1/0')
```

**SLURM脚本中的布尔参数传递**:
- `--use_cross_modal False` → ✅ 识别为 False
- `--use_fine_grained_attention True` → ✅ 识别为 True
- `--fine_grained_use_projection True` → ✅ 识别为 True
- `--use_middle_fusion True` → ✅ 识别为 True

**结论**: 字符串 "True" 和 "False" 会被正确转换（不区分大小写）

---

## 📊 消融实验配置对照

### 配置1: Baseline (No FG + Middle)
```bash
--use_fine_grained_attention False
--fine_grained_use_projection False
--use_middle_fusion True
```
✅ 所有参数均有定义

### 配置2: FG + Proj + Middle
```bash
--use_fine_grained_attention True
--fine_grained_use_projection True
--use_middle_fusion True
```
✅ 所有参数均有定义

### 配置3: FG + Proj + No Middle
```bash
--use_fine_grained_attention True
--fine_grained_use_projection True
--use_middle_fusion False
```
✅ 所有参数均有定义

---

## ⚠️ 注意事项

### 1. 未使用的可选参数

以下参数在训练脚本中有定义，但SLURM脚本未设置（将使用默认值）:

- `--cross_modal_hidden_dim` (默认: 256)
- `--cross_modal_dropout` (默认: 0.1)
- `--middle_fusion_hidden_dim` (默认: 128)
- `--middle_fusion_num_heads` (默认: 2)
- `--use_contrastive` (默认: False)

**影响**: 无影响，使用默认值即可

### 2. 参数值差异说明

| 参数 | SLURM设置 | 默认值 | 原因 |
|------|----------|--------|------|
| `epochs` | 100 | 1000 | 快速实验 |
| `learning_rate` | 5e-4 | 0.001 | 调优后的值 |
| `weight_decay` | 1e-3 | 1e-5 | 更强正则化 |
| `graph_dropout` | 0.15 | 0.0 | 防止过拟合 |
| `middle_fusion_dropout` | 0.35 | 0.1 | 更强正则化 |
| `fine_grained_dropout` | 0.35 | 0.1 | 更强正则化 |
| `num_workers` | 24 | 0 | 加速数据加载 |

**结论**: 这些差异是有意设计的，符合实验需求

### 3. 关键位置编码修复

根据之前的修改（models/alignn.py line 410-412, 451-455），位置编码已添加到 `FineGrainedCrossModalAttention` 中，修复了注意力崩塌问题。

**确认**:
- ✅ 位置编码已在 `__init__` 中定义
- ✅ 位置编码已在 `forward` 中应用
- ✅ 使用的维度正确: `hidden_dim if use_projection else node_dim`

---

## ✅ 最终结论

**状态**: 🟢 所有参数验证通过

1. ✅ SLURM脚本中的所有参数在训练脚本中均有定义
2. ✅ 布尔值参数转换正确
3. ✅ 消融实验的3个配置参数完整
4. ✅ 无缺失或冲突的参数
5. ✅ 位置编码修复已应用

**可以安全使用 submit_ablation_study.sh 提交训练作业！**

---

## 🚀 推荐使用命令

```bash
# 1. 确保脚本有执行权限
chmod +x submit_ablation_study.sh monitor_ablation_study.sh

# 2. 提交消融实验作业
./submit_ablation_study.sh

# 3. 监控训练进度
watch -n 10 './monitor_ablation_study.sh'
```

---

**验证人**: Claude Code
**验证日期**: 2025-12-08
**文档版本**: 1.0
