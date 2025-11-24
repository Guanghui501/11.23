# Optuna Pruning 策略指南

本指南详细介绍 Optuna 超参数调优中各种 Pruning（剪枝）策略的使用方法、特点和适用场景。

## 📋 目录

1. [什么是 Pruning](#什么是-pruning)
2. [可用策略](#可用策略)
3. [策略对比](#策略对比)
4. [使用示例](#使用示例)
5. [参数调优](#参数调优)
6. [最佳实践](#最佳实践)

## 🎯 什么是 Pruning

Pruning（剪枝）是 Optuna 的一个重要功能，用于**提前终止表现不佳的试验**，从而：
- ⚡ **节省计算资源** - 避免浪费时间在明显差的超参数组合上
- 🚀 **加快优化速度** - 更快找到最优超参数
- 📊 **提高效率** - 在相同时间内尝试更多有希望的参数组合

### 工作原理

1. 在训练过程中，定期向 Optuna 报告中间结果（如每个 epoch 的验证 MAE）
2. Pruner 根据当前性能与历史试验对比，决定是否终止
3. 如果试验被判定为"无希望"，则提前终止

## 📊 可用策略

### 1. MedianPruner（中位数剪枝）⭐ 推荐

**原理**: 如果试验在某一步的性能低于所有已完成试验在该步的中位数，则剪枝。

**特点**:
- ✅ 稳定可靠，适合大多数场景
- ✅ 不会过于激进，保留有潜力的试验
- ✅ 对异常值不敏感

**参数**:
```bash
--pruner median
--pruner_startup_trials 5      # 前5个试验不剪枝
--pruner_warmup_steps 10        # 每个试验前10步不剪枝
--pruner_interval_steps 1       # 每步检查一次
```

**适用场景**:
- ✅ 默认选择，适合大多数任务
- ✅ 训练稳定，收敛曲线规律
- ✅ 需要平衡探索和利用

### 2. HyperbandPruner（Hyperband 剪枝）🚀 高效

**原理**: 基于 Hyperband 算法，自适应地分配资源，快速淘汰差的试验。

**特点**:
- ✅ 高效，快速找到好的参数
- ✅ 自适应资源分配
- ⚠️ 可能过早淘汰潜力股

**参数**:
```bash
--pruner hyperband
# max_resource 会自动设置为 n_epochs
# reduction_factor 默认为 3
```

**适用场景**:
- ✅ 计算资源有限
- ✅ 需要快速得到结果
- ✅ 大规模超参数搜索

### 3. SuccessiveHalvingPruner（连续减半剪枝）⚡ 激进

**原理**: 在每个阶段淘汰一半表现最差的试验。

**特点**:
- ✅ 非常激进，快速收敛
- ✅ 适合预算有限的情况
- ⚠️ 可能错过慢热型参数组合

**参数**:
```bash
--pruner successive_halving
# reduction_factor 默认为 4
```

**适用场景**:
- ✅ 时间紧迫
- ✅ 初步筛选大量候选
- ✅ 训练收敛快的模型

### 4. PercentilePruner（百分位剪枝）📊 可控

**原理**: 如果试验的性能低于所有试验在该步的某个百分位，则剪枝。

**特点**:
- ✅ 灵活可控，可调节激进程度
- ✅ 通过百分位参数精确控制剪枝率
- ✅ 适合精细调优

**参数**:
```bash
--pruner percentile
--percentile_pruner_percentile 25.0  # 低于25%则剪枝
--pruner_startup_trials 5
--pruner_warmup_steps 10
```

**适用场景**:
- ✅ 需要精确控制剪枝激进程度
- ✅ 已经了解大致的性能分布
- ✅ 后期精细调优

### 5. PatientPruner（耐心剪枝）🛡️ 保守

**原理**: 包装其他 pruner，但会给予试验更多机会，在连续多步无改善时才剪枝。

**特点**:
- ✅ 更保守，不会过早放弃
- ✅ 适合训练不稳定的模型
- ✅ 避免错过慢热型参数

**参数**:
```bash
--pruner patient
--patient_pruner_patience 3      # 连续3步无改善才剪枝
--pruner_startup_trials 5
--pruner_warmup_steps 10
```

**适用场景**:
- ✅ 训练过程有波动
- ✅ 某些参数慢热但最终效果好
- ✅ 计算资源充足，不急于剪枝

### 6. NopPruner（不剪枝）

**原理**: 不进行任何剪枝，所有试验都完整运行。

**特点**:
- ✅ 确保所有参数组合都充分尝试
- ⚠️ 浪费资源在明显差的组合上

**参数**:
```bash
--pruner none
```

**适用场景**:
- ✅ 调试和验证
- ✅ 试验次数很少（<20）
- ✅ 需要完整的训练曲线数据

## 📊 策略对比

| 策略 | 激进程度 | 效率 | 稳定性 | 资源节省 | 推荐指数 |
|------|---------|------|--------|---------|---------|
| Median | 中等 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Hyperband | 高 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SuccessiveHalving | 很高 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Percentile | 可调 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Patient | 低 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| None | 无 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |

## 🚀 使用示例

### 示例 1: 默认设置（MedianPruner）

```bash
# 使用默认的 Median Pruner
python train_optuna.py --n_trials 50
python train_mbj_with_optuna.py --n_trials 50
```

### 示例 2: 快速筛选（HyperbandPruner）

```bash
# 快速尝试大量参数组合
python train_optuna.py \
    --n_trials 100 \
    --n_epochs 100 \
    --pruner hyperband
```

### 示例 3: 激进剪枝（SuccessiveHalvingPruner）

```bash
# 时间紧迫，需要快速结果
python train_mbj_with_optuna.py \
    --n_trials 200 \
    --n_epochs 50 \
    --pruner successive_halving
```

### 示例 4: 精细控制（PercentilePruner）

```bash
# 控制剪枝激进程度
python train_optuna.py \
    --n_trials 50 \
    --pruner percentile \
    --percentile_pruner_percentile 20.0  # 低于20%剪枝
```

### 示例 5: 保守策略（PatientPruner）

```bash
# 训练不稳定，需要更多耐心
python train_mbj_with_optuna.py \
    --n_trials 50 \
    --pruner patient \
    --patient_pruner_patience 5  # 连续5步无改善才剪枝
```

### 示例 6: 不剪枝（调试用）

```bash
# 调试或验证，运行所有试验
python train_optuna.py \
    --n_trials 20 \
    --pruner none
```

## ⚙️ 参数调优

### startup_trials（启动试验数）

**作用**: 在前 N 个试验中不进行剪枝，收集基准数据。

**调优建议**:
- 少量试验（<50）: `startup_trials=3`
- 中等试验（50-100）: `startup_trials=5`
- 大量试验（>100）: `startup_trials=10`

```bash
--pruner_startup_trials 5
```

### warmup_steps（预热步数）

**作用**: 每个试验的前 N 步不剪枝，允许初期波动。

**调优建议**:
- 快速收敛模型: `warmup_steps=5`
- 中等速度模型: `warmup_steps=10`
- 慢收敛模型: `warmup_steps=20`

```bash
--pruner_warmup_steps 10
```

### interval_steps（检查间隔）

**作用**: 每 N 步检查一次是否剪枝。

**调优建议**:
- 频繁检查: `interval_steps=1` （推荐）
- 降低开销: `interval_steps=5`

```bash
--pruner_interval_steps 1
```

### percentile（百分位阈值）

**作用**: PercentilePruner 的剪枝阈值。

**调优建议**:
- 激进剪枝: `percentile=10.0` （低于10%剪枝）
- 中等剪枝: `percentile=25.0` （默认）
- 保守剪枝: `percentile=40.0`

```bash
--percentile_pruner_percentile 25.0
```

### patience（耐心值）

**作用**: PatientPruner 的耐心值。

**调优建议**:
- 低耐心: `patience=2`
- 中等耐心: `patience=3` （默认）
- 高耐心: `patience=5`

```bash
--patient_pruner_patience 3
```

## 💡 最佳实践

### 1. 根据场景选择策略

```bash
# 场景1: 初次探索 - 使用 Median（稳定）
python train_optuna.py --pruner median --n_trials 50

# 场景2: 时间紧迫 - 使用 Hyperband（高效）
python train_optuna.py --pruner hyperband --n_trials 100

# 场景3: 精细调优 - 使用 Percentile（可控）
python train_optuna.py --pruner percentile --percentile_pruner_percentile 20
```

### 2. 分阶段优化

```bash
# 阶段1: 快速筛选（100试验，激进剪枝）
python train_mbj_with_optuna.py \
    --n_trials 100 \
    --n_epochs 50 \
    --pruner hyperband \
    --output_dir stage1_results

# 阶段2: 精细搜索（50试验，中等剪枝）
python train_mbj_with_optuna.py \
    --n_trials 50 \
    --n_epochs 100 \
    --pruner median \
    --output_dir stage2_results

# 阶段3: 最终验证（20试验，不剪枝）
python train_mbj_with_optuna.py \
    --n_trials 20 \
    --n_epochs 200 \
    --pruner none \
    --output_dir stage3_results
```

### 3. 调整参数以匹配数据集

```bash
# 小数据集（训练快，收敛快）
python train_optuna.py \
    --pruner median \
    --pruner_startup_trials 3 \
    --pruner_warmup_steps 5

# 大数据集（训练慢，收敛慢）
python train_optuna.py \
    --pruner patient \
    --patient_pruner_patience 5 \
    --pruner_warmup_steps 20
```

### 4. 监控剪枝效果

查看有多少试验被剪枝：

```python
import optuna

study = optuna.load_study(study_name="your_study", storage="sqlite:///optuna_study.db")
completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

print(f"完成: {completed}, 剪枝: {pruned}")
print(f"剪枝率: {pruned/(completed+pruned)*100:.1f}%")
```

**理想剪枝率**:
- 20-40%: 正常（Median, Patient）
- 40-60%: 正常（Hyperband, Percentile）
- 60-80%: 正常（SuccessiveHalving）
- >90%: 可能过于激进，考虑调整参数

### 5. 组合使用

```bash
# 组合1: Hyperband + 并行优化
python train_optuna.py \
    --pruner hyperband \
    --n_trials 200 \
    --n_jobs 4

# 组合2: Patient + 更多 warmup
python train_mbj_with_optuna.py \
    --pruner patient \
    --patient_pruner_patience 5 \
    --pruner_warmup_steps 20
```

## 🔍 故障排除

### 问题 1: 所有试验都被剪枝

**原因**: Pruner 设置过于激进

**解决方案**:
```bash
# 增加 startup_trials
--pruner_startup_trials 10

# 增加 warmup_steps
--pruner_warmup_steps 20

# 或使用更保守的策略
--pruner patient
```

### 问题 2: 几乎没有试验被剪枝

**原因**: Pruner 设置过于保守

**解决方案**:
```bash
# 使用更激进的策略
--pruner hyperband

# 或降低百分位阈值
--pruner percentile --percentile_pruner_percentile 15.0
```

### 问题 3: 剪枝太早，错过好参数

**原因**: warmup_steps 不足

**解决方案**:
```bash
# 增加预热步数
--pruner_warmup_steps 30

# 或使用 PatientPruner
--pruner patient --patient_pruner_patience 5
```

## 📚 参考

- [Optuna Pruning 文档](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
- [Hyperband 论文](https://arxiv.org/abs/1603.06560)
- [Optuna 最佳实践](https://optuna.readthedocs.io/en/stable/tutorial/index.html)

---

**提示**: 如果不确定使用哪种策略，从 **MedianPruner**（默认）开始总是一个安全的选择！
