# 文本遮挡鲁棒性评估指南

## 概述

`evaluate_text_masking.py` 脚本用于测试训练好的模型对文本信息缺失的鲁棒性。通过逐步增加文本遮挡率，观察模型预测性能（MAE、RMSE、R²）的变化趋势。

## 核心功能

### 1. 多种遮挡策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `random_token` | 随机遮挡单个token | 测试模型对词汇级信息缺失的鲁棒性 |
| `random_word` | 随机遮挡完整单词 | 测试对单词级信息缺失的鲁棒性 |
| `random_chunk` | 随机遮挡连续文本块 | 测试对连续信息片段缺失的鲁棒性 |
| `sentence` | 随机遮挡完整句子 | 测试对句子级信息缺失的鲁棒性 |
| `keep_keywords` | 只保留关键词（元素、结构术语） | 测试仅依靠关键信息的预测能力 |

### 2. 可配置的遮挡率

- 默认测试遮挡率：`0%, 10%, 20%, ..., 90%, 100%`
- 可自定义任意遮挡率组合
- 遮挡率 = 被遮挡的文本单元数量 / 总文本单元数量

### 3. 全面的性能指标

- **MAE (Mean Absolute Error)**: 平均绝对误差
- **RMSE (Root Mean Square Error)**: 均方根误差
- **R² (R-squared)**: 决定系数

### 4. 自动化报告生成

- **JSON结果文件**: 包含所有数值数据
- **文本报告**: 详细的统计分析和性能下降分析
- **可视化图表**: 4张子图展示不同指标随遮挡率的变化

## 使用方法

### 基础用法

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy random_token
```

### 测试不同遮挡策略

#### 1. 随机Token遮挡（推荐作为基准）

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --output_dir ./masking_eval_token
```

#### 2. 随机单词遮挡

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy random_word \
    --output_dir ./masking_eval_word
```

#### 3. 连续块遮挡（测试局部信息缺失）

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy random_chunk \
    --output_dir ./masking_eval_chunk
```

#### 4. 句子遮挡

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy sentence \
    --output_dir ./masking_eval_sentence
```

#### 5. 只保留关键词（最严格的测试）

```bash
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy keep_keywords \
    --output_dir ./masking_eval_keywords
```

### 自定义遮挡率

```bash
# 只测试几个关键遮挡率
python evaluate_text_masking.py \
    --checkpoint ./results/jarvis/formation_energy/best_model.pt \
    --preprocessed_dir ./preprocessed_data \
    --dataset jarvis \
    --property formation_energy \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.25 0.5 0.75 1.0
```

### 完整参数列表

```bash
python evaluate_text_masking.py \
    --checkpoint <模型checkpoint路径> \
    --preprocessed_dir <预处理数据目录> \
    --dataset <jarvis|mp> \
    --property <属性名称> \
    --masking_strategy <random_token|random_word|random_chunk|sentence|keep_keywords> \
    --masking_ratios <遮挡率列表> \
    --batch_size <批大小，默认64> \
    --output_dir <输出目录> \
    --seed <随机种子，默认42>
```

## 输出文件

评估完成后，会在输出目录生成以下文件：

### 1. JSON结果文件
```
text_masking_results_<strategy>.json
```
包含所有遮挡率下的数值结果：
```json
{
    "masking_ratios": [0.0, 0.1, 0.2, ...],
    "mae": [0.123, 0.156, 0.201, ...],
    "rmse": [0.234, 0.267, 0.312, ...],
    "r2": [0.91, 0.88, 0.83, ...],
    "property": "formation_energy",
    "strategy": "random_token"
}
```

### 2. 文本报告
```
text_masking_report_<strategy>.txt
```
包含：
- 完整的数值表格
- 性能下降分析
- 最敏感的遮挡率区间

示例输出：
```
================================================================================
文本遮挡鲁棒性评估报告
================================================================================

属性: formation_energy
遮挡策略: random_token

--------------------------------------------------------------------------------
Masking Ratio   MAE             RMSE            R²
--------------------------------------------------------------------------------
0.0%            0.1234          0.2345          0.9100
10.0%           0.1456          0.2567          0.8950
20.0%           0.1678          0.2789          0.8800
...

性能下降分析:
  基线 MAE (0% masking): 0.1234
  完全遮挡 MAE (100% masking): 0.4567
  MAE 增加: 270.18%

性能下降最快的区间:
  50.0% -> 60.0%
  MAE 变化: 0.2345 -> 0.3012
  梯度: 0.0667
```

### 3. 可视化图表
```
text_masking_analysis_<strategy>.png
```
包含4个子图：
1. **MAE vs Masking Ratio**: MAE随遮挡率变化
2. **RMSE vs Masking Ratio**: RMSE随遮挡率变化
3. **R² vs Masking Ratio**: R²随遮挡率变化
4. **All Metrics (Normalized)**: 所有指标归一化对比

## 结果解读

### 1. 理想的鲁棒性曲线

**良好的鲁棒性表现**：
- 低遮挡率（0-30%）：性能轻微下降
- 中等遮挡率（30-60%）：性能平缓下降
- 高遮挡率（60-90%）：性能明显下降
- 完全遮挡（100%）：完全依赖结构信息

### 2. 关键指标分析

#### MAE增长率
```
MAE增长率 = (MAE_遮挡 - MAE_基线) / MAE_基线 × 100%
```
- < 50%：优秀的鲁棒性
- 50-100%：良好的鲁棒性
- 100-200%：一般的鲁棒性
- > 200%：较弱的鲁棒性

#### R²保持率
```
R²保持率 = R²_遮挡 / R²_基线 × 100%
```
- > 80%：优秀
- 60-80%：良好
- 40-60%：一般
- < 40%：较弱

### 3. 不同策略的对比

建议依次测试所有5种策略，然后对比：

```python
# 批量运行所有策略
strategies = ['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords']
for strategy in strategies:
    os.system(f"""
    python evaluate_text_masking.py \
        --checkpoint ./results/jarvis/formation_energy/best_model.pt \
        --preprocessed_dir ./preprocessed_data \
        --dataset jarvis \
        --property formation_energy \
        --masking_strategy {strategy} \
        --output_dir ./masking_eval_{strategy}
    """)
```

然后对比各策略的结果，判断：
- 哪种信息缺失对模型影响最大？
- 模型最依赖哪类文本信息？
- 关键词是否足够支撑预测？

## 实际应用场景

### 1. 模型诊断
- 检测模型是否过度依赖文本信息
- 评估文本和结构信息的融合效果
- 发现模型对特定类型文本信息的敏感性

### 2. 对比实验
比较不同融合策略的鲁棒性：
```bash
# 方案1（门控融合）
python evaluate_text_masking.py \
    --checkpoint ./results/solution1/best_model.pt \
    ... \
    --output_dir ./masking_eval_solution1

# 方案2（单向注意力）
python evaluate_text_masking.py \
    --checkpoint ./results/solution2/best_model.pt \
    ... \
    --output_dir ./masking_eval_solution2
```

对比两个方案在文本遮挡下的表现，判断哪个方案更鲁棒。

### 3. 实际部署评估
如果实际应用中可能遇到不完整的文本描述：
- 测试模型在低质量文本下的可靠性
- 确定可接受的最低文本完整度
- 为用户提供文本质量要求指南

## 高级用法

### 1. 批量测试脚本

创建 `batch_masking_eval.sh`:
```bash
#!/bin/bash

CHECKPOINT="./results/jarvis/formation_energy/best_model.pt"
PREPROCESSED_DIR="./preprocessed_data"
DATASET="jarvis"
PROPERTY="formation_energy"

strategies=("random_token" "random_word" "random_chunk" "sentence" "keep_keywords")

for strategy in "${strategies[@]}"; do
    echo "Testing strategy: $strategy"
    python evaluate_text_masking.py \
        --checkpoint $CHECKPOINT \
        --preprocessed_dir $PREPROCESSED_DIR \
        --dataset $DATASET \
        --property $PROPERTY \
        --masking_strategy $strategy \
        --output_dir ./masking_eval_$strategy
done

echo "All strategies tested!"
```

运行：
```bash
chmod +x batch_masking_eval.sh
./batch_masking_eval.sh
```

### 2. 合并多个策略的结果

创建 `compare_strategies.py`:
```python
import json
import matplotlib.pyplot as plt

strategies = ['random_token', 'random_word', 'random_chunk', 'sentence', 'keep_keywords']
results = {}

for strategy in strategies:
    with open(f'./masking_eval_{strategy}/text_masking_results_{strategy}.json', 'r') as f:
        results[strategy] = json.load(f)

# 绘制对比图
plt.figure(figsize=(12, 6))
for strategy, data in results.items():
    plt.plot(data['masking_ratios'], data['mae'], 'o-', label=strategy, linewidth=2)

plt.xlabel('Text Masking Ratio', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('MAE Comparison Across Different Masking Strategies', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to strategy_comparison.png")
```

## 常见问题

### Q1: 评估速度太慢怎么办？
**A**: 减小批大小或只测试部分遮挡率：
```bash
--batch_size 128 \
--masking_ratios 0.0 0.3 0.5 0.7 1.0
```

### Q2: 如何确保结果可重复？
**A**: 使用固定的随机种子：
```bash
--seed 42
```

### Q3: 可以测试训练集或验证集吗？
**A**: 当前脚本只支持测试集。如需测试其他数据集，需要修改数据加载部分。

### Q4: 如何处理不同长度的文本？
**A**: 脚本自动处理。遮挡率是相对于每个样本的文本长度计算的。

### Q5: `keep_keywords` 策略如何定义关键词？
**A**: 自动识别：
- 化学元素符号（H, He, Li, ...）
- 晶体结构术语（cubic, tetragonal, hexagonal, ...）
- 可在 `TextMasker` 类中自定义

## 扩展阅读

### 相关研究
- **Ablation Studies**: 文本遮挡评估是消融研究的一种形式
- **Robustness Testing**: 评估模型在噪声和缺失数据下的表现
- **Multi-modal Fusion Analysis**: 分析不同模态的贡献度

### 进一步实验建议
1. **噪声注入**: 除了遮挡，还可以测试文本噪声（拼写错误、同义词替换等）
2. **梯度分析**: 分析不同文本位置对预测的贡献
3. **注意力可视化**: 查看模型在不同遮挡率下的注意力分布变化

## 总结

`evaluate_text_masking.py` 是一个强大的工具，用于：
- ✅ 系统评估模型对文本信息的依赖程度
- ✅ 测试模型在文本信息缺失情况下的鲁棒性
- ✅ 对比不同融合策略的效果
- ✅ 为模型改进提供数据驱动的insights

**建议的评估流程**：
1. 首先用 `random_token` 策略建立基准
2. 测试所有5种策略，找出最敏感的信息类型
3. 对比不同模型/方案的鲁棒性
4. 根据结果优化融合策略

祝评估顺利！ 🎯
