# MBJ Band Gap 模型文本遮挡评估指南

## 📋 概述

本指南专门用于评估你的 MBJ Band Gap 模型在文本信息缺失情况下的鲁棒性。

**模型信息**：
- Checkpoint: `/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt`
- 数据集: JARVIS
- 属性: mbj_bandgap

## 🚀 快速开始

### 前置要求

1. **需要的文件**（将这些文件复制到你的服务器工作目录）：
   - `evaluate_text_masking.py` - 主评估脚本
   - `compare_masking_strategies.py` - 策略对比脚本
   - `run_masking_eval_mbj_bandgap.sh` - 批量运行脚本
   - `quick_test_masking.sh` - 快速测试脚本

2. **预处理数据**：
   评估脚本需要预处理的数据。预处理数据应该位于如下结构：
   ```
   /public/home/ghzhang/preprocessed_data/
     jarvis/
       mbj_bandgap/
         train.pkl
         val.pkl
         test.pkl  ← 评估使用这个
   ```

   **如果你还没有预处理数据**，有两个选择：

   **选项A（推荐）**：使用现有的预处理脚本生成
   ```bash
   # 在你的服务器上运行
   python preprocess_dataset.py \
       --dataset jarvis \
       --property mbj_bandgap \
       --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
       --output_dir /public/home/ghzhang/preprocessed_data
   ```

   **选项B**：如果你的训练目录中已经有保存的数据加载器
   ```bash
   # 查找训练输出目录中的 test_loader.pkl
   find /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/ -name "*test*.pkl"
   ```

### 步骤1: 修改配置

编辑 `run_masking_eval_mbj_bandgap.sh` 和 `quick_test_masking.sh`，设置正确的预处理数据路径：

```bash
# 在两个脚本中找到这一行并修改
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"  # 改为你的实际路径
```

### 步骤2: 快速测试

首先运行快速测试以验证配置：

```bash
chmod +x quick_test_masking.sh
./quick_test_masking.sh
```

这会：
- 只测试 `random_token` 策略
- 只使用3个遮挡率 (0%, 50%, 100%)
- 快速验证所有配置是否正确

**如果成功**，你会看到：
```
✓ 测试成功！
结果保存在: ./test_masking_output
```

**如果失败**，请检查：
1. Checkpoint路径是否正确
2. 预处理数据路径是否正确
3. 预处理数据是否存在

### 步骤3: 完整评估

测试成功后，运行完整评估（测试所有5种策略，11个遮挡率）：

```bash
chmod +x run_masking_eval_mbj_bandgap.sh
./run_masking_eval_mbj_bandgap.sh
```

这会自动：
1. 测试所有5种遮挡策略：
   - `random_token`: 随机遮挡token
   - `random_word`: 随机遮挡单词
   - `random_chunk`: 随机遮挡文本块
   - `sentence`: 随机遮挡句子
   - `keep_keywords`: 只保留关键词

2. 每种策略测试11个遮挡率 (0%, 10%, 20%, ..., 100%)

3. 自动生成对比报告

**预计时间**：取决于测试集大小，大约 30-60 分钟

### 步骤4: 查看结果

评估完成后，结果保存在 `./masking_evaluation_mbj_bandgap/` 目录：

```
masking_evaluation_mbj_bandgap/
  random_token/
    text_masking_results_random_token.json      # 数值结果
    text_masking_report_random_token.txt        # 文本报告
    text_masking_analysis_random_token.png      # 可视化图表
  random_word/
    ...
  random_chunk/
    ...
  sentence/
    ...
  keep_keywords/
    ...

  # 对比结果（由 compare_masking_strategies.py 生成）
  strategy_comparison.png                        # 所有策略对比图
  strategy_comparison_report.txt                 # 鲁棒性排名报告
  strategy_comparison_summary.csv                # CSV汇总表
```

## 📊 结果解读

### 查看单个策略的报告

```bash
cat masking_evaluation_mbj_bandgap/random_token/text_masking_report_random_token.txt
```

示例输出：
```
================================================================================
文本遮挡鲁棒性评估报告
================================================================================

属性: mbj_bandgap
遮挡策略: random_token

Masking Ratio   MAE             RMSE            R²
--------------------------------------------------------------------------------
0.0%            0.1234          0.2345          0.9100
10.0%           0.1456          0.2567          0.8950
...

性能下降分析:
  基线 MAE (0% masking): 0.1234
  完全遮挡 MAE (100% masking): 0.4567
  MAE 增加: 270.18%

性能下降最快的区间:
  50.0% -> 60.0%
```

### 查看策略对比报告

```bash
cat masking_evaluation_mbj_bandgap/strategy_comparison_report.txt
```

这会显示：
- 鲁棒性排名（哪种策略对模型影响最大）
- 各策略的性能下降分析
- 关键发现和建议

### 可视化图表

使用任何图片查看器打开 `.png` 文件：

```bash
# 单个策略的4张子图
display masking_evaluation_mbj_bandgap/random_token/text_masking_analysis_random_token.png

# 所有策略对比
display masking_evaluation_mbj_bandgap/strategy_comparison.png
```

## 🔧 高级用法

### 只测试特定策略

如果你只想测试某个特定策略：

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --output_dir ./masking_eval_token_only
```

### 自定义遮挡率

如果你想测试特定的遮挡率：

```bash
python evaluate_text_masking.py \
    ... \
    --masking_ratios 0.0 0.25 0.5 0.75 1.0  # 只测试这5个遮挡率
```

### 调整批大小

如果遇到内存不足：

```bash
python evaluate_text_masking.py \
    ... \
    --batch_size 32  # 减小批大小
```

### 更改随机种子

测试结果的稳定性：

```bash
python evaluate_text_masking.py \
    ... \
    --seed 123  # 使用不同的随机种子
```

## ❓ 常见问题

### Q1: "找不到预处理数据"错误

**错误信息**：
```
FileNotFoundError: 找不到预处理数据: /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/test.pkl
```

**解决方案**：
1. 检查路径是否正确
2. 确认预处理数据已生成
3. 如果没有，运行数据预处理脚本

### Q2: "Checkpoint加载失败"错误

**可能原因**：
- Checkpoint文件损坏
- 模型配置不匹配

**解决方案**：
确认checkpoint中包含 `model_config`：
```python
import torch
ckpt = torch.load('your_checkpoint.pt')
print(ckpt.keys())  # 应该包含 'model_config'
```

### Q3: 评估速度太慢

**解决方案**：
1. 减小批大小（可能稍慢，但用更少内存）
2. 减少遮挡率数量：`--masking_ratios 0.0 0.3 0.5 0.7 1.0`
3. 先测试单个策略

### Q4: 如何解读鲁棒性分数？

**MAE增长率解读**：
- < 50%: 优秀的鲁棒性 ⭐⭐⭐
- 50-100%: 良好的鲁棒性 ⭐⭐
- 100-200%: 一般的鲁棒性 ⭐
- \> 200%: 较弱的鲁棒性

**示例**：
如果 `random_token` 的 MAE增长率是 150%，意味着完全遮挡文本后，MAE变为原来的2.5倍。

## 📝 示例工作流程

完整的评估流程：

```bash
# 1. 准备工作目录
cd /public/home/ghzhang/text_masking_evaluation
mkdir -p results

# 2. 复制必要文件
cp /path/to/evaluate_text_masking.py .
cp /path/to/compare_masking_strategies.py .
cp /path/to/run_masking_eval_mbj_bandgap.sh .
cp /path/to/quick_test_masking.sh .

# 3. 修改配置（编辑脚本中的路径）
vim run_masking_eval_mbj_bandgap.sh
vim quick_test_masking.sh

# 4. 快速测试
./quick_test_masking.sh

# 5. 查看测试结果
cat test_masking_output/text_masking_report_random_token.txt

# 6. 如果测试成功，运行完整评估
./run_masking_eval_mbj_bandgap.sh

# 7. 查看完整结果
cat masking_evaluation_mbj_bandgap/strategy_comparison_report.txt

# 8. 下载图表到本地查看
scp user@server:~/text_masking_evaluation/masking_evaluation_mbj_bandgap/*.png ./local_dir/
```

## 🎯 预期结果分析

对于你的 MBJ Band Gap 模型（使用 middle fusion + fine-grained attention），我们期望：

1. **基线性能** (0% masking):
   - 应该与你报告的测试集最佳性能一致

2. **低遮挡率** (10-30%):
   - 性能应该轻微下降
   - 如果下降很大，说明模型过度依赖每个token

3. **中等遮挡率** (40-60%):
   - 会有明显性能下降
   - 关键区域：观察下降的速度

4. **高遮挡率** (70-90%):
   - 性能明显下降
   - 但仍应保持一定预测能力（结构信息补偿）

5. **完全遮挡** (100%):
   - 完全依赖结构信息
   - 性能下降到最低点

**关键指标**：
- 如果 MAE 增长率 < 100%，说明模型有良好的鲁棒性
- 比较不同策略，找出最敏感的信息类型
- `keep_keywords` 策略可以告诉你关键词是否足够

## 📚 更多信息

详细的使用指南和理论说明，请参考：
- `TEXT_MASKING_EVALUATION_GUIDE.md` - 完整评估指南
- `evaluate_text_masking.py` - 源代码（包含详细注释）

## ✅ 检查清单

在运行之前，确认：

- [ ] 已将所有必要文件复制到服务器
- [ ] Checkpoint路径正确
- [ ] 预处理数据已准备好
- [ ] 已修改脚本中的路径配置
- [ ] 已运行快速测试并成功
- [ ] 有足够的磁盘空间（建议 > 5GB）
- [ ] 有足够的GPU内存（建议 > 8GB）

## 🚧 故障排除

如果遇到问题：

1. **检查日志**：评估脚本会输出详细的错误信息
2. **逐步测试**：先运行 `quick_test_masking.sh`
3. **减小规模**：使用更少的遮挡率和更小的批大小
4. **验证数据**：确认预处理数据可以正常加载

需要帮助？检查脚本输出的错误信息，或者查看 `TEXT_MASKING_EVALUATION_GUIDE.md` 中的常见问题部分。

祝评估顺利！🎉
