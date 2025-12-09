# 文本遮挡评估 - 快速开始指南

## 🎯 目标

评估模型对文本信息缺失的鲁棒性，测试不同遮挡策略下的性能变化。

---

## 🚀 一键运行（最简单）

在你的服务器上执行以下命令：

```bash
# 进入工作目录
cd /public/home/ghzhang/11.23

# 拉取最新代码（包含所有修复）
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 激活 conda 环境
conda activate MatMMFuse

# 安装必要的包（只需运行一次）
./install_pydantic_settings.sh

# 一键运行完整评估
./quick_extract_and_eval.sh
```

**就这么简单！** 脚本会自动：
1. ✅ 修复环境问题（GLIBCXX）
2. ✅ 从 CSV 提取正确的测试集
3. ✅ 快速验证（0%, 50%, 100%）
4. ✅ 完整评估（5种策略）
5. ✅ 生成对比报告

---

## 📊 预期输出

### 步骤1: 提取测试集

```
========================================
步骤1: 从CSV提取正确的测试集
========================================

读取CSV文件: .../predictions_best_test_model_test.csv

CSV文件信息:
  列名: ['id', 'target', 'prediction']
  行数: 1234

✓ 成功提取 1234 个测试集ID
✓ 成功匹配: 1234 个样本 (100.0%)
✓ 测试集已保存: ./corrected_test_set/test.pkl
```

### 步骤2: 快速验证

```
========================================
步骤2: 快速验证（0%, 50%, 100%遮挡）
========================================

遮挡率 0.0%
  MAE:  0.2478  ← 与训练测试集完全一致 ✓
  RMSE: 0.5234
  R²:   0.8765

遮挡率 50.0%
  MAE:  0.3123
  RMSE: 0.6012
  R²:   0.8234

遮挡率 100.0%
  MAE:  0.4567
  RMSE: 0.7890
  R²:   0.7123

✓ 验证通过！测试集正确
```

### 步骤3: 完整评估

```
========================================
步骤3: 运行完整评估（5种策略）
========================================

评估策略: random_token
[进度条] 100%

评估策略: random_word
[进度条] 100%

评估策略: random_chunk
[进度条] 100%

评估策略: sentence
[进度条] 100%

评估策略: keep_keywords
[进度条] 100%
```

### 步骤4: 对比报告

```
========================================
✓ 所有评估完成！
========================================

结果位置:
  • 测试集: ./corrected_test_set/test.pkl
  • 评估结果: ./masking_eval_corrected/
  • 对比报告: ./masking_eval_corrected/strategy_comparison_report.txt
  • 可视化图表: ./masking_eval_corrected/strategy_comparison.png
```

---

## 📁 查看结果

### 查看对比报告

```bash
cat ./masking_eval_corrected/strategy_comparison_report.txt
```

示例输出：
```
================================================================================
文本遮挡策略对比报告
================================================================================

数据集: jarvis - mbj_bandgap
测试集大小: 1234 样本

--------------------------------------------------------------------------------
策略排名 (按鲁棒性，从好到差)
--------------------------------------------------------------------------------

1. keep_keywords
   基线MAE: 0.2478  |  100%遮挡MAE: 0.3521  |  性能下降: 42.1%

2. sentence
   基线MAE: 0.2478  |  100%遮挡MAE: 0.4123  |  性能下降: 66.4%

3. random_word
   基线MAE: 0.2478  |  100%遮挡MAE: 0.4567  |  性能下降: 84.3%

4. random_chunk
   基线MAE: 0.2478  |  100%遮挡MAE: 0.5012  |  性能下降: 102.3%

5. random_token
   基线MAE: 0.2478  |  100%遮挡MAE: 0.5234  |  性能下降: 111.2%
```

### 查看各策略详细结果

```bash
# random_token 策略
cat ./masking_eval_corrected/random_token/text_masking_report_random_token.txt

# random_word 策略
cat ./masking_eval_corrected/random_word/text_masking_report_random_word.txt

# 其他策略类似
```

### 查看可视化图表（如果有图形界面）

```bash
# 对比图
open ./masking_eval_corrected/strategy_comparison.png

# 各策略独立图表
open ./masking_eval_corrected/random_token/masking_curve_random_token.png
```

---

## 🔧 分步运行（自定义需求）

如果需要更精细的控制，可以分步运行：

### 步骤1: 提取测试集

```bash
python extract_test_set_from_csv.py \
    --predictions_csv /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set
```

### 步骤2: 评估单个策略

```bash
python evaluate_text_masking.py \
    --checkpoint /path/to/best_test_model.pt \
    --test_data ./corrected_test_set/test.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --output_dir ./my_output \
    --batch_size 64
```

### 步骤3: 评估所有策略

```bash
for strategy in random_token random_word random_chunk sentence keep_keywords; do
    python evaluate_text_masking.py \
        --checkpoint /path/to/best_test_model.pt \
        --test_data ./corrected_test_set/test.pkl \
        --preprocessed_dir /public/home/ghzhang/preprocessed_data \
        --dataset jarvis \
        --property mbj_bandgap \
        --masking_strategy $strategy \
        --output_dir ./my_output/$strategy
done
```

### 步骤4: 生成对比报告

```bash
python compare_masking_strategies.py --input_dir ./my_output
```

---

## ⚠️ 常见问题及解决方案

### 问题1: GLIBCXX_3.4.30 not found

**解决方案**：脚本已自动处理。如果仍有问题：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

详见：[FIX_GLIBCXX_ERROR.md](FIX_GLIBCXX_ERROR.md)

### 问题2: cannot import name 'Literal' from 'pydantic.typing'

**解决方案**：代码已修复为使用 `from typing import Literal`

详见：[FIX_PYDANTIC_ERROR.md](FIX_PYDANTIC_ERROR.md)

### 问题3: BaseSettings has been moved to pydantic-settings

**解决方案**：运行安装脚本
```bash
./install_pydantic_settings.sh
```

详见：[FIX_BASESETTINGS_ERROR.md](FIX_BASESETTINGS_ERROR.md)

### 问题4: 0% 遮挡 MAE 与训练时不匹配

**原因**：Checkpoint 与 CSV 不匹配

**解决方案**：确保使用配对的 checkpoint 和 CSV
- CSV 来自 onlymiddle 模型 → 使用 onlymiddle checkpoint
- CSV 来自 middle+fine 模型 → 使用 middle+fine checkpoint

详见：[USE_CSV_FOR_TEST_SET.md](USE_CSV_FOR_TEST_SET.md)

---

## 📚 完整文档列表

| 文档 | 说明 |
|------|------|
| **QUICK_START.md** (本文件) | 快速开始指南 |
| **SOLUTION_SUMMARY.md** | 问题解决方案总结 |
| **USE_CSV_FOR_TEST_SET.md** | CSV 提取测试集详细指南 |
| **FIX_TEST_SPLIT.md** | 测试集划分问题通用修复 |
| **FIX_GLIBCXX_ERROR.md** | GLIBCXX 库版本问题修复 |
| **FIX_PYDANTIC_ERROR.md** | Pydantic Literal 导入修复 |
| **FIX_BASESETTINGS_ERROR.md** | BaseSettings 导入修复 |

---

## 🛠️ 核心脚本说明

| 脚本 | 功能 |
|------|------|
| **quick_extract_and_eval.sh** | 一键运行完整流程 |
| **extract_test_set_from_csv.py** | 从 CSV 提取测试集 |
| **evaluate_text_masking.py** | 文本遮挡评估主脚本 |
| **compare_masking_strategies.py** | 策略对比分析 |
| **install_pydantic_settings.sh** | 安装 pydantic-settings |
| **run_extract_with_fix.sh** | 带环境修复的提取脚本 |

---

## 🎯 5种遮挡策略说明

1. **random_token**: 随机遮挡单个 token（最激进）
2. **random_word**: 随机遮挡完整单词
3. **random_chunk**: 随机遮挡连续文本块
4. **sentence**: 随机遮挡完整句子
5. **keep_keywords**: 只保留关键词（元素名称、晶体结构等，最保守）

---

## ✅ 验证成功的标志

运行成功后，你应该看到：

1. ✅ 0% 遮挡 MAE ≈ 0.2478（与训练时一致）
2. ✅ MAE 随遮挡率增加而增加
3. ✅ 生成了 5 个策略的完整报告
4. ✅ 生成了策略对比报告和图表

---

## 📞 需要帮助？

如果遇到问题：

1. **检查日志**：所有输出都会显示在终端
2. **查看文档**：根据错误信息查阅对应的 FIX_*.md 文档
3. **提供信息**：
   ```bash
   # 环境信息
   conda list | grep -E "pydantic|dgl|torch"

   # Python 版本
   python --version

   # 错误日志
   ./quick_extract_and_eval.sh 2>&1 | tee run.log
   ```

---

**祝评估顺利！🎉**

所有问题都已修复，现在可以直接运行了。
