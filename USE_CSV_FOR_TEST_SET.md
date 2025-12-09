# 使用Predictions CSV文件提取正确的测试集

## ✅ 你已经找到了解决方案！

你提供的CSV文件：
```
/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv
```

这个文件包含了训练时使用的**所有测试集样本的ID**，格式：
```csv
id,target,prediction
14410, 7.730000, 7.682570
17680, 3.197000, 3.306344
10280, 0.000000, 0.000000
...
```

使用这个文件，我们可以**100%准确地**重建训练时的测试集！

---

## 🚀 方法1: 一键运行（推荐）

我已经创建了一个自动化脚本，它会：
1. ✅ 从CSV提取测试集ID
2. ✅ 从预处理数据中筛选对应样本
3. ✅ 快速验证（0%, 50%, 100%遮挡）
4. ✅ 运行完整评估（5种策略）
5. ✅ 生成对比报告

**在你的服务器上运行：**

```bash
cd /public/home/ghzhang/11.23

# 拉取最新代码
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 一键运行
./quick_extract_and_eval.sh
```

### 输出示例

```
========================================
步骤1: 从CSV提取正确的测试集
========================================

读取CSV文件: /public/home/.../predictions_best_test_model_test.csv

CSV文件信息:
  列名: ['id', 'target', 'prediction']
  行数: 1234

✓ 成功提取 1234 个测试集ID
✓ 加载 train 集: 8888 个样本
✓ 加载 val 集: 1111 个样本
✓ 加载 test 集: 1111 个样本

✓ 成功匹配: 1234 个样本 (100.0%)
✓ 测试集已保存: ./corrected_test_set/test.pkl

========================================
步骤2: 快速验证（0%, 50%, 100%遮挡）
========================================

训练时测试集MAE: 0.2478
评估时0%遮挡MAE: 0.2501
差异: 0.9%

✓ 验证通过！测试集正确

========================================
步骤3: 运行完整评估（5种策略）
========================================

[运行所有5种策略的评估...]

========================================
✓ 所有评估完成！
========================================

结果位置:
  • 测试集: ./corrected_test_set/test.pkl
  • 评估结果: ./masking_eval_corrected/
  • 对比报告: ./masking_eval_corrected/strategy_comparison_report.txt
```

---

## 🔧 方法2: 分步运行（调试/自定义）

如果你想更精细地控制每一步，可以手动运行：

### 步骤1: 提取测试集

```bash
python extract_test_set_from_csv.py \
    --predictions_csv /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set
```

**这个脚本会：**
- ✅ 读取CSV文件中的所有测试集ID
- ✅ 从预处理数据（train/val/test）中查找这些ID
- ✅ 自动处理ID格式差异（如 "14410" vs "JVASP-14410"）
- ✅ 验证目标值是否匹配
- ✅ 保存提取的测试集到 `./corrected_test_set/test.pkl`
- ✅ 保存提取信息到 `extraction_info.json`

### 步骤2: 快速验证

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --test_data ./corrected_test_set/test.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./quick_validation
```

**检查验证结果：**

```bash
cat ./quick_validation/text_masking_report_random_token.txt
```

**期望看到：**
```
遮挡率 0.0%
  MAE:  0.2478 ± 0.02  ← 应该非常接近训练时的 0.2478
  RMSE: 0.5xxx
  R²:   0.8xxx
```

✅ **如果 0% MAE ≈ 0.2478** → 测试集正确，继续下一步
❌ **如果 0% MAE 仍然 ≈ 0.08** → 需要检查checkpoint是否匹配

### 步骤3: 运行完整评估

```bash
# 评估所有5种策略
for strategy in random_token random_word random_chunk sentence keep_keywords; do
    python evaluate_text_masking.py \
        --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
        --test_data ./corrected_test_set/test.pkl \
        --preprocessed_dir /public/home/ghzhang/preprocessed_data \
        --dataset jarvis \
        --property mbj_bandgap \
        --masking_strategy $strategy \
        --output_dir ./masking_eval_corrected/$strategy \
        --batch_size 64 \
        --seed 42
done
```

### 步骤4: 生成对比报告

```bash
python compare_masking_strategies.py --input_dir ./masking_eval_corrected
```

**查看结果：**

```bash
# 查看对比报告
cat ./masking_eval_corrected/strategy_comparison_report.txt

# 查看可视化图表（如果有图形界面）
# open ./masking_eval_corrected/strategy_comparison.png

# 查看各个策略的详细报告
cat ./masking_eval_corrected/random_token/text_masking_report_random_token.txt
cat ./masking_eval_corrected/random_word/text_masking_report_random_word.txt
# ... 等等
```

---

## 🔍 提取脚本的智能功能

`extract_test_set_from_csv.py` 包含以下智能功能：

### 1. 自动ID格式匹配

处理多种ID格式差异：

```python
# CSV中的ID     →  预处理数据中的ID
14410          →  14410 (直接匹配)
14410          →  JVASP-14410 (添加前缀)
JVASP-14410    →  14410 (去掉前缀)
```

### 2. 目标值验证

验证CSV中的target值与预处理数据中的target是否一致：

```
样本1 (ID: 14410): PKL=7.730000, CSV=7.730000 ✓
样本2 (ID: 17680): PKL=3.197000, CSV=3.197000 ✓
...
✓ 前10个样本的目标值完全匹配
```

### 3. 详细统计

```json
{
  "source_csv": "/path/to/predictions_best_test_model_test.csv",
  "csv_samples": 1234,
  "matched_samples": 1234,
  "missing_samples": 0,
  "match_rate": 100.0,
  "targets_verified": true
}
```

---

## ⚠️ 重要提醒：Checkpoint匹配

你提供的两个路径：

**Predictions CSV来自：**
```
/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/
output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/
predictions_best_test_model_test.csv
```
👉 **只有middle fusion**

**Checkpoint来自：**
```
/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/
middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/
best_test_model.pt
```
👉 **middle + fine-grained fusion**

### ❓ 这两个模型不同！

**建议：使用匹配的checkpoint**

#### 选项A: 使用与CSV匹配的checkpoint（推荐用于验证）

```bash
# 使用onlymiddle的checkpoint（如果存在）
CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/best_test_model.pt"

# 这样0% MAE应该非常接近CSV中计算出的MAE
```

#### 选项B: 使用middle+fine的checkpoint评估

```bash
# 使用你之前指定的checkpoint
CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"

# 但这样的话，0% MAE可能与0.2478不匹配（因为是不同的模型）
```

**如果使用middle+fine checkpoint**，你需要找到对应的predictions CSV：

```bash
# 查找middle+fine模型的predictions文件
ls /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/predictions*.csv
```

---

## 📊 预期最终结果

使用正确的测试集和匹配的checkpoint后，你应该看到：

### 对比报告示例

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
   - 基线MAE: 0.2478
   - 100%遮挡MAE: 0.3521
   - 性能下降: 42.1%

2. sentence
   - 基线MAE: 0.2478
   - 100%遮挡MAE: 0.4123
   - 性能下降: 66.4%

3. random_word
   - 基线MAE: 0.2478
   - 100%遮挡MAE: 0.4567
   - 性能下降: 84.3%

4. random_chunk
   - 基线MAE: 0.2478
   - 100%遮挡MAE: 0.5012
   - 性能下降: 102.3%

5. random_token
   - 基线MAE: 0.2478
   - 100%遮挡MAE: 0.5234
   - 性能下降: 111.2%

--------------------------------------------------------------------------------
关键发现
--------------------------------------------------------------------------------

✓ 所有策略的0%遮挡MAE都是 0.2478（与训练测试集一致）
✓ keep_keywords策略最鲁棒（保留关键化学信息）
✓ random_token策略最激进（破坏性最强）
✓ 模型在50%文本遮挡下仍保持合理性能（MAE < 0.35）
```

---

## 🎯 快速检查清单

在开始之前，确认：

- [ ] CSV文件存在且可读
  ```bash
  head /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv
  ```

- [ ] 预处理数据存在
  ```bash
  ls /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/
  # 应该看到: train.pkl, val.pkl, test.pkl
  ```

- [ ] Checkpoint文件存在
  ```bash
  ls -lh /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt
  ```

- [ ] 在正确的工作目录
  ```bash
  cd /public/home/ghzhang/11.23
  pwd
  ```

- [ ] 已拉取最新代码
  ```bash
  git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8
  ```

---

## 🚀 现在就开始！

**最简单的方式 - 一条命令：**

```bash
cd /public/home/ghzhang/11.23 && \
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8 && \
./quick_extract_and_eval.sh
```

脚本会自动处理一切，并在最后显示所有结果的位置。

**Good luck! 🎉**
