# 测试集划分问题修复指南

## 🔴 问题描述

你的评估结果显示测试集划分存在问题：

```
训练时测试集MAE: 0.2478
评估时0%遮挡MAE: 0.0859  ← 比训练时好65%！
```

**这表明评估使用了错误的测试集**，可能的原因：
1. 预处理时使用了不同的随机种子
2. 预处理时使用了不同的分割比例
3. 数据泄露（测试集包含了部分训练数据）

## ✅ 解决方案

我已经创建了一个自动修复脚本 `fix_test_split.py`，它会：

1. ✓ 在训练目录中查找保存的测试集ID
2. ✓ 从预处理数据中筛选出正确的测试集
3. ✓ 保存修复后的测试集
4. ✓ 生成诊断报告

### 步骤1: 运行修复脚本

```bash
cd /public/home/ghzhang/11.23

python fix_test_split.py \
    --training_dir /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set
```

### 步骤2: 检查修复结果

脚本会输出：

```
================================================================================
测试集划分修复
================================================================================

训练目录: /path/to/training/output
预处理数据目录: /path/to/preprocessed_data

================================================================================
步骤1: 查找训练目录中的测试集ID
================================================================================

✓ 找到 2 个可能的文件:
  - id_test.json: /path/to/training/output/id_test.json
  - test_ids.json: /path/to/training/output/test_ids.json

================================================================================
步骤2: 加载测试集ID
================================================================================

尝试从 id_test.json 加载...
✓ 成功加载 1234 个测试集ID
  前5个ID: ['JVASP-1', 'JVASP-100', 'JVASP-1001', ...]

================================================================================
步骤3: 加载预处理数据
================================================================================

✓ 加载 train 集: 8888 个样本
✓ 加载 val 集: 1111 个样本
✓ 加载 test 集: 1111 个样本

================================================================================
测试集一致性检查
================================================================================

训练测试集ID数量: 1234
预处理测试集ID数量: 1111

共同ID数量: 800 (64.8%)
仅在训练中: 434
仅在预处理中: 311

⚠️ 测试集不一致！这就是为什么评估结果不准确

================================================================================
步骤4: 筛选正确的测试集
================================================================================

总数据量: 11110 个样本
目标测试集ID数量: 1234 个

✓ 成功匹配: 1234 个样本
✓ 缺失ID: 0 个

================================================================================
步骤5: 保存正确的测试集
================================================================================

✓ 正确的测试集已保存到: ./corrected_test_set/test.pkl
  样本数: 1234
✓ 测试集ID已保存到: ./corrected_test_set/test_ids.json
✓ 修复信息已保存到: ./corrected_test_set/correction_info.json
```

### 步骤3: 使用修复后的测试集进行评估

现在使用 `--test_data` 参数指定修复后的测试集：

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --test_data ./corrected_test_set/test.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output
```

**验证结果**：检查 0% 遮挡的 MAE 是否接近训练时的测试集 MAE (0.2478)

### 步骤4: 运行完整评估

如果步骤3的结果正确（0% MAE ≈ 0.2478），继续运行完整评估：

```bash
# 创建批处理脚本
cat > run_corrected_masking_eval.sh << 'EOF'
#!/bin/bash

CHECKPOINT="/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt"
TEST_DATA="./corrected_test_set/test.pkl"
PREPROCESSED_DIR="/public/home/ghzhang/preprocessed_data"
DATASET="jarvis"
PROPERTY="mbj_bandgap"
OUTPUT_DIR="./masking_evaluation_corrected"

strategies=("random_token" "random_word" "random_chunk" "sentence" "keep_keywords")

for strategy in "${strategies[@]}"; do
    echo "======================================"
    echo "评估策略: $strategy"
    echo "======================================"

    python evaluate_text_masking.py \
        --checkpoint "$CHECKPOINT" \
        --test_data "$TEST_DATA" \
        --preprocessed_dir "$PREPROCESSED_DIR" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --masking_strategy "$strategy" \
        --output_dir "$OUTPUT_DIR/$strategy" \
        --batch_size 64 \
        --seed 42
done

echo "======================================"
echo "生成对比报告"
echo "======================================"

python compare_masking_strategies.py --input_dir "$OUTPUT_DIR"

echo ""
echo "✓ 评估完成！"
echo "结果: $OUTPUT_DIR/strategy_comparison_report.txt"
EOF

chmod +x run_corrected_masking_eval.sh

# 运行
./run_corrected_masking_eval.sh
```

---

## 🔍 故障排除

### 情况1: 训练目录中没有测试集ID文件

如果脚本输出：

```
✗ 训练目录中没有找到测试集ID文件
```

**解决方法**：

#### 方法A: 查找训练日志中的随机种子和分割比例

```bash
# 查找训练日志
ls /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/*.log

# 查看日志开头
head -200 /path/to/training.log | grep -E "seed|split|train_size|val_size|test_size"
```

如果找到seed和split参数（例如：seed=42, train:val:test=0.8:0.1:0.1），使用相同参数重新预处理：

```bash
python preprocess_dataset.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --target_property mbj_bandgap \
    --output_dir /public/home/ghzhang/preprocessed_data_fixed \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --random_seed 42 \
    --cutoff 8.0 \
    --max_neighbors 12
```

#### 方法B: 查找训练脚本

```bash
# 查找训练脚本
find /public/home/ghzhang/crysmmnet-main-2 -name "train*.sh" -o -name "train*.py"

# 查看脚本内容
cat /path/to/train_script.sh
```

从脚本中找到seed和split参数，然后重新预处理。

#### 方法C: 使用训练时的数据文件

训练时可能保存了 `train_loader.pkl`, `val_loader.pkl`, `test_loader.pkl`：

```bash
# 查找
ls /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/*loader.pkl

# 如果找到test_loader.pkl，直接使用
python evaluate_text_masking.py \
    --checkpoint /path/to/best_test_model.pt \
    --test_data /path/to/test_loader.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output
```

### 情况2: 匹配到的样本数量不足

如果脚本输出：

```
✓ 成功匹配: 800 个样本
⚠ 缺失ID: 434 个
  前5个缺失ID: ['JVASP-1', 'JVASP-100', ...]
```

**原因**：预处理数据不完整，缺少部分样本

**解决方法**：使用完整的原始数据重新预处理

```bash
python preprocess_dataset.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --target_property mbj_bandgap \
    --output_dir /public/home/ghzhang/preprocessed_data_complete \
    --random_seed 42 \
    --cutoff 8.0 \
    --max_neighbors 12
```

然后重新运行fix_test_split.py。

### 情况3: 修复后评估结果仍然不对

如果使用修复后的测试集，0% 遮挡的 MAE 仍然与训练时差距很大：

```
训练测试集MAE: 0.2478
修复后0%遮挡MAE: 0.0859  ← 仍然不对
```

**可能原因**：

1. **模型checkpoint不对**：使用了错误的checkpoint
2. **数据格式问题**：预处理格式与训练时不一致
3. **模型代码变化**：评估时的模型代码与训练时不同

**解决方法**：

```bash
# 1. 确认checkpoint是否正确
python inspect_checkpoint.py /path/to/best_test_model.pt

# 查看checkpoint中记录的测试集MAE
# 应该在输出中看到: best_test_loss: 0.2478

# 2. 检查是否有模型加载警告
python evaluate_text_masking.py ... 2>&1 | grep -i "warning\|unexpected\|missing"

# 3. 如果有很多unexpected/missing keys，可能需要使用训练时的模型代码
```

---

## 📊 预期结果

修复后，你应该看到：

### 快速测试输出

```
================================================================================
文本遮挡鲁棒性评估
  策略: random_token
  遮挡率: [0.0, 0.5, 1.0]
================================================================================

遮挡率 0.0%
  MAE:  0.2501  ← 应该接近 0.2478
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

✓ 结果已保存到: ./test_output
```

**关键验证**：0% 遮挡的 MAE 应该在 0.24-0.26 之间（接近训练时的 0.2478）

### 如果验证通过

0% 遮挡 MAE ≈ 0.2478 ✓
- 说明测试集正确
- 可以继续运行完整评估
- 评估结果可信

### 如果验证失败

0% 遮挡 MAE 与 0.2478 差距 > 10%
- 测试集仍然不对
- 需要进一步诊断
- 参考"故障排除"部分

---

## 🎯 快速参考

### 完整流程（一键运行）

```bash
# 1. 修复测试集
python fix_test_split.py \
    --training_dir /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set

# 2. 快速验证
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --test_data ./corrected_test_set/test.pkl \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output

# 3. 检查结果
cat ./test_output/text_masking_report_random_token.txt

# 4. 如果0% MAE ≈ 0.2478，运行完整评估
./run_corrected_masking_eval.sh
```

---

## 📞 需要帮助？

如果仍然无法解决，请提供：

1. **fix_test_split.py的完整输出**
2. **训练目录的文件列表**：
   ```bash
   ls -lh /path/to/training/output/
   ```
3. **训练日志（如果有）**：
   ```bash
   head -200 /path/to/training.log
   ```
4. **评估输出**：
   ```bash
   cat ./test_output/text_masking_report_random_token.txt
   ```

---

**总结**：这个问题很关键，必须使用正确的测试集才能得到准确的评估结果！按照上述步骤操作即可解决。🎯
