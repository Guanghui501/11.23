# 模型加载键不匹配问题解决方案

## 🔴 新错误信息

```
RuntimeError: Error(s) in loading state_dict for ALIGNN:
	Unexpected key(s) in state_dict: "fine_grained_attention.atom_position_embedding.weight".
```

## 📋 问题分析

这个错误表明：
- ✅ Checkpoint中**有** `fine_grained_attention.atom_position_embedding.weight` 参数
- ❌ 当前模型代码中**没有**这个参数

**原因**：训练时的模型代码与当前评估时的模型代码**不一致**。

可能的情况：
1. 模型代码被更新了（移除了某些参数）
2. 使用了不同版本的模型代码
3. 训练时使用的是旧版本/自定义版本的模型

## ✅ 已修复！

我已经更新了 `evaluate_text_masking.py`，现在会**自动处理**这种情况。

### 新的加载策略

```python
# 策略1: 首先尝试严格加载（所有键必须完全匹配）
try:
    model.load_state_dict(checkpoint['model'], strict=True)
    print("✓ 模型加载成功（严格匹配）")

# 策略2: 如果失败，使用宽松加载（允许部分不匹配）
except RuntimeError:
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model'],
        strict=False  # ← 允许不匹配
    )
    print("✓ 模型加载成功（宽松匹配）")
```

### 输出示例

现在运行评估时，你会看到：

```
加载模型: /path/to/best_test_model.pt
✓ 从checkpoint加载config
⚠ 严格加载失败，尝试宽松加载...
   原因: Error(s) in loading state_dict for ALIGNN:
	Unexpected key(s) in state_dict: "fine_grained_attention.atom_position_embedding.weight".

⚠ Checkpoint中有 1 个额外的键（将被忽略）:
   - fine_grained_attention.atom_position_embedding.weight

✓ 模型加载成功（宽松匹配）
  注意: 部分参数不匹配，但已成功加载大部分权重

✓ 模型加载成功

加载tokenizer...
...
```

## 🔍 详细说明

### Unexpected Keys（额外的键）

**定义**: Checkpoint中有，但当前模型代码中没有

**影响**: 这些参数会被**忽略**，不会加载到模型中

**示例**: `fine_grained_attention.atom_position_embedding.weight`

**是否安全**: 通常是安全的，只要额外的键不是核心参数

### Missing Keys（缺失的键）

**定义**: 当前模型代码中有，但checkpoint中没有

**影响**: 这些参数会使用**随机初始化**

**是否安全**:
- ❌ 如果缺失的是核心参数（如主干网络权重）→ 不安全，会导致评估结果错误
- ✅ 如果缺失的是新添加的参数 → 可能安全，取决于具体参数

**保护机制**: 如果缺失键 > 10个，脚本会拒绝继续并抛出错误

## 🚀 现在你可以运行评估了

从仓库获取最新代码：

```bash
cd /public/home/ghzhang/11.23
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8
```

然后直接运行评估：

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output
```

## ⚠️ 注意事项

### 1. 检查警告信息

运行评估时，注意查看：
- 有多少个 unexpected keys？
- 有多少个 missing keys？

**一般规则**：
- Unexpected keys < 5 个：通常安全
- Missing keys = 0 个：最理想
- Missing keys > 10 个：危险，会被脚本拒绝

### 2. 验证评估结果

如果有键不匹配，建议：
1. 先在小数据集上测试
2. 对比基线性能（0% masking）与训练时的测试集性能
3. 如果差异很大（>10%），说明加载有问题

**示例验证**：
```
训练时测试集 MAE: 0.2578
评估时 0% masking MAE: 0.2601  ← 接近，说明加载正常
评估时 0% masking MAE: 0.8888  ← 差距大，说明加载有问题
```

### 3. 如果担心结果准确性

**最安全的方法**：使用训练时的模型代码进行评估

如果你有训练时使用的模型代码（`alignn.py`），可以：

```bash
# 1. 备份当前模型代码
cp models/alignn.py models/alignn_new.py

# 2. 使用训练时的模型代码
cp /path/to/training/alignn.py models/alignn.py

# 3. 运行评估
python evaluate_text_masking.py ...

# 4. 恢复新版本代码
cp models/alignn_new.py models/alignn.py
```

## ❓ 常见问题

### Q1: 为什么会出现 atom_position_embedding？

**A**: 这可能是训练时模型的一个特性：
- 细粒度注意力模块添加了位置嵌入
- 用于编码原子在晶体结构中的位置信息
- 后来的模型版本可能移除了这个参数

### Q2: 忽略 atom_position_embedding 会影响评估吗？

**A**: 影响应该不大，因为：
- 这只是一个附加的位置编码参数
- 模型的主要参数（嵌入层、注意力层等）都已正确加载
- 如果担心，可以对比 0% masking 的结果与训练时的测试集 MAE

### Q3: 如何确保评估结果可靠？

**A**: 检查以下几点：
1. **基线性能**：0% masking 的 MAE 应该接近训练时的测试集 MAE
2. **Missing keys = 0**：没有缺失的核心参数
3. **Unexpected keys < 5**：额外的参数不多

### Q4: 什么情况下应该停止评估？

**A**: 如果出现以下情况，应该重新检查：
- Missing keys > 10 个（脚本会自动拒绝）
- 0% masking 的性能与训练时差距 > 20%
- 大量的 unexpected keys（> 50个）

## 🎯 推荐工作流程

```bash
# 1. 获取最新代码
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 2. 快速测试（只测试3个遮挡率）
python evaluate_text_masking.py \
    --checkpoint <your_checkpoint> \
    --preprocessed_dir <preprocessed_dir> \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output

# 3. 检查输出
cat test_output/text_masking_report_random_token.txt

# 4. 验证基线性能（0% masking的MAE）
# 应该接近训练时的测试集MAE

# 5. 如果看起来正常，运行完整评估
for strategy in random_token random_word random_chunk sentence keep_keywords; do
    python evaluate_text_masking.py \
        --checkpoint <your_checkpoint> \
        --preprocessed_dir <preprocessed_dir> \
        --dataset jarvis \
        --property mbj_bandgap \
        --masking_strategy $strategy \
        --output_dir ./masking_eval/$strategy
done

# 6. 生成对比报告
python compare_masking_strategies.py --input_dir ./masking_eval
```

## 📊 你的具体情况

根据检查结果：
- ✅ Checkpoint有配置（在 `config` 字段）
- ⚠️ 有1个额外的键：`fine_grained_attention.atom_position_embedding.weight`
- ✅ 训练时测试集 MAE: **0.2578**

**建议**：
1. 运行快速测试
2. 检查 0% masking 的 MAE 是否接近 0.2578
3. 如果接近，说明加载正常，可以继续完整评估
4. 如果差距很大，需要进一步调查

## 🆘 如果仍然失败

如果更新后仍然失败，请提供：
1. 完整的错误日志
2. 输出中显示的 unexpected keys 和 missing keys
3. 0% masking 的评估结果（如果能运行到那一步）

## 📚 相关文档

- **QUICK_FIX_GUIDE.md** - Checkpoint配置问题快速修复
- **CHECKPOINT_CONFIG_TROUBLESHOOTING.md** - Checkpoint配置完整故障排除
- **README_COMPLETE_WORKFLOW.md** - 完整评估工作流程

---

**总结**: 现在评估脚本会自动处理模型代码不匹配的问题。只要不匹配的参数不是核心参数，评估就可以正常进行！🎉
