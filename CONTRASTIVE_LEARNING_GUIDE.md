# 对比学习调优指南

## 📊 对比学习在降低MAE中的作用

基于当前训练曲线和CKA分析，对比学习可以从以下几个方面帮助降低MAE：

### 1. 核心问题分析

**当前状态**：
- 训练MAE: ~3（低）
- 验证MAE: ~10（高）
- Train-Val Gap: ~3-4x（过拟合）
- CKA分析: text_fine差异大(0.29) → text_final差异消失(0.98)

**根本原因**：
1. 模型过拟合训练数据
2. 融合效果在最终层被"平滑"掉
3. 跨模态表示没有充分学习

**对比学习的解决方案**：
```
对比学习 → 强制保持模态差异 → 融合效果保持 → 更好的泛化 → MAE降低
```

---

## 🎯 对比学习的三大作用

### 作用1: 防止特征坍塌

**问题**：
- 不同模态的特征最终变得几乎相同（CKA=0.98）
- 融合机制的作用被抵消

**对比学习的作用**：
```python
# 对比损失鼓励同样本的不同模态表示接近，但保持差异性
contrastive_loss = -log(exp(sim(graph, text_pos) / τ) /
                        Σ exp(sim(graph, text_neg) / τ))
```

**期望效果**：
- CKA (fused): 0.98 → 0.85-0.92
- 保持适度的模态差异
- 融合效果不被后续层抵消

### 作用2: 正则化效果（降低过拟合）

**原理**：
```
传统训练：
Loss = MSE(prediction, target)
→ 只关注预测，容易过拟合

+ 对比学习：
Loss = MSE(prediction, target) + λ * Contrastive_Loss
→ 同时优化特征空间结构
→ 更鲁棒的表示
→ 更好的泛化性能
```

**期望效果**：
- 验证MAE: 10 → 7-8（降低20-30%）
- Train-Val Gap: 3-4x → 1.5-2x

### 作用3: 增强跨模态学习

**当前问题**：
- 融合层可能只是"形式上"的融合
- 没有真正学习到有效的跨模态关联

**对比学习的作用**：
- 显式地鼓励同一材料的图和文本表示对齐
- 同时与其他材料的表示区分开
- 学习更有意义的跨模态表示

---

## 🧪 实验方案

### 实验1: 启用对比学习（保守配置）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.1 \
    --contrastive_temperature 0.1 \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --epochs 300 \
    --n_early_stopping 100 \
    --output_dir ./with_contrastive_conservative
```

**参数说明**：
- `--use_contrastive_loss 1`: 启用对比学习
- `--contrastive_loss_weight 0.1`: 对比损失权重（较小，保守）
- `--contrastive_temperature 0.1`: 温度参数（控制分布锐度）

### 实验2: 增强对比学习（激进配置）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.3 \
    --contrastive_temperature 0.07 \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --output_dir ./with_contrastive_aggressive
```

**参数说明**：
- `--contrastive_loss_weight 0.3`: 更高的权重（更强的约束）
- `--contrastive_temperature 0.07`: 更低的温度（更锐利的分布）

### 实验3: 对比学习 + 最优正则化组合

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.08 \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --fine_grained_dropout 0.25 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --epochs 300 \
    --n_early_stopping 100 \
    --output_dir ./with_contrastive_best_combined
```

---

## 🔧 对比学习超参数调优

### 关键超参数

#### 1. `contrastive_loss_weight` (λ)

**作用**：控制对比学习的强度

| 值 | 效果 | 建议使用场景 |
|----|------|-------------|
| 0.05-0.1 | 轻微约束 | 模型已经表现不错，只是微调 |
| 0.1-0.2 | 中等约束 | **推荐起点** - 平衡预测和对比 |
| 0.2-0.5 | 强约束 | 过拟合严重，需要强正则化 |
| >0.5 | 过强 | 可能损害预测性能 |

**调优策略**：
```python
# 从小到大逐步尝试
weights_to_try = [0.05, 0.1, 0.15, 0.2, 0.3]
```

#### 2. `contrastive_temperature` (τ)

**作用**：控制对比分布的锐度

| 值 | 效果 | 特点 |
|----|------|------|
| 0.05 | 非常锐利 | 强约束，可能过于严格 |
| 0.07-0.1 | 锐利 | **推荐** - 清晰的正负样本区分 |
| 0.1-0.2 | 适中 | 平衡的约束 |
| >0.2 | 平滑 | 约束较弱 |

**直观理解**：
```
温度低 (0.07) → 只有非常相似的才算正样本 → 更严格
温度高 (0.2) → 相似度要求放松 → 更宽松
```

**调优策略**：
- 如果验证loss震荡 → 提高温度
- 如果对比效果不明显 → 降低温度

---

## 📈 预期效果

### 1. 训练曲线变化

**无对比学习**：
```
Epoch   Train MAE   Val MAE   Gap
50      3.2         10.5      3.3x
100     2.8         10.2      3.6x  ← 过拟合
200     2.5         10.3      4.1x  ← 更严重
```

**有对比学习（预期）**：
```
Epoch   Train MAE   Val MAE   Gap
50      4.0         9.5       2.4x  ← 训练MAE略高
100     3.5         8.2       2.3x  ← 验证MAE降低！
200     3.2         7.8       2.4x  ← 持续改善
```

**关键观察**：
- ✅ 验证MAE降低 20-30%
- ✅ Train-Val Gap缩小到 2-2.5x
- ⚠️ 训练MAE可能略微上升（正常，说明正则化起作用）

### 2. CKA分析变化

**无对比学习（当前）**：
```
Stage           CKA Score   状态
text_fine       0.287       ✓ 差异大
text_final      0.976       ✗ 差异消失
fused           0.980       ✗ 几乎相同
```

**有对比学习（预期）**：
```
Stage           CKA Score   状态
text_fine       0.25-0.35   ✓ 保持差异
text_final      0.82-0.88   ✓ 适度差异
fused           0.85-0.92   ✓ 融合效果保持！
```

### 3. 性能指标变化

| 指标 | 当前 | 预期（对比学习） | 改善 |
|------|------|------------------|------|
| 验证MAE | ~10 | **7-8** | -20~30% |
| 测试MAE | ~10 | **7.5-8.5** | -15~25% |
| R² | 0.85 | **0.90-0.92** | +5~7% |
| Train-Val Gap | 3-4x | **1.5-2.5x** | -40~50% |

---

## ⚠️ 注意事项

### 1. 对比学习不是万能药

**有效的场景**（您的情况符合！）：
- ✅ 过拟合严重
- ✅ 多模态融合
- ✅ 特征坍塌（CKA分析显示）
- ✅ 数据量中等以上

**可能无效的场景**：
- ✗ 数据量太少（<500样本）
- ✗ 模型容量不足
- ✗ 单模态任务

### 2. 可能的副作用

**训练MAE上升**：
```
这是正常现象！
对比学习 = 正则化 → 训练更难 → 训练误差略高
但验证误差降低 → 这才是目标！
```

**训练时间增加**：
```
对比损失计算 → 额外的前向传播和梯度计算
预计训练时间增加 10-20%
```

### 3. 超参数敏感性

对比学习对超参数较敏感，需要调优：

**如果验证MAE不降反升**：
- 降低 `contrastive_loss_weight` (0.3 → 0.1)
- 提高 `contrastive_temperature` (0.07 → 0.15)
- 检查实现是否正确

**如果训练不稳定**：
- 降低学习率
- 增加 warmup 步数
- 使用梯度裁剪

---

## 🧪 完整实验计划

### Phase 1: 验证对比学习有效性

```bash
# 1.1 Baseline（无对比学习，作为对照组）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property optb88vdw_bandgap \
    --use_contrastive_loss 0 \
    --output_dir ./baseline_no_contrastive

# 1.2 启用对比学习（保守）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.1 \
    --output_dir ./with_contrastive_0.1

# 1.3 启用对比学习（中等）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --output_dir ./with_contrastive_0.2
```

**评估**：
```bash
# 对比性能
python compare_experiments.py \
    --experiment_dirs ./baseline_no_contrastive ./with_contrastive_*

# CKA分析
python compare_twin_models_cka.py \
    --ckpt_model1 ./baseline_no_contrastive/best_model.pt \
    --ckpt_model2 ./with_contrastive_0.2/best_model.pt
```

### Phase 2: 调优对比学习参数

如果Phase 1效果好，继续调优：

```bash
# 2.1 调优权重
for weight in 0.15 0.25 0.3; do
    python train_with_cross_modal_attention.py \
        --use_contrastive_loss 1 \
        --contrastive_loss_weight $weight \
        --contrastive_temperature 0.1 \
        --output_dir ./tune_weight_$weight
done

# 2.2 调优温度
for temp in 0.05 0.07 0.1 0.15; do
    python train_with_cross_modal_attention.py \
        --use_contrastive_loss 1 \
        --contrastive_loss_weight 0.2 \
        --contrastive_temperature $temp \
        --output_dir ./tune_temp_$temp
done
```

### Phase 3: 最优组合

基于Phase 1-2的最佳参数：

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.08 \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --epochs 300 \
    --output_dir ./final_best_with_contrastive
```

---

## 📊 评估清单

每次实验后，检查：

### ✅ 必须检查的指标

1. **验证MAE是否降低**
   ```bash
   # 查看 history_val.csv
   # 对比 best_val_mae
   ```

2. **Train-Val Gap是否缩小**
   ```bash
   # 计算 val_mae / train_mae
   # 目标：<2.0
   ```

3. **训练曲线是否稳定**
   ```bash
   # 查看loss和mae曲线
   # 不应该剧烈震荡
   ```

### ✅ 推荐检查的指标

4. **CKA分析**
   ```bash
   python compare_twin_models_cka.py \
       --ckpt_model1 baseline.pt \
       --ckpt_model2 with_contrastive.pt

   # 检查 fused CKA 是否降低到 0.85-0.92
   ```

5. **性能对比**
   ```bash
   python analyze_model_performance.py \
       --ckpt_model1 baseline.pt \
       --ckpt_model2 with_contrastive.pt

   # 检查各项指标是否改善
   ```

---

## 💡 关键结论

### 对比学习可能有效的信号（您的情况）：

✅ **严重过拟合** - Train-Val Gap 3-4x
✅ **特征坍塌** - CKA从0.29上升到0.98
✅ **多模态融合** - 有图和文本两个模态
✅ **融合效果被抵消** - 中间有差异，最后消失

### 推荐的起始配置：

```python
use_contrastive_loss = True
contrastive_loss_weight = 0.2      # 中等强度
contrastive_temperature = 0.08     # 较锐利的分布
```

### 预期改善（保守估计）：

- 验证MAE: 10 → **8** (-20%)
- Train-Val Gap: 3.5x → **2.2x** (-37%)
- CKA (fused): 0.98 → **0.88** (保持融合效果)

### 如果效果不明显：

1. 尝试更大的 `contrastive_loss_weight` (0.3-0.5)
2. 检查对比损失的实现是否正确
3. 结合其他正则化方法（dropout, weight_decay）
4. 考虑调整融合层位置

---

## 🚀 立即开始

最简单的尝试：

```bash
# 在当前最佳配置基础上，只添加对比学习
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --use_contrastive_loss 1 \
    --contrastive_loss_weight 0.2 \
    --contrastive_temperature 0.1 \
    [... 其他参数保持不变 ...]
    --output_dir ./quick_test_contrastive
```

运行后对比：
- 验证MAE是否降低
- 训练曲线是否更稳定
- CKA分析是否显示更好的融合保持

如果有效 → 进入完整的调优流程
如果无效 → 分析原因，调整参数或策略
