# 模型参数调优建议

## 📊 当前训练状态分析

### 观察到的现象

从训练曲线可以看出：

1. **收敛特征**：
   - Loss和MAE在前100个epoch快速下降
   - 100 epoch后曲线完全平稳，几乎无变化
   - 训练持续到800+ epoch，但后700个epoch无明显改善

2. **性能表现**：
   - 训练MAE: ~2-3
   - 验证MAE: ~10
   - **Train-Val Gap**: 明显的过拟合信号（验证误差约为训练误差的3-4倍）

3. **潜在问题**：
   - ⚠️ **轻微过拟合** - 训练和验证误差差距较大
   - ⚠️ **提前饱和** - 100 epoch后无改善，但训练了800 epoch
   - ⚠️ **Early stopping未触发** - 设置了150个epoch，但实际训练了800个

---

## 🎯 参数调整建议（按优先级排序）

### 优先级1️⃣: 正则化参数（缓解过拟合）⭐⭐⭐

#### 当前配置：
```json
{
  "graph_dropout": 0.15,
  "weight_decay": 0.0005,
  "cross_modal_dropout": 0.1,
  "fine_grained_dropout": 0.2,
  "middle_fusion_dropout": 0.1
}
```

#### 建议调整：

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| **graph_dropout** | 0.15 | **0.2-0.25** | 增加图层dropout，减少过拟合 |
| **weight_decay** | 0.0005 | **0.001-0.002** | 增强L2正则化 |
| **cross_modal_dropout** | 0.1 | **0.15-0.2** | 全局注意力需要更强正则化 |
| **fine_grained_dropout** | 0.2 | **0.25-0.3** | 细粒度注意力容易过拟合 |
| **middle_fusion_dropout** | 0.1 | **0.15-0.2** | 增强融合层正则化 |

#### 实验方案：
```bash
# 实验1: 保守调整
--graph_dropout 0.2 \
--weight_decay 0.001 \
--cross_modal_dropout 0.15 \
--fine_grained_dropout 0.25

# 实验2: 激进调整
--graph_dropout 0.25 \
--weight_decay 0.002 \
--cross_modal_dropout 0.2 \
--fine_grained_dropout 0.3
```

---

### 优先级2️⃣: 融合机制参数（提升融合效果）⭐⭐⭐

#### 当前配置：
```json
{
  "use_middle_fusion": true,
  "middle_fusion_layers": "2",      // 仅第2层
  "middle_fusion_hidden_dim": 128,   // 较小
  "middle_fusion_num_heads": 2       // 较少
}
```

#### 观察：
从您之前的CKA分析可以看出，**text_fine阶段差异大但最终又收敛**，说明融合影响可能被后续层抵消。

#### 建议调整：

**策略A: 增加融合层数**
```json
{
  "middle_fusion_layers": "1,2"    // 更早融合
  // 或
  "middle_fusion_layers": "2,3"    // 更深度融合
  // 或
  "middle_fusion_layers": "1,2,3"  // 全面融合（激进）
}
```

**策略B: 增强融合容量**
```json
{
  "middle_fusion_hidden_dim": 256,     // 当前128 → 256（匹配主网络）
  "middle_fusion_num_heads": 4         // 当前2 → 4（更多注意力头）
}
```

**策略C: 组合调整（推荐）**
```json
{
  "middle_fusion_layers": "2,3",
  "middle_fusion_hidden_dim": 256,
  "middle_fusion_num_heads": 4,
  "middle_fusion_dropout": 0.15
}
```

#### 实验方案：
```bash
# 实验3: 双层融合 + 增强容量
--middle_fusion_layers "2,3" \
--middle_fusion_hidden_dim 256 \
--middle_fusion_num_heads 4

# 实验4: 更早融合
--middle_fusion_layers "1,2" \
--middle_fusion_hidden_dim 256 \
--middle_fusion_num_heads 4

# 实验5: 三层融合（激进）
--middle_fusion_layers "1,2,3" \
--middle_fusion_hidden_dim 256 \
--middle_fusion_num_heads 4
```

---

### 优先级3️⃣: 学习率和训练策略 ⭐⭐

#### 当前配置：
```json
{
  "learning_rate": 0.001,
  "optimizer": "adamw",
  "scheduler": "onecycle",
  "warmup_steps": 2000,
  "epochs": 100,          // 但实际训练了800
  "n_early_stopping": 150
}
```

#### 观察：
- 曲线在100 epoch后完全平稳
- Early stopping设置为150但未触发（可能bug或验证loss有微小波动）
- OneCycle调度器可能不是最佳选择

#### 建议调整：

**学习率：**
```json
{
  "learning_rate": 0.0005  // 当前0.001可能偏大，降低到0.0005
}
```

**调度器：**
```json
{
  "scheduler": "lambda"     // 改用lambda调度器
  // 或
  "scheduler": "cosine"     // 改用cosine调度器
}
```

**Early Stopping：**
```json
{
  "n_early_stopping": 100   // 从150降低到100（因为曲线在100 epoch就饱和了）
}
```

**Epoch数量：**
```json
{
  "epochs": 300            // 设置更合理的上限（当前设置100但实际训练了800）
}
```

#### 实验方案：
```bash
# 实验6: 降低学习率 + 更好的调度器
--learning_rate 0.0005 \
--scheduler lambda \
--n_early_stopping 100 \
--epochs 300

# 实验7: 更保守的学习率
--learning_rate 0.0003 \
--scheduler cosine \
--n_early_stopping 100
```

---

### 优先级4️⃣: 注意力机制参数 ⭐

#### 当前配置：
```json
{
  "cross_modal_num_heads": 4,        // 全局注意力
  "fine_grained_num_heads": 8,       // 细粒度注意力
  "cross_modal_hidden_dim": 256,
  "fine_grained_hidden_dim": 256
}
```

#### 建议调整：

从CKA分析看，**全局注意力可能过强**，导致融合效果被"平滑"掉：

```json
{
  "cross_modal_num_heads": 2         // 从4降到2（减弱全局注意力）
  // 或暂时关闭
  "use_cross_modal_attention": false  // 测试是否是全局注意力抵消了融合
}
```

#### 实验方案：
```bash
# 实验8: 减弱全局注意力
--cross_modal_num_heads 2

# 实验9: 关闭全局注意力（测试）
--use_cross_modal_attention 0
```

---

### 优先级5️⃣: Batch Size ⭐

#### 当前配置：
```json
{
  "batch_size": 128
}
```

#### 建议调整：

较小的batch size通常有更好的泛化性能：

```json
{
  "batch_size": 64    // 降低batch size可能改善泛化
}
```

#### 实验方案：
```bash
# 实验10: 减小batch size
--batch_size 64

# 实验11: 更小的batch size（如果内存允许多accumulate）
--batch_size 32
```

---

### 优先级6️⃣: 模型架构参数 ⭐

#### 当前配置：
```json
{
  "alignn_layers": 4,
  "gcn_layers": 4,
  "hidden_features": 256,
  "embedding_features": 64
}
```

#### 建议调整（谨慎）：

**如果过拟合严重，可以考虑减小模型容量：**
```json
{
  "hidden_features": 128     // 从256降到128（减小模型容量）
  // 或
  "alignn_layers": 3,        // 从4降到3
  "gcn_layers": 3
}
```

**注意**：这个调整会改变模型架构，需要重新训练。

---

## 🧪 推荐的实验计划

### Phase 1: 快速改进（解决过拟合）

```bash
# 组合实验1: 正则化 + 学习率调整
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --fine_grained_dropout 0.25 \
    --middle_fusion_dropout 0.15 \
    --learning_rate 0.0005 \
    --scheduler lambda \
    --n_early_stopping 100 \
    --epochs 300 \
    --output_dir ./tuned_regularization
```

### Phase 2: 增强融合机制

```bash
# 组合实验2: 增强融合 + 正则化
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --n_early_stopping 100 \
    --epochs 300 \
    --output_dir ./tuned_enhanced_fusion
```

### Phase 3: 测试全局注意力影响

```bash
# 组合实验3: 减弱全局注意力 + 增强融合
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --output_dir ./tuned_weak_cross_attention
```

### Phase 4: 激进调整（如果前面效果不明显）

```bash
# 组合实验4: 三层融合 + 强正则化
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --middle_fusion_layers "1,2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --graph_dropout 0.3 \
    --weight_decay 0.002 \
    --cross_modal_dropout 0.2 \
    --fine_grained_dropout 0.3 \
    --learning_rate 0.0003 \
    --batch_size 64 \
    --output_dir ./tuned_aggressive
```

---

## 📈 评估指标

每次实验后，使用以下工具进行评估：

### 1. 训练曲线检查
```bash
# 查看训练曲线，确认：
# - Train-Val gap是否缩小
# - 验证MAE是否降低
# - 是否还有过拟合
```

### 2. CKA分析（关键！）
```bash
# 使用完整测试集进行CKA分析
python compare_twin_models_cka.py \
    --ckpt_model1 ./baseline/best_model.pt \
    --ckpt_model2 ./tuned_enhanced_fusion/best_model.pt \
    --model1_name "Original" \
    --model2_name "Tuned" \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --save_dir ./cka_tuned_comparison

# 关注：
# - text_fine CKA是否仍然很低（说明融合在起作用）
# - text_final/fused CKA是否降低（说明融合效果保持到了最后）
```

### 3. 性能对比
```bash
# 对比实际预测性能
python analyze_model_performance.py \
    --ckpt_model1 ./baseline/best_model.pt \
    --ckpt_model2 ./tuned_enhanced_fusion/best_model.pt \
    --model1_name "Original" \
    --model2_name "Tuned" \
    --dataset jarvis \
    --property optb88vdw_bandgap \
    --save_dir ./performance_tuned_comparison

# 关注：
# - 验证MAE是否降低
# - Train-Val gap是否缩小
# - R²是否提升
```

---

## 🎯 预期效果

### 成功的调优应该带来：

1. **验证MAE降低**: 从 ~10 降低到 ~8 或更低
2. **Train-Val gap缩小**: 从 3-4倍 缩小到 2倍以内
3. **CKA模式改善**:
   - text_fine保持低CKA（0.2-0.4）
   - text_final/fused CKA降低（从0.98降到0.85-0.92）
4. **泛化性能提升**: 在新数据上表现更好

### 如果调优后：

| 结果 | 说明 | 下一步 |
|------|------|--------|
| ✅ 验证MAE降低 | 调优成功 | 继续fine-tune参数 |
| ⚠️ 验证MAE不变 | 需要更激进的改变 | 尝试Phase 4的激进调整 |
| ❌ 验证MAE上升 | 调整过度 | 回退到更保守的设置 |

---

## 💡 额外建议

### 1. 数据层面
```bash
# 检查数据分布
# - 训练集和测试集是否分布一致
# - 是否有异常值
# - 是否需要数据归一化调整
```

### 2. 集成学习
```bash
# 训练多个随机种子的模型
for seed in 1 2 3 4 5; do
    python train_with_cross_modal_attention.py \
        --random_seed $seed \
        --output_dir ./ensemble_seed_$seed \
        # ... 其他最优参数
done

# 集成预测
python ensemble_predictions.py --model_dirs ./ensemble_seed_*
```

### 3. 对比学习损失（当前未使用）
```json
{
  "use_contrastive_loss": true,
  "contrastive_loss_weight": 0.1,
  "contrastive_temperature": 0.1
}
```

这可能帮助融合机制学习更有区分度的特征。

---

## 📝 总结

### 最关键的三个调整（立即尝试）：

1. **增强正则化** - 解决当前的过拟合问题
   ```bash
   --graph_dropout 0.25 --weight_decay 0.001
   ```

2. **增强融合机制** - 提升融合效果，防止被后续层抵消
   ```bash
   --middle_fusion_layers "2,3" \
   --middle_fusion_hidden_dim 256 \
   --middle_fusion_num_heads 4
   ```

3. **降低学习率** - 更稳定的训练
   ```bash
   --learning_rate 0.0005
   ```

### 评估优先级：

1. **训练曲线** - 检查过拟合是否改善
2. **CKA分析** - 检查融合效果是否保持
3. **性能指标** - 检查实际预测是否提升

建议先运行**组合实验1和2**，对比效果后决定是否需要更激进的调整。
