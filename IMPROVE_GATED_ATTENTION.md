# Gated Cross-Attention 性能优化指南

## 🔍 问题现状

| 模型配置 | MAE | 性能差距 |
|---------|-----|---------|
| **SGANet** (中期融合+细粒度+跨模态) | **0.2554** | baseline |
| **Gated** (中期融合+细粒度+Gated CA) | **0.2672** | **+4.6%** ❌ |

**问题**：Gated Cross-Attention没有达到预期性能，反而比原始跨模态注意力差。

---

## 🎯 可能的原因

### 1. **过度门控 (Over-Gating)**

**症状**：
- Effective Weight均值 < 0.2
- 文本信息被过度抑制
- 模型退化为"仅图结构"模式

**原因**：
```python
effective_weight = quality_score * fusion_weight
```
两个[0, 1]范围的值相乘，导致权重过小。

**诊断方法**：
```bash
python diagnose_gated_attention.py \
    --checkpoint your_gated_model.pt \
    --dataset jarvis \
    --property mbj_bandgap
```

---

### 2. **质量检测过于保守**

**症状**：
- Quality Score均值 < 0.3
- 即使是干净文本，质量得分也很低

**原因**：
- Norm检测阈值不合理：`sigmoid(feat_norm - 3.0)`
- 质量网络未充分训练

---

## 💡 改进方案

### 方案 1: **调整门控公式** (推荐优先尝试)

**修改位置**：`models/alignn.py:959`

**解决方案 A - 软门控**：
```python
# 原始（乘性）
effective_weight = quality_score * fusion_weight

# 改进（软门控）
effective_weight = fusion_weight * (0.3 + 0.7 * quality_score)
# 解释：quality=0时，effective_weight = 0.3 * fusion_weight（保留30%）
#      quality=1时，effective_weight = fusion_weight（完整权重）
```

**解决方案 B - 加权平均**：
```python
alpha = 0.7  # 可调超参数
effective_weight = alpha * fusion_weight + (1 - alpha) * quality_score
```

---

### 方案 2: **调整质量检测阈值**

**修改位置**：`models/alignn.py:809-817`

**改进 A - 移除norm检测**：
```python
# 在TextQualityGate.__init__中
self.use_norm_detection = False
```

**改进 B - 自适应阈值**：
```python
self.norm_threshold = nn.Parameter(torch.tensor(3.0))

def forward(self, text_feat):
    quality_score = self.quality_network(text_feat)
    if self.use_norm_detection:
        feat_norm = torch.norm(text_feat, dim=-1, keepdim=True)
        norm_quality = torch.sigmoid(feat_norm - self.norm_threshold)
        quality_score = quality_score * norm_quality
    return quality_score
```

---

### 方案 3: **延长训练**

```bash
python train_with_cross_modal_attention.py \
    --use_gated_cross_attention True \
    --use_middle_fusion True \
    --use_fine_grained_attention True \
    --epochs 150 \  # 增加50%训练轮数
    ...
```

---

## 🚀 推荐实施步骤

### Step 1: 诊断当前问题

```bash
python diagnose_gated_attention.py \
    --checkpoint your_gated_model.pt \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./diagnosis_results
```

**查看诊断结果**：
- 如果 Effective Weight < 0.2 → **过度门控**，优先尝试方案1
- 如果 Quality Score < 0.3 → **质量检测问题**，优先尝试方案2

---

### Step 2: 快速修复（推荐）

**修改 `models/alignn.py:959`**：

```python
# 原始代码
effective_weight = quality_score * fusion_weight

# 改为软门控（推荐）
effective_weight = fusion_weight * (0.3 + 0.7 * quality_score)
```

**重新训练**。

---

### Step 3: 验证改进

```bash
# 重新训练后，再次诊断
python diagnose_gated_attention.py \
    --checkpoint improved_gated_model.pt \
    --output_dir ./diagnosis_improved
```

**目标**：
- Effective Weight均值 > 0.4
- Quality Score均值 > 0.5
- MAE ≤ 0.2554（至少匹配SGANet）

---

## 📊 预期结果

### 优化前
```
Effective Weight均值: 0.15  ← 过低
Quality Score均值:    0.25  ← 过低
MAE:                  0.2672 ← 性能差
```

### 优化后（目标）
```
Effective Weight均值: 0.50  ← 合理
Quality Score均值:    0.70  ← 正常
MAE:                  0.2550 ← 匹配SGANet

鲁棒性测试（100%遮挡）:
MAE:                  0.5358 ← 匹配无融合模型
```

---

**祝调试顺利！** 🚀
