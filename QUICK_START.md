# 🚀 DynamicFusionModule 快速开始指南

## 使用你熟悉的 train_with_cross_modal_attention.py

---

## ⚡ 最快开始方式

### 方式 1: 一键脚本

```bash
./run_dynamic_fusion_training.sh
```

### 方式 2: 快速测试（5 epochs）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --n_train 100 \
    --n_val 20 \
    --n_test 20 \
    --epochs 5 \
    --output_dir ./output_test/
```

### 方式 3: 完整训练

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --root_dir ../dataset/ \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --epochs 100 \
    --batch_size 32 \
    --early_stopping_patience 20 \
    --output_dir ./output_dynamic_fusion/
```

---

## 📊 训练时会看到什么

### 启动时：
```
==========================================
CrysMMNet 训练 - 跨模态注意力机制
==========================================

中期融合配置:
  启用: True
  融合层: 2
  隐藏维度: 128

✅ DynamicFusionModule weight monitoring enabled (logs every 5 epochs)
```

### 每 5 个 epoch：
```
================================================================================
DynamicFusionModule Weight Statistics (Epoch 50)
================================================================================

Fusion Module: layer_2

Router learned weights:
  w_graph: 0.6842  ← 路由器给图的权重
  w_text:  0.3158  ← 路由器给文本的权重

Effective weights (with double residual):
  Graph:  1.6842 (84.2%)  ← 图的实际影响力
  Text:   0.3158 (15.8%)  ← 文本的实际影响力

✅ Graph dominant (ratio: 5.33x)  ← 理想状态
================================================================================
```

---

## 📁 输出文件

训练完成后在 `output_dynamic_fusion/formation_energy_peratom/`：

```
best_val_model.pt           # 最佳验证集模型
best_test_model.pt          # 最佳测试集模型
fusion_weights.csv          # ⭐ 权重演化记录（重点）
history_val.json            # 验证集历史
history_train.json          # 训练集历史
config.json                 # 训练配置
```

---

## 🔍 分析结果

### 查看权重演化：

```bash
python analyze_fusion_weights.py \
    --output_dir ./output_dynamic_fusion/formation_energy_peratom/
```

输出：
```
================================================================================
权重统计分析
================================================================================

总记录数: 20
训练轮数: 5 - 100

最终权重 (Epoch 100):
  layer_2_w_graph: 0.684217
  layer_2_w_text:  0.315783
  layer_2_eff_ratio: 5.334512

趋势分析:
  layer_2_w_graph: 0.6523 → 0.6842 (变化: +4.89%)
  layer_2_w_text: 0.3477 → 0.3158 (变化: -9.17%)

健康检查:
  ✅ layer_2_eff_ratio: 5.33x (图占主导)

✅ 图表已保存: output_dynamic_fusion/formation_energy_peratom/layer_2_weights.png
```

### 手动查看：

```bash
# 查看权重日志
cat output_dynamic_fusion/formation_energy_peratom/fusion_weights.csv

# 查看最后一次记录
tail -1 output_dynamic_fusion/formation_energy_peratom/fusion_weights.csv

# 查看训练历史
cat output_dynamic_fusion/formation_energy_peratom/history_val.json
```

---

## 🎯 关键参数

### DynamicFusionModule 参数（必须）

```bash
--use_middle_fusion True              # 启用动态融合
--middle_fusion_layers "2"            # 在第2层应用融合
--middle_fusion_hidden_dim 128        # 路由器隐藏维度
--middle_fusion_dropout 0.1           # Dropout 率
```

### 其他推荐参数

```bash
--use_cross_modal True                # 启用跨模态注意力
--cross_modal_num_heads 4             # 注意力头数
--epochs 100                          # 训练轮数
--batch_size 32                       # 批次大小
--early_stopping_patience 20          # 早停耐心值
```

---

## 📋 不同任务示例

### JARVIS - Formation Energy
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy_peratom \
    --use_middle_fusion True \
    --epochs 100
```

### JARVIS - Band Gap
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_middle_fusion True \
    --epochs 100
```

### Material Project - Band Gap
```bash
python train_with_cross_modal_attention.py \
    --dataset mp \
    --property band_gap \
    --use_middle_fusion True \
    --n_train 60000 \
    --epochs 100
```

---

## ⚙️ 高级用法

### 多层融合
```bash
--middle_fusion_layers "2,3"          # 在第2和第3层应用
```

### 联合细粒度注意力
```bash
--use_middle_fusion True \
--use_fine_grained_attention True \
--use_cross_modal True
```

### 添加对比学习
```bash
--use_middle_fusion True \
--use_contrastive True \
--contrastive_weight 0.1
```

---

## 🎯 健康指标

训练时关注这些权重指标：

| 指标 | 健康范围 | 含义 |
|------|---------|------|
| **w_graph** | 0.5-0.9 | 路由器给图的权重 |
| **w_text** | 0.1-0.5 | 路由器给文本的权重 |
| **图/文本比例** | 3-10x | 图应该占主导 |

⚠️ **警告信号**：
- 比例 < 2x → 文本权重过高
- w_text > 0.7 → 可能过拟合文本描述

---

## ⚠️ 常见问题

### Q: 找不到数据集文件？
```bash
# 检查 --root_dir 参数
# 在项目根目录运行：
--root_dir ./dataset/

# 在 src 目录运行：
--root_dir ../dataset/
```

### Q: 没有生成 fusion_weights.csv？
**原因**：
- 训练轮数 < 5
- `--use_middle_fusion` 未设置为 True

**解决**：
```bash
--use_middle_fusion True --epochs 5
```

### Q: 权重比例异常？
查看分析报告：
```bash
python analyze_fusion_weights.py --output_dir ./output_xxx/
```

---

## 📚 详细文档

- **完整示例**: `TRAINING_EXAMPLES.md`
- **通用命令**: `TRAINING_COMMANDS.md`
- **集成指南**: `INTEGRATION_CHECKLIST.md`
- **实现代码**: `models/alignn.py` (第 121-257 行)

---

## 💡 推荐工作流

```bash
# 1. 验证集成
python test_integration.py

# 2. 快速测试（5 epochs）
python train_with_cross_modal_attention.py \
    --use_middle_fusion True \
    --n_train 100 --epochs 5 \
    --output_dir ./test/

# 3. 查看结果
python analyze_fusion_weights.py --output_dir ./test/

# 4. 完整训练
python train_with_cross_modal_attention.py \
    --use_middle_fusion True \
    --epochs 100 \
    --output_dir ./final/

# 5. 分析最终结果
python analyze_fusion_weights.py --output_dir ./final/
```

---

## 🎓 为什么图应该占主导？

**物理本质**：
- 材料性质由原子结构决定（Schrödinger 方程）
- 文本只是对结构的描述，是二手信息
- 图包含完整信息，文本可能不完备

**双重残差的作用**：
```python
# 最终输出
out = node_feat + (w_graph * node_feat + w_text * text_feat)
    = (1 + w_graph) * node_feat + w_text * text_feat

# 图特征永远有 ≥1.0 的基础权重
# 文本只能起"调节"作用，不能喧宾夺主
```

---

**准备好了吗？开始训练：**
```bash
./run_dynamic_fusion_training.sh
```

或

```bash
python train_with_cross_modal_attention.py --use_middle_fusion True --epochs 100
```

🚀 祝训练顺利！
