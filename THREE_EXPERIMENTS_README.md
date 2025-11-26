# 三个融合机制对比实验说明

## 🎯 实验目的

对比**中期融合**与不同注意力机制组合的效果，所有实验都使用**图特征单独预测**模式。

---

## 📊 实验设计

### 实验配置对比表

| 实验 | 中期融合 | 细粒度注意力 | 跨模态注意力 | 图特征预测 | 输出目录 |
|------|---------|------------|------------|-----------|---------|
| **实验1** | ✅ 层2 | ❌ | ❌ | ✅ | `output_100epochs_42_bs128_middle_fusion_only` |
| **实验2** | ✅ 层2 | ✅ 8头 | ❌ | ✅ | `output_100epochs_42_bs128_middle_fine_grained` |
| **实验3** | ✅ 层2 | ❌ | ✅ 4头 | ✅ | `output_100epochs_42_bs128_middle_cross_modal` |

---

## 🔄 执行流程对比

### 实验1: 中期融合 → 图预测

```
文本编码
   ↓
ALIGNN层1 (节点更新)
   ↓
ALIGNN层2 (节点更新 + 中期融合 ⭐)
   ↓  [文本信息注入节点特征]
ALIGNN层3-4
   ↓
GCN层1-4
   ↓
Readout (图池化)
   ↓
图投影 (64维)
   ↓
预测 (只用图特征)
```

**特点**: 文本在ALIGNN编码早期注入，影响后续所有层

---

### 实验2: 中期融合 + 细粒度注意力 → 图预测

```
文本编码 (token序列)
   ↓
ALIGNN层1 (节点更新)
   ↓
ALIGNN层2 (节点更新 + 中期融合 ⭐)
   ↓  [文本信息注入节点特征]
ALIGNN层3-4
   ↓
GCN层1-4
   ↓
细粒度注意力 ⭐⭐
   原子 ↔ 词元交互 (8头)
   ↓  [节点特征再次增强]
Readout (图池化)
   ↓
图投影 (64维)
   ↓
预测 (只用图特征)
```

**特点**: 两阶段文本增强
- 第一阶段：中期融合（早期，粗粒度）
- 第二阶段：细粒度注意力（后期，细粒度）

---

### 实验3: 中期融合 + 跨模态注意力 → 图预测

```
文本编码 (CLS向量)
   ↓
ALIGNN层1 (节点更新)
   ↓
ALIGNN层2 (节点更新 + 中期融合 ⭐)
   ↓  [文本信息注入节点特征]
ALIGNN层3-4
   ↓
GCN层1-4
   ↓
Readout (图池化)
   ↓
图投影 (64维)
   ↓
跨模态注意力 ⭐⭐
   图 ↔ 文本交互 (4头)
   ↓  [图特征增强]
预测 (只用增强后的图特征)
```

**特点**: 两阶段文本增强
- 第一阶段：中期融合（节点级）
- 第二阶段：跨模态注意力（图级）

---

## 🚀 运行方法

### 方法1: 一键运行所有实验

```bash
./run_three_fusion_experiments.sh
```

这个脚本会：
- ✅ 依次运行三个实验（串行执行）
- ✅ 每个实验完成后自动开始下一个
- ✅ 如果某个实验失败，终止后续实验
- ✅ 自动创建输出目录和日志文件
- ✅ 使用相同的随机种子(42)保证可复现性

**预计总时间**: 约6-9小时（取决于GPU和数据集大小）

---

### 方法2: 单独运行某个实验

#### 实验1: 只用中期融合

```bash
python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 128 \
    --epochs 100 \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --use_only_graph_for_prediction True \
    --output_dir ./output_middle_fusion_only \
    --random_seed 42
```

#### 实验2: 中期融合 + 细粒度

```bash
python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 128 \
    --epochs 100 \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention True \
    --fine_grained_num_heads 8 \
    --use_only_graph_for_prediction True \
    --output_dir ./output_middle_fine_grained \
    --random_seed 42
```

#### 实验3: 中期融合 + 跨模态

```bash
python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --batch_size 128 \
    --epochs 100 \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers 2 \
    --use_fine_grained_attention False \
    --use_only_graph_for_prediction True \
    --output_dir ./output_middle_cross_modal \
    --random_seed 42
```

---

## 📈 查看结果

### 训练过程监控

在脚本运行时，可以实时查看某个实验的日志：

```bash
# 实验1
tail -f ./output_100epochs_42_bs128_middle_fusion_only/train_*.log

# 实验2
tail -f ./output_100epochs_42_bs128_middle_fine_grained/train_*.log

# 实验3
tail -f ./output_100epochs_42_bs128_middle_cross_modal/train_*.log
```

### 快速查看最佳结果

```bash
# 实验1
grep 'Best test MAE' ./output_100epochs_42_bs128_middle_fusion_only/train_*.log | tail -1

# 实验2
grep 'Best test MAE' ./output_100epochs_42_bs128_middle_fine_grained/train_*.log | tail -1

# 实验3
grep 'Best test MAE' ./output_100epochs_42_bs128_middle_cross_modal/train_*.log | tail -1
```

### 结果文件位置

每个实验的输出目录包含：

```
output_XXX/
├── train_YYYYMMDD_HHMMSS.log      # 训练日志
├── best_test_model.pt              # 测试集最佳模型
├── best_val_model.pt               # 验证集最佳模型
├── checkpoint_epoch_X.pt           # 定期检查点
└── training_history.json           # 训练历史（loss、MAE等）
```

---

## 🔬 预期结果分析

### 假设

基于融合机制的特点，预期结果：

| 实验 | 预期MAE | 训练速度 | 原因 |
|------|---------|---------|------|
| **实验1** | ~0.040 | 快 | 只有中期融合，信息传播较浅 |
| **实验2** | ~0.035 | 中 | 中期+细粒度双重增强，局部对齐精准 |
| **实验3** | ~0.037 | 快 | 中期+跨模态双重增强，全局语义强 |

### 分析维度

对比三个实验时，关注：

1. **性能指标**
   - 测试集MAE
   - 验证集MAE
   - 收敛速度

2. **融合效果**
   - 中期融合的贡献
   - 细粒度vs跨模态的增强效果

3. **计算效率**
   - 训练时间
   - 内存占用

---

## 📊 结果汇总脚本

实验全部完成后，可以使用以下脚本生成对比报告：

```python
import json
import pandas as pd

# 读取训练历史
exp1 = json.load(open('./output_100epochs_42_bs128_middle_fusion_only/training_history.json'))
exp2 = json.load(open('./output_100epochs_42_bs128_middle_fine_grained/training_history.json'))
exp3 = json.load(open('./output_100epochs_42_bs128_middle_cross_modal/training_history.json'))

# 汇总结果
results = {
    '实验': ['中期融合', '中期+细粒度', '中期+跨模态'],
    '最佳测试MAE': [
        min(exp1['test_mae']),
        min(exp2['test_mae']),
        min(exp3['test_mae'])
    ],
    '最佳验证MAE': [
        min(exp1['val_mae']),
        min(exp2['val_mae']),
        min(exp3['val_mae'])
    ],
    '收敛Epoch': [
        exp1['test_mae'].index(min(exp1['test_mae'])),
        exp2['test_mae'].index(min(exp2['test_mae'])),
        exp3['test_mae'].index(min(exp3['test_mae']))
    ]
}

df = pd.DataFrame(results)
print(df)
df.to_csv('fusion_comparison_results.csv', index=False)
```

---

## 💡 实验建议

### 1. 监控训练

定期检查：
- 损失曲线是否正常下降
- 是否出现过拟合
- GPU利用率

### 2. 提前停止

如果某个实验：
- 100 epoch内MAE降到满意水平 → 可以提前停止
- 长时间不收敛 → 检查超参数

### 3. 资源优化

如果GPU内存不足：
```bash
--batch_size 64          # 减小批次
--num_workers 12         # 减少worker
```

---

## 🎯 核心问题回答

### Q: 为什么都用中期融合？

**A**: 中期融合作为基础，对比**在此基础上**添加细粒度或跨模态注意力的增量效果。

### Q: 为什么不测试细粒度+跨模态？

**A**: 可以单独测试。如需添加第四个实验：

```bash
--use_middle_fusion True \
--use_fine_grained_attention True \
--use_cross_modal True \
--use_only_graph_for_prediction True
```

这是最强配置（三重增强）。

### Q: 为什么都用图特征单独预测？

**A**:
1. 统一对比基准（避免融合方式不同导致的混淆）
2. 评估文本增强图特征的能力
3. 更好的可解释性

---

## 📚 相关文档

- [图特征单独预测详解](GRAPH_ONLY_PREDICTION.md)
- [快速开始指南](QUICK_START_GRAPH_ONLY.md)
- [融合位置对比实验](FUSION_EXPERIMENT_README.md)

---

## 🎉 总结

这三个实验将回答：

1. **中期融合的基准性能** （实验1）
2. **细粒度注意力的增量贡献** （实验2 vs 实验1）
3. **跨模态注意力的增量贡献** （实验3 vs 实验1）

通过对比，可以确定**最适合你数据集的融合策略**！

祝实验顺利！🚀
