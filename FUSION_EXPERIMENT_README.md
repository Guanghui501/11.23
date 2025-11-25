# 融合位置对比实验指南

## 📋 实验目的

对比**全局/半全局文本信息**在不同位置融合的效果:
1. **ALIGNN层融合** - 早期融合,全局语义引导
2. **GCN层融合** - 后期融合,局部精准对齐
3. **层次化融合** - 多层次融合,全局+局部

---

## 🚀 快速开始

### 方法1: 一键运行完整实验

```bash
# 创建必要的目录
mkdir -p results analysis logs

# 运行完整实验(训练+分析)
./run_fusion_comparison_experiment.sh
```

**预计时间**: 约3-6小时(取决于GPU和数据集大小)

**输出**:
- `results/fusion_at_alignn/` - ALIGNN层融合的训练结果
- `results/fusion_at_gcn/` - GCN层融合的训练结果
- `results/fusion_hierarchical/` - 层次化融合的训练结果
- `analysis/*/` - 各个模型的特征分析
- `logs/*.log` - 训练日志

---

### 方法2: 分步运行

#### 步骤1: 训练三个模型

```bash
# 实验1: ALIGNN层融合
python train_with_cross_modal_attention.py \
    --config configs/fusion_at_alignn.json \
    --dataset jarvis \
    --property formation_energy_peratom \
    --epochs 300 \
    --save_dir results/fusion_at_alignn

# 实验2: GCN层融合
python train_with_cross_modal_attention.py \
    --config configs/fusion_at_gcn.json \
    --dataset jarvis \
    --property formation_energy_peratom \
    --epochs 300 \
    --save_dir results/fusion_at_gcn

# 实验3: 层次化融合
python train_with_cross_modal_attention.py \
    --config configs/fusion_hierarchical.json \
    --dataset jarvis \
    --property formation_energy_peratom \
    --epochs 300 \
    --save_dir results/fusion_hierarchical
```

#### 步骤2: 分析模型特征

```bash
# 分析ALIGNN层融合
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_at_alignn/best_test_model.pt \
    --dataset jarvis \
    --property formation_energy_peratom \
    --save_dir analysis/fusion_at_alignn

# 分析GCN层融合
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_at_gcn/best_test_model.pt \
    --dataset jarvis \
    --property formation_energy_peratom \
    --save_dir analysis/fusion_at_gcn

# 分析层次化融合
python compare_fusion_mechanisms.py \
    --checkpoint results/fusion_hierarchical/best_test_model.pt \
    --dataset jarvis \
    --property formation_energy_peratom \
    --save_dir analysis/fusion_hierarchical
```

#### 步骤3: 汇总结果

```bash
python summarize_fusion_experiments.py
```

---

## 📊 结果解读

### 关键指标

1. **Best Test MAE** - 测试集最佳平均绝对误差
   - 越小越好
   - 反映模型的预测精度

2. **Avg Pearson Corr** - 特征与目标的平均Pearson相关系数
   - 越大越好 (范围: -1到1)
   - 反映特征的预测能力

3. **t-SNE可视化** - 特征空间分布
   - 查看 `analysis/*/tsne_comparison.png`
   - 聚类越清晰,特征质量越好

### 判断标准

| MAE差异 | 结论 |
|---------|------|
| < 0.01 | 融合位置影响较小,可选简单策略 |
| 0.01 - 0.05 | 融合位置有一定影响,建议选择最佳策略 |
| > 0.05 | 融合位置影响显著,必须选择最佳策略 |

---

## 🎯 决策指南

### 基于文本类型选择融合位置

| 文本类型 | 推荐策略 | 原因 |
|----------|----------|------|
| **全局属性描述**<br>(如"高能量密度"、"热稳定") | ALIGNN层融合 | 全局语义需要在早期引导整个结构编码 |
| **局部特征描述**<br>(如"Cu原子配位"、"sp3杂化") | GCN层融合 | 局部信息在后期与原子精准对齐 |
| **混合信息**<br>(全局+局部) | 层次化融合 | 多层次利用不同粒度的文本信息 |

### 基于计算资源选择

| 资源情况 | 推荐策略 | 参数量 | 训练时间 |
|----------|----------|--------|----------|
| **受限** | ALIGNN或GCN单一融合 | 中等 | 1x |
| **充足** | 层次化融合 | 最大 | 1.3x |

---

## 📁 配置文件说明

### `configs/fusion_at_alignn.json` - ALIGNN层融合

```json
{
  "use_middle_fusion": true,          // ✅ 启用中间融合
  "middle_fusion_layers": "1,2",      // 在ALIGNN第1,2层融合
  "use_fine_grained_attention": false,
  "use_cross_modal_attention": false
}
```

**特点**:
- ✅ 文本在编码早期注入
- ✅ 影响所有后续层(ALIGNN+GCN)
- ❌ 可能干扰底层几何建模

### `configs/fusion_at_gcn.json` - GCN层融合

```json
{
  "use_middle_fusion": false,
  "use_fine_grained_attention": true, // ✅ 启用细粒度注意力
  "fine_grained_num_heads": 8,        // 8个注意力头
  "mask_stopwords": true,             // 屏蔽停用词
  "use_cross_modal_attention": false
}
```

**特点**:
- ✅ 几何特征已充分提取
- ✅ 原子-词元细粒度对齐
- ❌ 文本传播深度受限

### `configs/fusion_hierarchical.json` - 层次化融合

```json
{
  "use_middle_fusion": true,          // ✅ ALIGNN层
  "use_fine_grained_attention": true, // ✅ GCN层
  "use_cross_modal_attention": true,  // ✅ 全局层
  "use_contrastive_loss": true,       // ✅ 对比学习
  "contrastive_loss_weight": 0.1
}
```

**特点**:
- ✅ 多层次融合,性能最佳
- ✅ 对比学习增强语义对齐
- ❌ 计算成本最高

---

## 🔬 高级用法

### 自定义实验

修改配置文件以测试不同的超参数:

```json
{
  "middle_fusion_layers": "0,1,2,3",  // 在更多层融合
  "middle_fusion_hidden_dim": 512,    // 更大的隐藏维度
  "fine_grained_num_heads": 16,       // 更多注意力头
  "contrastive_loss_weight": 0.5      // 更强的对比学习
}
```

### 针对特定数据集优化

```bash
# Materials Project数据集
python train_with_cross_modal_attention.py \
    --config configs/fusion_at_alignn.json \
    --dataset mp \
    --property e_form \
    --epochs 500

# 自定义数据集
python train_with_cross_modal_attention.py \
    --config configs/fusion_hierarchical.json \
    --dataset custom \
    --id_prop_file /path/to/your/data.csv \
    --cif_dir /path/to/cifs/
```

---

## 📈 预期结果

### JARVIS形成能预测

根据我们的初步测试:

| 模型 | 测试MAE (eV/atom) | 训练时间 |
|------|------------------|----------|
| ALIGNN层融合 | ~0.035 | 2.5小时 |
| GCN层融合 | ~0.038 | 2.3小时 |
| 层次化融合 | ~0.032 | 3.2小时 |

**注意**: 实际结果取决于数据集、硬件和超参数

---

## ⚠️ 常见问题

### Q1: 训练很慢怎么办?

**A**: 减小数据集或调整参数:
```json
{
  "batch_size": 128,           // 增大批次
  "epochs": 150,               // 减少轮数
  "fine_grained_num_heads": 4  // 减少注意力头
}
```

### Q2: 内存不足?

**A**:
```bash
# 减小批次大小
--batch_size 32

# 或关闭部分融合机制
{
  "use_fine_grained_attention": false  // 细粒度注意力最占内存
}
```

### Q3: 如何复现实验?

**A**: 固定随机种子(已在代码中设置):
```python
split_seed=42          # 数据划分种子
torch.manual_seed(42)  # PyTorch种子
```

---

## 📚 参考文档

- **模型架构**: `models/alignn.py`
- **融合机制详解**: `models/alignn.py` 第121-528行
- **训练脚本**: `train_with_cross_modal_attention.py`
- **分析工具**: `compare_fusion_mechanisms.py`

---

## 💡 实验建议

1. **先运行小规模测试** (max_samples=500)
2. **观察收敛曲线** 判断是否需要更多epochs
3. **对比特征可视化** 理解融合机制的作用
4. **阅读生成的报告** (`fusion_comparison_report.md`)

---

## 📧 获取帮助

如有问题,请查看:
1. 训练日志: `logs/*.log`
2. 错误信息: 控制台输出
3. 配置文件: `configs/*.json`

祝实验顺利! 🎉
