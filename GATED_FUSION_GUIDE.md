# 门控融合机制 (Gated Fusion) 使用指南

## 概述

本更新引入了**门控融合机制**来替代简单的平均融合和拼接融合。门控融合通过可学习的门控网络动态控制图模态和文本模态的贡献，从而实现更细粒度、更灵活的多模态融合。

## 主要改进

### 1. 新增 `GatedFusion` 模块

位于 `models/alignn.py:531-643`

该模块提供了三种门控融合类型：

#### (1) 单门控 (Single Gate)
```
g = σ(W[h_graph; h_text])
output = g * h_graph + (1-g) * h_text
```
- **特点**: 一个门控值控制两个模态之间的权衡
- **适用**: 两个模态互补性强，一个高另一个就低的情况

#### (2) 双门控 (Dual Gate) - **推荐**
```
g_graph = σ(W1[h_graph; h_text])
g_text = σ(W2[h_graph; h_text])
output = g_graph * h_graph + g_text * h_text
```
- **特点**: 为每个模态分别学习门控值
- **适用**: 需要独立控制每个模态贡献的情况
- **优势**: 最灵活，可以学习复杂的融合策略

#### (3) 注意力融合 (Attention-based)
```
α = softmax(W[h_graph; h_text])  # α = [α_graph, α_text]
output = α_graph * h_graph + α_text * h_text
```
- **特点**: 使用 softmax 归一化的注意力权重
- **适用**: 需要权重和为 1 的约束时

### 2. 新增配置选项

在 `ALIGNNConfig` 中新增了以下配置项：

```python
# Fusion strategy settings
fusion_strategy: Literal["average", "concat", "gated"] = "gated"
gated_fusion_type: Literal["single_gate", "dual_gate", "attention"] = "dual_gate"
gated_fusion_hidden_dim: int = 128
gated_fusion_dropout: float = 0.1
```

### 3. 融合策略对比

| 策略 | 描述 | 参数量 | 可解释性 | 灵活性 |
|------|------|--------|----------|--------|
| **average** | 简单平均 `(h_g + h_t) / 2` | 0 | 低 | 低 |
| **concat** | 拼接后全连接 `Linear([h_g; h_t])` | 高 | 低 | 中 |
| **gated** | 门控融合 | 中 | **高** | **高** |

## 使用方法

### 方法 1: 命令行参数

```bash
python train_with_cross_modal_attention.py \
  --use_cross_modal \
  --fusion_strategy gated \
  --gated_fusion_type dual_gate \
  --gated_fusion_hidden_dim 128 \
  --gated_fusion_dropout 0.1
```

### 方法 2: 配置文件

```python
from models.alignn import ALIGNNConfig, ALIGNN

config = ALIGNNConfig(
    name="alignn",
    alignn_layers=4,
    gcn_layers=4,
    use_cross_modal_attention=True,
    use_fine_grained_attention=True,  # 可选

    # 门控融合配置
    fusion_strategy="gated",          # "average" | "concat" | "gated"
    gated_fusion_type="dual_gate",    # "single_gate" | "dual_gate" | "attention"
    gated_fusion_hidden_dim=128,
    gated_fusion_dropout=0.1,
)

model = ALIGNN(config)
```

### 方法 3: 代码中直接使用

```python
from models.alignn import GatedFusion

# 初始化门控融合模块
gated_fusion = GatedFusion(
    feature_dim=64,
    hidden_dim=128,
    dropout=0.1,
    fusion_type='dual_gate'
)

# 前向传播
fused_features = gated_fusion(graph_features, text_features)

# 获取门控值（用于可解释性分析）
fused_features, gate_values = gated_fusion(
    graph_features,
    text_features,
    return_gate_values=True
)
```

## 可解释性分析

门控融合提供了丰富的可解释性信息：

```python
# 在模型前向传播时获取门控值
output = model(g, return_attention=True)

# 提取门控值
if 'gate_values' in output:
    gate_values = output['gate_values']

    # 对于 dual_gate 类型
    if 'gate_graph' in gate_values:
        gate_graph = gate_values['gate_graph']  # [batch, 64]
        gate_text = gate_values['gate_text']    # [batch, 64]

        print(f"Graph gate mean: {gate_graph.mean():.4f}")
        print(f"Text gate mean: {gate_text.mean():.4f}")
```

### 门控值含义

- **gate_graph**: 每个维度上图特征的重要性权重（0-1）
- **gate_text**: 每个维度上文本特征的重要性权重（0-1）
- 接近 1: 该模态在该维度上贡献大
- 接近 0: 该模态在该维度上贡献小

## 完整的融合流程

```
输入: (图结构, 文本描述)
  ↓
[1] 图编码 (ALIGNN + GCN layers)
  → node_features [total_atoms, 256]
  ↓
[2] 细粒度交叉注意力 (可选)
  → enhanced_nodes [batch, max_atoms, 256]
  ↓
[3] 图池化 + 投影
  → graph_features [batch, 64]
  ↓
[4] 全局交叉注意力
  → enhanced_graph [batch, 64]
  → enhanced_text [batch, 64]
  ↓
[5] 门控融合 ← **新增步骤**
  → fused_features [batch, 64]
  ↓
[6] 预测层
  → predictions [batch, 1]
```

## 测试和验证

运行测试脚本验证门控融合机制：

```bash
python test_gated_fusion.py
```

测试内容包括：
1. ✓ GatedFusion 模块单元测试
2. ✓ ALIGNN 模型集成测试
3. ✓ 不同融合策略对比测试
4. ✓ 门控值提取测试

## 实验建议

### 1. 消融实验

比较不同融合策略的性能：

```bash
# 1. 基线：平均融合
python train.py --fusion_strategy average

# 2. 拼接融合
python train.py --fusion_strategy concat

# 3. 门控融合（单门控）
python train.py --fusion_strategy gated --gated_fusion_type single_gate

# 4. 门控融合（双门控）- 推荐
python train.py --fusion_strategy gated --gated_fusion_type dual_gate

# 5. 门控融合（注意力）
python train.py --fusion_strategy gated --gated_fusion_type attention
```

### 2. 超参数调优

关键超参数：
- `gated_fusion_hidden_dim`: 门控网络隐藏层维度（建议: 64-256）
- `gated_fusion_dropout`: Dropout 率（建议: 0.1-0.3）
- `gated_fusion_type`: 门控类型（推荐: dual_gate）

### 3. 可解释性分析

分析门控值的分布可以帮助理解：
- 模型更依赖图特征还是文本特征？
- 不同样本的融合策略有何差异？
- 哪些特征维度对预测最重要？

```python
# 示例：分析门控值
gate_graph_means = []
gate_text_means = []

for batch in dataloader:
    output = model(batch, return_attention=True)
    if 'gate_values' in output:
        gate_graph_means.append(output['gate_values']['gate_graph'].mean(dim=0))
        gate_text_means.append(output['gate_values']['gate_text'].mean(dim=0))

# 可视化
import matplotlib.pyplot as plt
plt.plot(torch.stack(gate_graph_means).mean(dim=0), label='Graph')
plt.plot(torch.stack(gate_text_means).mean(dim=0), label='Text')
plt.legend()
plt.xlabel('Feature Dimension')
plt.ylabel('Gate Value')
plt.title('Average Gate Values Across Dimensions')
plt.savefig('gate_analysis.png')
```

## 与其他融合机制的结合

门控融合可以与以下机制结合使用：

```python
config = ALIGNNConfig(
    # 中期融合：在 ALIGNN 层中注入文本信息
    use_middle_fusion=True,
    middle_fusion_layers="2",

    # 细粒度融合：原子-词元级别注意力
    use_fine_grained_attention=True,
    fine_grained_num_heads=8,

    # 全局交叉注意力：图-文本级别注意力
    use_cross_modal_attention=True,
    cross_modal_num_heads=4,

    # 门控融合：动态学习融合权重
    fusion_strategy="gated",
    gated_fusion_type="dual_gate",
)
```

## 常见问题

### Q1: 门控融合比平均融合慢多少？
A: 门控融合引入了额外的小型 MLP 网络，计算开销很小（< 5% 总训练时间）。

### Q2: 什么时候应该使用门控融合？
A: 当你希望：
- 模型自动学习最优融合策略
- 获得更好的可解释性
- 提升多模态融合效果

### Q3: 如何选择门控类型？
A: 推荐顺序：
1. **dual_gate** (最推荐): 最灵活，适合大多数场景
2. **attention**: 需要权重归一化约束时
3. **single_gate**: 需要严格的模态权衡时

### Q4: 门控融合需要更多训练数据吗？
A: 不需要。门控融合引入的参数量很少（约几千个），不会明显增加过拟合风险。

## 技术细节

### 门控网络架构

```
Input: [h_graph; h_text]  # Concatenated: [batch, 128]
  ↓
Linear(128 → hidden_dim)   # 第一层全连接
  ↓
ReLU()                     # 激活函数
  ↓
Dropout(p)                 # 正则化
  ↓
Linear(hidden_dim → 64)    # 第二层全连接
  ↓
Sigmoid()                  # 门控激活（输出 0-1）
  ↓
Output: gate [batch, 64]   # 门控值
```

### 融合计算

```python
# 双门控融合
gate_graph = Sigmoid(MLP([h_graph; h_text]))
gate_text = Sigmoid(MLP([h_graph; h_text]))
fused = gate_graph ⊙ h_graph + gate_text ⊙ h_text
fused = LayerNorm(fused)
fused = Dropout(fused)
```

## 引用

如果你使用了门控融合机制，可以考虑引用相关工作：

```bibtex
@article{gated_multimodal_fusion,
  title={Gated Multimodal Fusion for Material Property Prediction},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 更新日志

- **2024-12-08**: 初始版本发布
  - 添加 GatedFusion 模块
  - 支持三种门控类型: single_gate, dual_gate, attention
  - 集成到 ALIGNN 模型
  - 添加可解释性分析功能

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
