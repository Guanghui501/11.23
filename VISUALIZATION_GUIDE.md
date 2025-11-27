# 双模型可视化对比指南

## 快速开始

### 基本用法

```bash
python visualize_twin_models.py \
    --ckpt_base /path/to/baseline_model/best_model.pt \
    --ckpt_sga /path/to/sganet_model/best_model.pt \
    --dataset jarvis \
    --property mbj_bandgap \
    --feature_stage base  # 推荐：评估中期融合的独立贡献
```

### 完整参数

```bash
python visualize_twin_models.py \
    --ckpt_base ./baseline/best_model.pt \           # 基线模型（无中期融合）
    --ckpt_sga ./sganet/best_model.pt \              # SGANet（有中期融合）
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --max_samples 1000 \                             # 最大样本数（推荐500-2000）
    --batch_size 64 \
    --save_dir ./twin_model_visualization \          # 结果保存目录
    --device cuda \                                  # 使用GPU加速
    --feature_stage base                             # 特征提取阶段（见下方说明）
```

### 特征提取阶段选择 ⭐ NEW!

使用 `--feature_stage` 参数选择在哪个阶段提取特征：

#### `--feature_stage base` ⭐ **推荐用于评估中期融合**

```bash
python visualize_twin_models.py \
    --ckpt_base baseline.pt \
    --ckpt_sga sganet.pt \
    --feature_stage base
```

**提取时机**：GCN层后，所有注意力机制前

**对比内容**：
- Baseline: ALIGNN + GCN
- SGANet: ALIGNN + **中期融合** + GCN

**优点**：
- ✅ 差异**主要来自中期融合**
- ✅ 不受注意力机制影响
- ✅ 最能体现中期融合的独立贡献

**适用场景**：
- 验证中期融合模块的有效性
- 论文中的消融实验
- 理解融合如何改变GNN特征

#### `--feature_stage middle`

```bash
--feature_stage middle
```

**提取时机**：中期融合后立即提取（ALIGNN层结束，GCN层前）

**注意**：仅SGANet模型有此阶段，基线模型会回退到其他阶段

**适用场景**：
- 研究中期融合的即时影响
- 对比"融合后+GCN"vs"仅融合后"

#### `--feature_stage fine`

```bash
--feature_stage fine
```

**提取时机**：细粒度注意力后（原子-文本token交互后）

**对比内容**：
- 包含中期融合 + GCN + 细粒度注意力
- 不包含全局跨模态注意力

**适用场景**：
- 评估细粒度注意力的贡献
- 研究原子级别的跨模态交互

#### `--feature_stage final` (默认)

```bash
--feature_stage final  # 或省略（默认值）
```

**提取时机**：所有模块处理后的最终图特征

**对比内容**：
- 完整的图模态输出
- 包含所有模块的综合效果（但仅图特征）

**适用场景**：
- 评估整体模型的图表示质量
- 端到端的图特征对比

#### `--feature_stage fused` ⭐ **完整多模态融合**

```bash
--feature_stage fused
```

**提取时机**：最终的图特征 + 文本特征拼接

**对比内容**：
- `graph_features + text_features` 拼接
- 完整的多模态表示

**优点**：
- ✅ 评估**完整的多模态融合**效果
- ✅ 包含图和文本的所有信息
- ✅ 最接近模型实际用于预测的特征

**适用场景**：
- 评估多模态融合的整体质量
- 论文中展示最终模型的表示能力
- 对比纯图模型 vs 多模态模型

**特征维度**：
- base/final: [batch, 256] (仅图特征)
- fused: [batch, 512] (图256 + 文本256)

### 不同阶段对比的意义

| 阶段 | 对比内容 | 特征维度 | CKA预期 | 相关性预期 | 适用论文章节 |
|-----|---------|---------|---------|-----------|------------|
| **base** | 中期融合的纯粹影响 | [N, 256] | 0.85-0.95 | +10-20% | Ablation Study |
| **middle** | 融合后vs融合+GCN | [N, 256] | - | - | Module Analysis |
| **fine** | 细粒度注意力贡献 | [N, 256] | 0.90-0.97 | +5-15% | Attention Mechanism |
| **final** | 整体图特征性能 | [N, 256] | 0.92-0.98 | +8-15% | Main Results |
| **fused** ⭐ | 完整多模态融合 | [N, 512] | 0.88-0.96 | +15-25% | **Main Results** |

### 推荐的实验流程

```bash
# 1. 首先用 base 验证中期融合的独立贡献
python visualize_twin_models.py \
    --ckpt_base baseline.pt --ckpt_sga sganet.pt \
    --feature_stage base --save_dir ./viz_base

# 2. 用 fused 展示完整多模态融合效果（推荐用于主结果）
python visualize_twin_models.py \
    --ckpt_base baseline.pt --ckpt_sga sganet.pt \
    --feature_stage fused --save_dir ./viz_fused

# 3. (可选) 用 final 对比仅图特征的性能
python visualize_twin_models.py \
    --ckpt_base baseline.pt --ckpt_sga sganet.pt \
    --feature_stage final --save_dir ./viz_final

# 4. 对比分析：
#    - base:  中期融合的纯粹贡献（GCN后，注意力前）
#    - fused: 完整多模态表示（图+文本）⭐ 论文主结果
#    - final: 仅图特征表示
```

## 生成的图表

### 1. `tsne_comparison.png` - t-SNE 降维可视化

**作用**：二维空间展示特征分布的差异

**解读**：
- 左图：基线模型的特征空间
- 右图：SGANet 的特征空间
- 颜色：代表目标值（如带隙值）

**好的结果**：
- ✅ SGANet 的点更聚集（同颜色的点更紧密）
- ✅ 不同颜色的区域更清晰分离
- ✅ 渐变更平滑（相近颜色的点相邻）

**示例**：
```
如果看到 SGANet 的红色点（高带隙）和蓝色点（低带隙）分离得更清楚
→ 说明融合后的特征对带隙的区分能力更强
```

### 2. `pca_comparison.png` - PCA 主成分分析

**作用**：展示特征的主要变化方向

**解读**：
- PCA 保留了最大方差方向
- 与 t-SNE 相比，PCA 是线性变换，更能反映特征的真实结构

**好的结果**：
- ✅ SGANet 在 PC1/PC2 上的分离度更高
- ✅ 主成分能解释更多方差

### 3. `correlation_heatmap.png` - 特征-目标相关性热图

**作用**：可视化每个特征维度与目标的相关性

**图示**：
```
          D0    D1    D2    D3    ...   D49
Baseline  [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
SGANet    [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
```

**颜色含义**：
- 🔴 红色：正相关
- 🔵 蓝色：负相关
- ⚪ 白色：无相关

**好的结果**：
- ✅ SGANet 行有更多的深红/深蓝（强相关）
- ✅ 相关性模式更清晰

**实际案例**：
```
如果看到:
Baseline: [浅色, 浅色, 浅色, ...]  ← 相关性弱
SGANet:   [深色, 深色, 深色, ...]  ← 相关性强
→ 融合显著提升了特征的预测性
```

### 4. `metrics_comparison.png` - 综合指标对比

**包含4个子图**：

#### 4.1 Avg Pearson Corr（左上）
- **含义**：平均线性相关性
- **理想**：SGANet 的柱子更高
- **标注**：显示提升百分比

#### 4.2 Max Pearson Corr（右上）
- **含义**：最强相关的那个维度
- **理想**：SGANet > 0.6

#### 4.3 Feature Variance（左下）
- **含义**：特征的表达能力
- **理想**：SGANet 适度增加（10-30%）
- **警惕**：如果大幅下降 → 特征坍缩

#### 4.4 Feature Norm（右下）
- **含义**：特征向量的平均长度
- **理想**：SGANet 适度增加

**每个子图上方都标注了改进百分比**：
- 绿色框：正向改进
- 红色框：负向变化

### 5. `feature_distribution.png` - 特征范数分布

**作用**：检查特征是否坍缩或异常

**图示**：
```
Baseline:  [正态分布的直方图]  Mean: 5.2
SGANet:    [正态分布的直方图]  Mean: 6.1
```

**好的结果**：
- ✅ 两个分布都是正态的（钟形）
- ✅ SGANet 的均值适度增加
- ✅ 没有明显的双峰或长尾

**警惕信号**：
- ❌ SGANet 的分布严重右偏 → 可能过拟合
- ❌ 双峰分布 → 特征不稳定
- ❌ 范数接近0 → 特征坍缩

### 6. `comparison_report.txt` - 详细文本报告

**内容**：
```
╔═══════════════════════════════════════════════╗
║    Twin Model Feature Space Comparison       ║
╚═══════════════════════════════════════════════╝

1. Feature Structure Similarity (CKA Score)
   CKA Score: 0.9553
   ✓ Feature spaces are highly similar

2. Physical Property Correlation
   Avg Pearson Corr: 0.5608 → 0.6333 (+12.9%)
   Max Pearson Corr: 0.7234 → 0.8102 (+12.0%)

3. Feature Expressiveness
   Feature Variance: 0.4137 → 0.4852 (+17.3%)

4. Overall Assessment
   ✓ Structural Stability:  Excellent
   ✓ Predictive Quality:    Significantly Improved
   ✓ Feature Richness:      Enhanced

   Recommendation:
   ✓ Middle fusion module is effective and ready for publication!
```

## 解读指南

### 情况1: 理想结果 ✅

```
CKA Score: 0.92-0.98
Avg Pearson: +10% ~ +20%
Variance: +10% ~ +30%
```

**结论**：
- 中期融合有效！
- 特征空间稳定（CKA高）
- 预测性增强（相关性提升）
- 表达能力增强（方差增加）

**论文写法**：
> Our middle fusion module achieves a 12.9% improvement in feature-target correlation while maintaining high structural similarity (CKA=0.96), demonstrating effective and stable feature enhancement.

### 情况2: 过拟合警告 ⚠️

```
CKA Score: 0.65
Avg Pearson: +5%
Variance: -20%
```

**问题**：
- 特征空间改变太大（CKA低）
- 相关性提升不明显
- 方差降低（可能坍缩）

**建议**：
- 减小融合模块的dropout
- 减少融合层数
- 检查训练过程是否过拟合

### 情况3: 改进不足 ⚠️

```
CKA Score: 0.98
Avg Pearson: +2%
Variance: +3%
```

**问题**：
- 特征几乎没变（CKA太高）
- 改进微弱

**建议**：
- 增加融合层的表达能力
- 检查融合模块是否真的在起作用
- 可能需要调整融合位置

### 情况4: 不稳定 ❌

```
CKA Score: 0.45
Avg Pearson: -5%
Variance: +100%
```

**问题**：
- 特征完全重构（CKA很低）
- 性能下降
- 方差爆炸

**建议**：
- 检查模型训练是否收敛
- 可能需要重新训练
- 调整超参数

## 论文使用建议

### 推荐的图表组合

**主图（Main Figure）**：
- `tsne_comparison.png` - 最直观的可视化
- `metrics_comparison.png` - 量化对比

**补充材料（Supplementary）**：
- `correlation_heatmap.png` - 详细的维度分析
- `pca_comparison.png` - 线性降维对比
- `feature_distribution.png` - 分布检查

### 图注模板

```latex
\caption{
    \textbf{Feature space comparison between baseline and SGANet.}
    (a) t-SNE visualization of learned features, colored by target values.
    SGANet exhibits clearer cluster separation (CKA=0.96).
    (b) Quantitative metrics comparison showing 12.9\% improvement
    in feature-target correlation with enhanced feature variance.
}
```

## 常见问题

### Q1: 为什么 t-SNE 每次运行结果不同？

**A**: t-SNE 是随机算法，但我们固定了 `random_state=42`，所以结果应该是可重复的。

### Q2: 应该用多少样本？

**A**:
- 推荐：500-2000 样本
- 最少：200 样本（统计不够稳定）
- 最多：5000 样本（计算太慢）

### Q3: CKA 多少算合理？

**A**:
- **0.95-1.0**: 特征几乎相同（改进保守）
- **0.85-0.95**: 适度改变（理想范围）
- **0.7-0.85**: 较大改变（激进创新）
- **< 0.7**: 完全重构（可能不稳定）

### Q4: 相关性提升多少算显著？

**A**:
- **> 15%**: 非常显著
- **10-15%**: 显著
- **5-10%**: 中等
- **< 5%**: 微弱

### Q5: 可以对比多个模型吗？

**A**: 当前版本只支持两个模型。如果需要对比多个，可以运行多次并手动合并结果。

## 高级用法

### 批量对比多个属性

```bash
#!/bin/bash
PROPERTIES=("mbj_bandgap" "bulk_modulus_kv" "formation_energy_peratom")

for prop in "${PROPERTIES[@]}"; do
    echo "Processing $prop..."
    python visualize_twin_models.py \
        --ckpt_base baseline.pt \
        --ckpt_sga sganet.pt \
        --property $prop \
        --save_dir ./viz_$prop
done
```

### 生成 LaTeX 表格

从 `comparison_report.txt` 提取数据：

```python
import re

with open('comparison_report.txt') as f:
    report = f.read()

# 提取数值
cka = re.search(r'CKA Score: ([\d.]+)', report).group(1)
pearson_base = re.search(r'Avg Pearson Corr\s+([\d.]+)', report).group(1)
pearson_sga = re.search(r'([\d.]+)\s+\+', report).group(1)

# 生成 LaTeX 表格
latex = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Feature quality comparison}}
\\begin{{tabular}}{{lcc}}
\\hline
Metric & Baseline & SGANet \\\\
\\hline
CKA Score & - & {cka} \\\\
Avg Pearson & {pearson_base} & {pearson_sga} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
print(latex)
```

## 技术细节

### CKA 计算

```python
def centered_kernel_alignment(X, Y):
    X = X - X.mean(axis=0)  # 中心化
    Y = Y - Y.mean(axis=0)
    K = X @ X.T             # Gram 矩阵
    L = Y @ Y.T
    hsic = np.sum(K * L)    # HSIC
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return hsic / denom
```

### Pearson 相关系数

```python
from scipy.stats import pearsonr

# 对每个特征维度
correlations = []
for i in range(features.shape[1]):
    corr, p_value = pearsonr(features[:, i], targets)
    correlations.append(abs(corr))

avg_correlation = np.mean(correlations)
```

## 引用

如果使用 CKA 指标，请引用：

```
@inproceedings{kornblith2019similarity,
  title={Similarity of neural network representations revisited},
  author={Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle={ICML},
  year={2019}
}
```

## 总结

这个可视化脚本帮助你：

1. ✅ **验证融合有效性**：通过相关性提升
2. ✅ **检查稳定性**：通过 CKA 分数
3. ✅ **发现问题**：通过方差和分布
4. ✅ **准备论文图表**：高质量 300 DPI 图片
5. ✅ **生成报告**：自动化的文本总结

**最重要的指标排序**：
1. Avg Pearson Correlation（预测性）
2. CKA Score（稳定性）
3. Feature Variance（表达能力）
4. t-SNE 可视化（直观展示）
