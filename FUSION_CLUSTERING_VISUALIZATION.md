# 中期融合特征聚类可视化指南

展示有中期融合后，特征按晶系聚类得更完美。

## 📊 目标

创建定性可视化图，对比**有/无中期融合**时，模型学习到的特征在按晶系（crystal system）分组时的聚类质量。

### 预期效果

```
无中期融合                    有中期融合
┌─────────────┐            ┌─────────────┐
│ ● ● ●       │            │ ●●●         │
│   ● ●   ■■  │            │             │
│ ●     ■■■   │    VS      │   ■■■       │
│   ▲▲  ■     │            │    ■■       │
│  ▲▲▲    ◆◆  │            │       ◆◆◆   │
│   ▲    ◆◆◆  │            │      ◆◆◆    │
└─────────────┘            └─────────────┘
  混杂分散                    清晰聚类
```

**关键指标改善**：
- ✅ Silhouette Score ↑ (轮廓系数越高越好)
- ✅ Davies-Bouldin Index ↓ (DB指数越低越好)
- ✅ Calinski-Harabasz Score ↑ (CH分数越高越好)

## 🚀 快速开始

### 步骤 1: 准备两个模型

你需要训练两个模型用于对比：

#### 模型 A: 无中期融合
```bash
python train_with_cross_modal_attention.py \
    --root_dir /path/to/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_middle_fusion False \
    --output_dir output_no_middle_fusion
```

#### 模型 B: 有中期融合
```bash
python train_with_cross_modal_attention.py \
    --root_dir /path/to/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3" \
    --output_dir output_with_middle_fusion
```

### 步骤 2: 修改配置脚本

编辑 `compare_fusion_clustering.sh`：

```bash
# 修改这两行为实际的模型路径
MODEL_WITHOUT_FUSION="output_no_middle_fusion/best_model.pth"
MODEL_WITH_FUSION="output_with_middle_fusion/best_model.pth"
```

### 步骤 3: 运行可视化

```bash
chmod +x compare_fusion_clustering.sh
./compare_fusion_clustering.sh
```

或直接使用 Python 脚本：

```bash
python visualize_middle_fusion_clustering.py \
    --checkpoint_without_fusion output_no_middle_fusion/best_model.pth \
    --checkpoint_with_fusion output_with_middle_fusion/best_model.pth \
    --data_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    --n_samples 1000 \
    --reduction_method tsne \
    --output_dir fusion_clustering_results
```

## 📈 输出结果

### 1. 聚类对比图 (`clustering_comparison.png`)

左右对比图，展示：
- **左图**: 无中期融合的t-SNE/UMAP可视化
- **右图**: 有中期融合的t-SNE/UMAP可视化
- **颜色**: 不同颜色代表不同晶系
  - 🔴 红色: 立方 (cubic)
  - 🔵 蓝色: 六方 (hexagonal)
  - 🟢 绿色: 三方 (trigonal)
  - 🟠 橙色: 四方 (tetragonal)
  - 🟣 紫色: 正交 (orthorhombic)
  - 🔷 青色: 单斜 (monoclinic)
  - 🟤 深橙: 三斜 (triclinic)

### 2. 指标对比图 (`metrics_comparison.png`)

三个柱状图对比：

#### Silhouette Score (轮廓系数)
- **范围**: [-1, 1]
- **越高越好**: 接近 1 表示聚类紧密且分离良好
- **预期**: 中期融合后从 0.3 提升至 0.5+

#### Davies-Bouldin Index (DB指数)
- **范围**: [0, ∞)
- **越低越好**: 接近 0 表示簇间距离大、簇内距离小
- **预期**: 中期融合后从 2.0 降至 1.5-

#### Calinski-Harabasz Score (CH分数)
- **范围**: [0, ∞)
- **越高越好**: 值越大表示簇内紧密、簇间分散
- **预期**: 中期融合后显著提升

## 📊 聚类质量指标详解

### Silhouette Score (轮廓系数)
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

其中:
- a(i): 样本i到同簇其他点的平均距离
- b(i): 样本i到最近异簇点的平均距离
```

**解读**：
- s > 0.5: 聚类效果很好
- 0.2 < s < 0.5: 聚类效果一般
- s < 0.2: 聚类效果差

### Davies-Bouldin Index
```
DB = (1/k) Σ max_j≠i [(σ_i + σ_j) / d(c_i, c_j)]

其中:
- σ_i: 簇i的平均内部距离
- d(c_i, c_j): 簇中心间的距离
```

**解读**：
- DB < 1.0: 优秀
- 1.0 < DB < 2.0: 良好
- DB > 2.0: 需要改进

### Calinski-Harabasz Score
```
CH = [Σ_k n_k ||c_k - c||² / (k-1)] / [Σ_k Σ_i∈C_k ||x_i - c_k||² / (n-k)]

其中:
- 分子: 簇间离散度
- 分母: 簇内离散度
```

**解读**：
- 相对值，越大越好
- 通常 > 100 表示较好的聚类

## 🎯 预期改进示例

### 场景 1: 显著改进（理想情况）
```
指标                 无融合    有融合    改进
─────────────────────────────────────────
Silhouette Score     0.28  →  0.54   +93%  ✅
Davies-Bouldin       2.15  →  1.32   -39%  ✅
Calinski-Harabasz    156   →  387    +148% ✅
```

### 场景 2: 中等改进
```
指标                 无融合    有融合    改进
─────────────────────────────────────────
Silhouette Score     0.31  →  0.42   +35%  ✅
Davies-Bouldin       1.89  →  1.56   -17%  ✅
Calinski-Harabasz    201   →  265    +32%  ✅
```

### 场景 3: 无显著改进（你可能遇到的情况）
```
指标                 无融合    有融合    改进
─────────────────────────────────────────
Silhouette Score     0.35  →  0.36   +3%   ⚠️
Davies-Bouldin       1.72  →  1.69   -2%   ⚠️
Calinski-Harabasz    245   →  251    +2%   ⚠️
```

**如果改进不明显，可能的原因**：
1. 融合层位置不合适（尝试更早或更晚的层）
2. 融合机制设计需要优化
3. 晶系本身在特征空间中就不易区分
4. 超参数需要调整

## 🔧 故障排除

### 问题 1: 缺少晶系信息

**错误**: `未找到晶系信息`

**解决**:
```bash
# 确保CIF文件存在
ls /path/to/dataset/jarvis/mbj_bandgap/cif/*.cif | head

# 检查CIF文件是否包含晶格信息
python -c "
from jarvis.core.atoms import Atoms
atoms = Atoms.from_cif('sample.cif')
print(atoms.lattice.lattice_system)
"
```

### 问题 2: 特征提取失败

**错误**: `无法提取特征`

**解决**: 确保模型返回中间特征
- 检查模型的 `forward()` 方法是否支持 `return_features=True`
- 修改模型代码以返回特征字典

### 问题 3: UMAP未安装

**错误**: `UMAP未安装`

**解决**:
```bash
pip install umap-learn
```

## 📝 自定义选项

### 更改降维方法

```bash
# 使用 t-SNE (默认)
python visualize_middle_fusion_clustering.py ... --reduction_method tsne

# 使用 UMAP (更快)
python visualize_middle_fusion_clustering.py ... --reduction_method umap
```

### 调整样本数量

```bash
# 使用更多样本（更准确但更慢）
python visualize_middle_fusion_clustering.py ... --n_samples 2000

# 使用更少样本（更快）
python visualize_middle_fusion_clustering.py ... --n_samples 500
```

### 使用GPU加速

```bash
python visualize_middle_fusion_clustering.py ... --device cuda
```

## 📚 相关文献

如果你要在论文中使用这个可视化，可以引用：

1. **t-SNE**: van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
2. **UMAP**: McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
3. **Silhouette**: Rousseeuw (1987). "Silhouettes: A graphical aid to the interpretation"
4. **Davies-Bouldin**: Davies & Bouldin (1979). "A Cluster Separation Measure"

## 💡 提示

1. **对比要公平**: 确保两个模型除了中期融合外，其他超参数尽量一致
2. **多次运行**: t-SNE有随机性，建议运行多次取平均
3. **样本选择**: 使用测试集或验证集，避免过拟合影响
4. **配色方案**: 可以根据论文风格自定义 `CRYSTAL_SYSTEM_COLORS`

## ❓ 常见问题

**Q: 为什么我的图看不出明显差异？**

A: 可能原因：
- 两个模型性能本身就接近
- 晶系在你的数据集中分布不均
- 降维损失了关键信息（尝试增加perplexity或n_neighbors）

**Q: 可以用其他分组方式吗（不是晶系）？**

A: 可以！修改脚本，将晶系替换为：
- 空间群 (space group)
- 元素组成类别
- 性质值范围

**Q: 如何在论文中展示这个结果？**

A: 建议创建一个2x2的图：
- 左上: 无融合t-SNE
- 右上: 有融合t-SNE
- 左下: 无融合UMAP
- 右下: 有融合UMAP

## 🎓 使用示例（论文图表）

### Figure Caption 示例

```
Figure X: Feature space visualization comparing models with and without
middle fusion. (a,b) t-SNE projections of learned features colored by
crystal system for models without (a) and with (b) middle fusion.
(c) Quantitative clustering metrics showing improved feature separation
with middle fusion. Middle fusion enables better discrimination of
crystal systems in the learned feature space, as evidenced by higher
Silhouette scores (0.54 vs 0.28) and lower Davies-Bouldin indices
(1.32 vs 2.15).
```

---

**创建时间**: 2025-11-24
**适用版本**: train_mbj_with_optuna.py v1.0+
