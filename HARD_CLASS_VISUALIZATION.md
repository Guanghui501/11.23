# 难分样本可视化 (Hard Class Subset Visualization)

## 概述

这个工具专门用于可视化最容易混淆的晶系对，通过聚焦于几何上最相似的晶系，可以更清楚地展示模型的区分能力。

## 为什么需要这个工具？

在原始的7大晶系可视化中，所有类别挤在一起会显得很乱，难以看出模型的真实区分能力。通过只选择**最容易混淆的晶系对**进行对比，可以：

1. **清晰展示模型优势** - 聚焦于最困难的分类任务
2. **定量评估分离度** - 计算类间距离、类内距离和分离比率
3. **减少视觉混乱** - 只显示2个类别，图像更清晰

## 最容易混淆的晶系对

根据晶体几何结构的相似性：

| 晶系对 | 相似度 | 描述 |
|--------|--------|------|
| **Cubic vs Tetragonal** | ⭐⭐⭐⭐⭐ | 四方晶系可以看作是立方晶系在一个方向上的拉伸/压缩 |
| Hexagonal vs Trigonal | ⭐⭐⭐⭐ | 三方晶系是六方晶系的子群 |
| Orthorhombic vs Tetragonal | ⭐⭐⭐ | 都有直角，但对称性不同 |

**推荐从 Cubic vs Tetragonal 开始**，这是最难区分的一对。

## 使用方法

### 基本用法

```bash
python visualize_hard_class_subset.py \
    --checkpoint_without_fusion models/model_no_fusion.pth \
    --checkpoint_with_fusion models/model_with_fusion.pth \
    --data_dir /path/to/dataset \
    --class_pair cubic,tetragonal \
    --output_dir hard_class_results
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint_without_fusion` | 无中期融合的模型路径 | 必需 |
| `--checkpoint_with_fusion` | 有中期融合的模型路径 | 必需 |
| `--data_dir` | 数据集目录 | 必需 |
| `--class_pair` | 要对比的晶系对（逗号分隔） | `cubic,tetragonal` |
| `--n_samples` | 初始采样数（筛选前） | 2000 |
| `--reduction_method` | 降维方法 (tsne/umap) | `tsne` |
| `--output_dir` | 输出目录 | `hard_class_results` |
| `--dataset` | 数据集名称 | `jarvis` |
| `--property` | 目标属性 | `mbj_bandgap` |

### 完整示例

```bash
# 示例1: Cubic vs Tetragonal (推荐，最难区分)
python visualize_hard_class_subset.py \
    --checkpoint_without_fusion outputs/alignn_no_fusion/best_model.pth \
    --checkpoint_with_fusion outputs/alignn_with_fusion/best_model.pth \
    --data_dir /home/user/11.23/jarvis_data \
    --class_pair cubic,tetragonal \
    --n_samples 3000 \
    --output_dir results/hard_class_cubic_tetragonal

# 示例2: Hexagonal vs Trigonal
python visualize_hard_class_subset.py \
    --checkpoint_without_fusion outputs/alignn_no_fusion/best_model.pth \
    --checkpoint_with_fusion outputs/alignn_with_fusion/best_model.pth \
    --data_dir /home/user/11.23/jarvis_data \
    --class_pair hexagonal,trigonal \
    --output_dir results/hard_class_hexagonal_trigonal

# 示例3: 使用UMAP降维（速度更快）
python visualize_hard_class_subset.py \
    --checkpoint_without_fusion outputs/alignn_no_fusion/best_model.pth \
    --checkpoint_with_fusion outputs/alignn_with_fusion/best_model.pth \
    --data_dir /home/user/11.23/jarvis_data \
    --class_pair cubic,tetragonal \
    --reduction_method umap \
    --output_dir results/hard_class_umap
```

## 输出文件

运行后会生成以下文件：

### 1. 主可视化图 (`hard_class_cubic_vs_tetragonal.png`)

左右对比展示：
- **左图**: 无中期融合模型的特征分布
- **右图**: 有中期融合模型的特征分布

每个图包含：
- 两个晶系的散点图（不同颜色）
- Silhouette Score（轮廓系数）
- Separation Ratio（分离比率）
- Inter-class Distance（类间距离）
- Intra-class Distance（类内距离）

### 2. 分离度指标图 (`separation_metrics_cubic_vs_tetragonal.png`)

三个柱状图对比：
1. **Inter-class Distance（类间距离）** - 越大越好
   - 衡量两个类别的质心之间的距离

2. **Average Intra-class Distance（平均类内距离）** - 越小越好
   - 衡量每个类别内部的紧凑程度

3. **Separation Ratio（分离比率）** - 越大越好
   - 公式: `类间距离 / (类内距离1 + 类内距离2)`
   - 综合衡量类别的可分性

### 3. 结果摘要 (`summary_cubic_vs_tetragonal.txt`)

包含：
- 数据集信息
- 样本数统计
- 聚类指标对比（Silhouette, Davies-Bouldin, Calinski-Harabasz）
- 类分离度详细指标
- 各项指标的改进百分比

## 指标解读

### 1. Silhouette Score（轮廓系数）

- **范围**: [-1, 1]
- **含义**: 衡量样本与自己的簇相似度，与其他簇的差异度
- **期望**: 越接近1越好（说明聚类质量高）

### 2. Davies-Bouldin Index（DB指数）

- **范围**: [0, ∞)
- **含义**: 聚类内部距离与聚类间距离的比值
- **期望**: 越接近0越好（说明类内紧凑，类间分离）

### 3. Calinski-Harabasz Score（CH指数）

- **范围**: [0, ∞)
- **含义**: 类间方差与类内方差的比值
- **期望**: 越大越好（说明类间差异大，类内紧凑）

### 4. Separation Ratio（分离比率）

- **公式**: `Inter-class Distance / (Intra-class Distance 1 + Intra-class Distance 2)`
- **含义**: 综合衡量类别可分性
- **期望**: 越大越好
  - 分子大 → 两个类别离得远
  - 分母小 → 每个类别内部紧凑

## 如何判断模型改进？

如果中期融合模型相比无融合模型有以下表现，说明改进显著：

✅ **Silhouette Score** ↑ (增加)
✅ **Davies-Bouldin Index** ↓ (减少)
✅ **Calinski-Harabasz Score** ↑ (增加)
✅ **Separation Ratio** ↑ (增加)
✅ **Inter-class Distance** ↑ (增加)
✅ **Intra-class Distance** ↓ (减少)

在可视化图中，好的模型应该：
- 两个颜色的点云分离明显（不重叠）
- 每个点云内部紧凑（不分散）

## 与全晶系可视化的对比

| 特性 | 全晶系可视化 | 难分样本可视化 |
|------|--------------|----------------|
| 类别数 | 7个 | 2个 |
| 视觉清晰度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 区分难度 | 混合（简单+困难） | 仅最困难的对 |
| 适用场景 | 整体性能概览 | 细节优势展示 |
| 指标丰富度 | 基础聚类指标 | 聚类+分离度指标 |

## 推荐工作流

1. **先运行全晶系可视化** - 了解整体性能
   ```bash
   python visualize_middle_fusion_clustering.py ...
   ```

2. **再运行难分样本可视化** - 细节展示优势
   ```bash
   python visualize_hard_class_subset.py --class_pair cubic,tetragonal ...
   ```

3. **可选: 多个难分对** - 全面评估
   ```bash
   # Cubic vs Tetragonal
   python visualize_hard_class_subset.py --class_pair cubic,tetragonal ...

   # Hexagonal vs Trigonal
   python visualize_hard_class_subset.py --class_pair hexagonal,trigonal ...
   ```

## 技术细节

### 类分离度计算

```python
# 类间距离: 两个类质心之间的欧氏距离
centroid1 = features_class1.mean(axis=0)
centroid2 = features_class2.mean(axis=0)
inter_class_dist = ||centroid1 - centroid2||

# 类内距离: 样本到自己类质心的平均距离
intra_class_dist = mean(||sample - centroid|| for sample in class)

# 分离比率
separation_ratio = inter_class_dist / (intra_class_dist_1 + intra_class_dist_2)
```

### 降维方法选择

- **t-SNE**:
  - 优点: 保持局部结构，可视化效果好
  - 缺点: 速度较慢
  - 推荐: 样本数 < 5000

- **UMAP**:
  - 优点: 速度快，保持全局和局部结构
  - 缺点: 需要额外安装 `pip install umap-learn`
  - 推荐: 样本数 > 5000

## 常见问题

### Q1: 筛选后样本数太少怎么办？

增加 `--n_samples` 参数：
```bash
python visualize_hard_class_subset.py \
    --n_samples 5000 \  # 增加初始采样数
    ...
```

### Q2: 如何选择其他晶系对？

修改 `--class_pair` 参数：
```bash
--class_pair hexagonal,trigonal
--class_pair orthorhombic,tetragonal
--class_pair monoclinic,triclinic
```

### Q3: 如何加速降维？

使用UMAP代替t-SNE：
```bash
--reduction_method umap
```

### Q4: 图像太小看不清？

修改脚本中的 `figsize` 参数，或者使用更高的DPI。

## 依赖项

```bash
pip install numpy pandas torch matplotlib seaborn scikit-learn scipy tqdm
pip install umap-learn  # 可选，用于UMAP降维
```

## 性能建议

- **小数据集 (< 1000样本)**: t-SNE, perplexity=30
- **中等数据集 (1000-5000样本)**: t-SNE或UMAP
- **大数据集 (> 5000样本)**: UMAP, 或使用PCA预降维

## 总结

难分样本可视化是展示模型细节优势的利器：

1. **专注最难任务** - Cubic vs Tetragonal
2. **清晰直观** - 只有2个类别
3. **指标丰富** - 聚类+分离度
4. **定量评估** - 准确计算改进幅度

配合全晶系可视化使用，可以全面展示模型性能！
