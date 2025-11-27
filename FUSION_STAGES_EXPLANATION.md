# 融合阶段特征提取说明

## 问题：`compare_fusion_mechanisms.py` 生成的 graph 都是什么阶段的？

### 特征流程图

```
输入图
  ↓
ALIGNN 层（边、角更新）
  ↓
[中期融合层] ← 在这里应用 DynamicFusionModule (可选)
  ↓                 ↓
  ↓           【graph_middle】← 中期融合后的图特征
  ↓
GCN 层
  ↓
图 Readout + 投影
  ↓
【graph_base】← 基础图特征（投影后，所有注意力机制之前）
  ↓
[细粒度注意力] ← 原子-文本token级别交互 (可选)
  ↓
【graph_fine】← 细粒度注意力后的图特征
  ↓
[全局跨模态注意力] ← 图嵌入-文本嵌入级别交互 (可选)
  ↓
【graph_cross】← 全局注意力后的图特征
  ↓
【graph_final】← 最终图特征
```

## 关键回答

### 1. **`graph_base` 不是中期融合后的特征**
   - ❌ `graph_base` 是在 **GCN 层之后，但在所有注意力机制之前** 提取的
   - ❌ 它**不包含**中期融合的影响
   - ✅ 它是"纯粹的图神经网络处理结果"（ALIGNN + GCN + Readout + Projection）

### 2. **`graph_middle` 才是中期融合后的特征**
   - ✅ `graph_middle` 在 **ALIGNN 层中应用中期融合后立即提取**
   - ✅ 它**包含了** DynamicFusionModule 的影响
   - ✅ 它在 GCN 层之后、细粒度注意力之前

### 3. **为什么 `graph_base` 在 `graph_middle` 之后提取？**
   这是一个容易混淆的点：
   - `graph_base` 虽然命名为"base"，但在代码执行顺序上，它在 GCN 层之后才提取
   - 中期融合发生在 ALIGNN 层内部（第 965-968 行）
   - `graph_base` 是在整个 ALIGNN + GCN 流程完成后、但在注意力机制前提取的"基准特征"
   - 命名"base"是相对于注意力机制而言的（注意力前的基础状态）

## 完整对比

| 特征名称 | 提取时间点 | 包含中期融合？ | 包含细粒度注意力？ | 包含全局注意力？ |
|---------|----------|--------------|------------------|----------------|
| `graph_base` | GCN后 | ✅ 是 | ❌ 否 | ❌ 否 |
| `graph_middle` | 中期融合后 | ✅ 是 | ❌ 否 | ❌ 否 |
| `graph_fine` | 细粒度注意力后 | ✅ 是 | ✅ 是 | ❌ 否 |
| `graph_cross` | 全局注意力后 | ✅ 是 | ✅ 是 | ✅ 是 |
| `graph_final` | 最终输出 | ✅ 是 | ✅ 是 | ✅ 是 |

## 重要更正

**原始理解可能存在的误区：**
- ❌ 误解：`graph_base` 是融合前的特征
- ✅ 实际：`graph_base` 包含了中期融合（如果启用），但不包含任何注意力机制

**为了获得真正"无融合"的基准特征：**
- 需要在**不启用中期融合**的情况下运行模型
- 此时的 `graph_base` 才是纯 GNN 特征
- 或者需要在中期融合之前单独提取特征（目前代码未实现）

## 代码位置参考

- `graph_base` 提取：`models/alignn.py` 第 984-986 行
- `graph_middle` 提取：`models/alignn.py` 第 972-975 行
- `graph_fine` 提取：`models/alignn.py` 第 1042-1043 行
- `graph_cross` 提取：`models/alignn.py` 第 1106 行
- 中期融合应用：`models/alignn.py` 第 965-968 行（在 ALIGNN 层内部）

## 建议

如果要对比"完全无融合"vs"有中期融合"的效果：
1. 运行两次模型：一次 `use_middle_fusion=False`，一次 `use_middle_fusion=True`
2. 对比两次的 `graph_base` 特征
3. 这样才能看到纯粹的中期融合影响
