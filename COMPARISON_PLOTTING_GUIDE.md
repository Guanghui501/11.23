# 模型对比作图指南

## 🎯 快速开始（推荐）

### 一键运行所有步骤

```bash
# 1. 修改脚本中的模型路径
vim quick_compare_and_plot.sh

# 2. 运行
chmod +x quick_compare_and_plot.sh
./quick_compare_and_plot.sh
```

**自动完成**：
- ✅ 预生成遮挡数据（所有策略和遮挡率）
- ✅ 评估模型1（使用预生成数据）
- ✅ 评估模型2（使用相同数据）
- ✅ 生成所有对比图表

**输出**：
```
comparison_plots/
├── comparison_random_token.png      # random_token策略对比图
├── comparison_sentence.png          # sentence策略对比图
├── comparison_keep_keywords.png     # keep_keywords策略对比图
└── comprehensive_comparison.png     # 综合对比图（重点）
```

---

## 📊 生成的图表

### 1. 单个策略对比图 (4个子图)

**文件**: `comparison_{strategy}.png`

**包含内容**：
- **左上**: MAE对比
  - 两条曲线对比
  - 标注100%遮挡的改进百分比
- **右上**: RMSE对比
- **左下**: R²对比
- **右下**: MAE改进百分比柱状图
  - 绿色 = 模型2更好
  - 红色 = 模型1更好

**示例**：
```
comparison_random_token.png:
┌─────────────────────────┬─────────────────────────┐
│ MAE对比                 │ RMSE对比                │
│ ○-○ 中期融合+跨模态+细粒度│                         │
│ □-□ 跨模态+细粒度        │                         │
├─────────────────────────┼─────────────────────────┤
│ R²对比                  │ MAE改进百分比           │
│                         │ [柱状图]                │
└─────────────────────────┴─────────────────────────┘
```

### 2. 综合对比图 (4个子图) ⭐重点

**文件**: `comprehensive_comparison.png`

**包含内容**：
- **左上**: 所有策略MAE对比
  - 6条曲线（3策略 × 2模型）
  - 一目了然看出哪个策略最鲁棒

- **右上**: 100%遮挡柱状图对比
  - 每个策略的柱状图对比
  - 清楚显示极端情况表现

- **左下**: 平均改进百分比
  - 每个策略的平均改进
  - 绿色=模型2更好，红色=模型1更好

- **右下**: 统计摘要文本
  - 0%遮挡性能
  - 100%遮挡性能
  - 总体评价

**示例输出**：
```
统计摘要
==============================================================

random_token:
  0% 遮挡:   模型1=0.251, 模型2=0.274 (-9.2%)
  100% 遮挡: 模型1=1.930, 模型2=0.780 (+59.6%) ✓

sentence:
  0% 遮挡:   模型1=0.251, 模型2=0.274 (-9.2%)
  100% 遮挡: 模型1=0.340, 模型2=0.290 (+14.7%) ✓

==============================================================
总体评价:

✓ 跨模态+细粒度 平均性能更好
  平均改进: 25.3%
```

---

## 🔧 手动步骤（分步执行）

### 步骤1: 预生成遮挡数据

```bash
python pregenerate_masked_dataset.py \
    --input_data ./corrected_test_set/test.pkl \
    --output_dir ./masked_datasets \
    --strategies random_token sentence keep_keywords \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --seed 42
```

### 步骤2: 评估模型1

```bash
# 示例：评估 random_token 50%遮挡
python evaluate_with_premasked_data.py \
    --checkpoint /path/to/model1.pt \
    --masked_data ./masked_datasets/random_token_0.5.pkl \
    --output_file ./results/model1_random_token_0.5.json
```

### 步骤3: 评估模型2

```bash
# 使用相同的遮挡数据
python evaluate_with_premasked_data.py \
    --checkpoint /path/to/model2.pt \
    --masked_data ./masked_datasets/random_token_0.5.pkl \
    --output_file ./results/model2_random_token_0.5.json
```

### 步骤4: 生成对比图

```bash
python plot_model_comparison.py \
    --model1_results "./results/model1_*.json" \
    --model2_results "./results/model2_*.json" \
    --model1_name "中期融合+跨模态+细粒度" \
    --model2_name "跨模态+细粒度" \
    --output_dir ./comparison_plots
```

---

## 📋 对比表格（终端输出）

运行 `plot_model_comparison.py` 后，会在终端输出详细的对比表格：

```
====================================================================================================
模型对比摘要
====================================================================================================

策略: random_token
----------------------------------------------------------------------------------------------------
遮挡率     中期融合+跨模态+细粒度 MAE   跨模态+细粒度 MAE         改进            备注
----------------------------------------------------------------------------------------------------
0%         0.2510                 0.2740                 -9.17%          ✗ 更差
50%        0.3921                 0.3154                 19.55%          ✓ 更好
100%       1.9300                 0.7800                 59.59%          ✓ 更好

策略: sentence
----------------------------------------------------------------------------------------------------
遮挡率     中期融合+跨模态+细粒度 MAE   跨模态+细粒度 MAE         改进            备注
----------------------------------------------------------------------------------------------------
0%         0.2510                 0.2740                 -9.17%          ✗ 更差
50%        0.2879                 0.2654                 7.81%           ✓ 更好
100%       0.3401                 0.2901                 14.70%          ✓ 更好

====================================================================================================
```

---

## 🎨 图表自定义

### 修改颜色

编辑 `plot_model_comparison.py`:

```python
# 第85行左右
color1 = '#2E86AB'  # 蓝色 - 模型1
color2 = '#A23B72'  # 紫色 - 模型2

# 修改为你喜欢的颜色
color1 = '#FF6B6B'  # 红色
color2 = '#4ECDC4'  # 青色
```

### 调整图表大小

```python
# 第128行左右（单个策略对比图）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 修改这里

# 第245行左右（综合对比图）
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 修改这里
```

### 修改字体大小

```python
# 全局字体设置（第14-16行）
plt.rcParams['font.size'] = 12  # 添加这行

# 或单独修改
ax1.set_xlabel('遮挡率 (%)', fontsize=14)  # 修改 fontsize
```

---

## 📸 查看图表

### Linux

```bash
# 使用eog (Eye of GNOME)
eog ./comparison_plots/comprehensive_comparison.png

# 或使用其他查看器
feh ./comparison_plots/comprehensive_comparison.png
display ./comparison_plots/comprehensive_comparison.png
```

### 批量查看

```bash
# 打开所有对比图
eog ./comparison_plots/*.png
```

---

## 💡 使用技巧

### 技巧1: 只对比关键遮挡率

如果只想对比 0%, 50%, 100%：

```bash
# 修改 quick_compare_and_plot.sh
RATIOS="0.0 0.5 1.0"  # 只生成3个遮挡率
```

**优势**：
- 更快（生成数据和评估都更快）
- 图表更清晰（点少更易读）

### 技巧2: 只对比单个策略

```bash
# 只对比 sentence 策略
STRATEGIES="sentence"
```

### 技巧3: 增加遮挡率精度

```bash
# 更细粒度的遮挡率
RATIOS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
```

**适合**：
- 寻找性能突变点
- 绘制更平滑的曲线

### 技巧4: 保存高分辨率图片

修改 `plot_model_comparison.py`:

```python
# 第226行和第418行
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 改为 dpi=600
```

---

## 🔍 结果解读

### 看什么指标？

#### 1. MAE（平均绝对误差）
- **越小越好**
- 最直观的性能指标
- **重点关注**: 100%遮挡时的MAE

#### 2. RMSE（均方根误差）
- **越小越好**
- 对大误差更敏感
- 适合检测异常值

#### 3. R²（决定系数）
- **越接近1越好**，负数表示性能很差
- R² < 0 表示模型比简单平均还差
- **重点关注**: R²何时变负

#### 4. 改进百分比
- **正数表示模型2更好**
- **负数表示模型1更好**
- **重点关注**: 100%遮挡的改进

### 关键问题

#### Q1: 哪个模型在极端情况下更鲁棒？

**看**: 100%遮挡的MAE

```
模型1: MAE=1.93 (崩溃)
模型2: MAE=0.78 (正常) ✓
→ 模型2更鲁棒
```

#### Q2: 哪个策略最鲁棒？

**看**: 综合对比图左上角的所有曲线

```
sentence:   MAE从0.25→0.34 (增加36%)
random_token: MAE从0.25→0.78 (增加212%) ✗
→ sentence策略最鲁棒
```

#### Q3: Middle Fusion是否有效？

**看**: 0%遮挡 vs 100%遮挡的对比

```
0%遮挡:   有Middle=0.251, 无Middle=0.274 (Middle好9%) ✓
100%遮挡: 有Middle=1.930, 无Middle=0.780 (Middle差60%) ✗
→ Middle在正常情况好，极端情况差
```

---

## 📊 示例分析

### 示例1: 发现Middle Fusion的问题

**图表**: `comprehensive_comparison.png` 右上角

观察到：
```
100%遮挡柱状图:
  random_token:  模型1=1.93, 模型2=0.78
  sentence:      模型1=0.34, 模型2=0.29
  keep_keywords: 模型1=12.32, 模型2=9.93
```

**结论**:
- Middle Fusion在100%遮挡时反而更差（除了sentence策略）
- 原因：固定融合权重无法适应文本质量变化
- **解决方案**: 使用Gated Cross-Attention（自适应权重）

### 示例2: 选择最佳策略

**图表**: `comprehensive_comparison.png` 左下角

```
平均改进百分比:
  random_token:  +35.2% ✓
  sentence:      +12.8% ✓
  keep_keywords: +15.7% ✓
```

**结论**:
- 所有策略下模型2都更好
- random_token改进最大（因为模型1在此策略下崩溃严重）
- sentence最鲁棒（改进虽小但全程稳定）

---

## 🎯 最佳实践

### 1. 完整评估流程

```bash
# 1. 预生成数据（覆盖所有情况）
python pregenerate_masked_dataset.py \
    --strategies random_token sentence keep_keywords \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# 2. 评估所有组合
for strategy in random_token sentence keep_keywords; do
    for ratio in 0.0 0.5 1.0; do
        # 评估模型1
        python evaluate_with_premasked_data.py ...

        # 评估模型2
        python evaluate_with_premasked_data.py ...
    done
done

# 3. 生成图表
python plot_model_comparison.py ...
```

### 2. 重点对比（快速版）

```bash
# 只对比关键遮挡率和策略
STRATEGIES="random_token sentence"
RATIOS="0.0 0.5 1.0"

./quick_compare_and_plot.sh
```

### 3. 发表论文用

```bash
# 高分辨率 + 完整遮挡率
python pregenerate_masked_dataset.py \
    --ratios 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# 修改 dpi=600
vim plot_model_comparison.py  # 修改 dpi

python plot_model_comparison.py ...
```

---

## 🐛 故障排除

### 问题1: 中文显示乱码

**解决**：

```python
# 在 plot_model_comparison.py 开头添加
import matplotlib
matplotlib.rc('font', family='SimHei')  # Linux
# 或
matplotlib.rc('font', family='Microsoft YaHei')  # Windows
```

### 问题2: 图表太小看不清

**解决**：

```python
# 增大图表尺寸
fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # 默认 (16, 12)
```

### 问题3: 结果文件找不到

**检查**：

```bash
# 确认文件存在
ls -lh ./results/model1_*.json
ls -lh ./results/model2_*.json

# 检查通配符
python plot_model_comparison.py \
    --model1_results "./results/model1_*.json" \  # 使用引号
    --model2_results "./results/model2_*.json"
```

### 问题4: 内存不足

**解决**：

```bash
# 减少batch size
python evaluate_with_premasked_data.py \
    --batch_size 32  # 默认64
```

---

## 📝 完整示例

```bash
# 1. 配置环境
export CUDA_VISIBLE_DEVICES=0

# 2. 修改脚本
vim quick_compare_and_plot.sh
# 修改MODEL1和MODEL2路径

# 3. 运行
chmod +x quick_compare_and_plot.sh
./quick_compare_and_plot.sh

# 4. 查看结果
eog comparison_plots/comprehensive_comparison.png

# 5. 保存论文用图（高分辨率）
# 修改dpi=600后重新生成
python plot_model_comparison.py \
    --model1_results "./comparison_results/model1_*.json" \
    --model2_results "./comparison_results/model2_*.json" \
    --model1_name "中期融合+跨模态+细粒度" \
    --model2_name "跨模态+细粒度" \
    --output_dir ./paper_figures
```

---

## ✅ 检查清单

在运行前确认：

- [ ] 测试集路径正确
- [ ] 两个模型checkpoint路径正确
- [ ] 输出目录有写入权限
- [ ] CUDA可用（或使用CPU）
- [ ] 磁盘空间充足（至少2GB）
- [ ] 安装了必要的Python包：
  ```bash
  pip install matplotlib seaborn numpy
  ```

---

**开始对比**：

```bash
./quick_compare_and_plot.sh
```

就这么简单！🎉
