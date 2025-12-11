# 论文写作实验清单

## 实验任务列表

### ✅ 已完成
- [x] 门控跨模态注意力模型训练
- [x] 最佳模型checkpoint保存

### 📋 待完成实验

#### 1. 性能对比实验
- [ ] CGCNN baseline训练
- [ ] ALIGNN baseline训练（无文本）
- [ ] ALIGNN + Text Concat训练
- [ ] 在3个数据集上重复：
  - [ ] Formation Energy
  - [ ] Band Gap
  - [ ] Elastic Modulus (如果有数据)

#### 2. 消融实验
- [ ] Graph Only (ALIGNN)
- [ ] + Text (Simple Concat)
- [ ] + Cross-Attention (No Gate)
- [ ] + Gate Mechanism (当前模型)
- [ ] + Fine-Grained Attention
- [ ] + Contrastive Loss

#### 3. 超参数分析
- [ ] 注意力头数: 1, 2, 4, 8
- [ ] 隐藏层维度: 128, 256, 512
- [ ] Dropout率: 0.0, 0.1, 0.2

#### 4. 可解释性分析
- [ ] 提取100个样本的注意力权重
- [ ] 计算注意力统计（均值、方差、分布）
- [ ] 选择3个代表性材料做Case Study
- [ ] 原子重要性计算（梯度法）
- [ ] 特征空间t-SNE可视化（500个样本）

#### 5. 错误分析
- [ ] 按材料类别统计预测误差
- [ ] 识别高误差样本
- [ ] 分析门控值与预测误差的关系

---

## 代码生成计划

### 脚本1: 性能对比实验
```bash
./run_baseline_comparison.sh
```
- 训练所有baseline模型
- 自动记录结果到CSV

### 脚本2: 消融实验
```bash
./run_ablation_study.sh
```
- 系统性关闭各个模块
- 生成Table 5数据

### 脚本3: 生成Figure 3-9
```bash
python generate_paper_figures.py \
  --checkpoint best_model.pt \
  --output_dir paper_figures/
```
- 自动生成所有可视化图片
- 保存为高分辨率PDF/PNG

### 脚本4: 统计表格生成
```bash
python generate_paper_tables.py \
  --results_dir experiments/ \
  --output tables.tex
```
- 从实验结果自动生成LaTeX表格

---

## 预计工作量

| 任务 | 预计GPU时间 | 预计人工时间 |
|------|-------------|--------------|
| Baseline对比 | 48 GPU小时 | 4小时（设置实验） |
| 消融实验 | 36 GPU小时 | 3小时 |
| 超参数分析 | 24 GPU小时 | 2小时 |
| 可解释性分析 | 2 GPU小时 | 8小时（写代码+分析） |
| 图表生成 | - | 12小时（编写+调试） |
| **总计** | **~110 GPU小时** | **~30小时** |

---

## 期刊投稿建议

### 推荐期刊（按影响因子）：

**顶级期刊** (IF > 10):
1. **Nature Communications** (IF: 16.6)
   - 接受计算材料+机器学习
   - 需要强可解释性和实验验证

2. **Advanced Materials** (IF: 29.4)
   - 偏重材料应用
   - 需要实验验证

**一区期刊** (IF: 5-10):
3. **npj Computational Materials** (IF: 9.7) ⭐ **强烈推荐**
   - Nature旗下，纯计算期刊
   - 接受方法创新论文
   - 审稿周期短（2-3个月）

4. **Materials Today** (IF: 31.6)
   - 综述+研究文章
   - 需要广泛应用

5. **Chemistry of Materials** (IF: 8.6)
   - ACS旗下
   - 接受计算+实验

**二区期刊** (IF: 3-5):
6. **Journal of Chemical Information and Modeling** (IF: 5.6) ⭐ **推荐**
   - 专注机器学习方法
   - 接受纯计算论文

7. **Journal of Physical Chemistry C** (IF: 3.7)
   - 材料物理化学
   - 接受DFT+ML

8. **Computational Materials Science** (IF: 3.3)
   - 纯计算期刊
   - 容易接受

### 投稿策略：

**优先级1** (冲击高分):
- npj Computational Materials
  - 强调：门控机制创新 + 可解释性

**优先级2** (保底):
- Journal of Chemical Information and Modeling
  - 强调：多模态学习方法

**优先级3** (快速发表):
- Computational Materials Science
  - 如果前两个被拒

---

## 论文写作时间表

| 周次 | 任务 |
|------|------|
| Week 1-2 | 完成所有实验（baseline, 消融, 超参数） |
| Week 3 | 可解释性分析 + 图表生成 |
| Week 4 | 撰写初稿（Introduction + Method） |
| Week 5 | 撰写实验和讨论部分 |
| Week 6 | 修改润色 + 导师审阅 |
| Week 7 | 提交投稿 |

---

## 需要的代码支持

我可以立即帮你编写以下脚本：

1. ✅ `diagnose_gated_attention.py` (已完成)
2. ⏳ `generate_paper_figures.py` (生成所有图表)
3. ⏳ `generate_paper_tables.py` (生成LaTeX表格)
4. ⏳ `run_ablation_study.sh` (自动化消融实验)
5. ⏳ `error_analysis.py` (错误分析和案例研究)

**需要哪个脚本？我现在就可以开始编写！**
