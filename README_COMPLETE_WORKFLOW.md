# MBJ Band Gap 模型完整评估工作流程

## 📋 目录

1. [概述](#概述)
2. [准备工作](#准备工作)
3. [步骤1: 数据预处理](#步骤1-数据预处理)
4. [步骤2: 文本遮挡评估](#步骤2-文本遮挡评估)
5. [步骤3: 结果分析](#步骤3-结果分析)
6. [完整命令参考](#完整命令参考)
7. [故障排除](#故障排除)

## 概述

本文档提供了对你的 **MBJ Band Gap 模型**进行文本遮挡鲁棒性评估的完整工作流程。

**模型信息**：
- Checkpoint: `/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt`
- 数据集: JARVIS
- 属性: mbj_bandgap
- 架构: Middle Fusion + Fine-grained Attention

## 准备工作

### 1. 文件清单

将以下文件复制到你的服务器工作目录：

#### 核心脚本
- ✅ `preprocess_dataset.py` - 数据预处理脚本
- ✅ `preprocess_mbj_bandgap.sh` - MBJ Band Gap预处理快捷脚本
- ✅ `evaluate_text_masking.py` - 文本遮挡评估脚本
- ✅ `compare_masking_strategies.py` - 策略对比脚本

#### 快捷脚本
- ✅ `quick_test_masking.sh` - 快速测试脚本
- ✅ `run_masking_eval_mbj_bandgap.sh` - 完整评估脚本

#### 文档
- ✅ `PREPROCESSING_GUIDE.md` - 预处理指南
- ✅ `TEXT_MASKING_EVALUATION_GUIDE.md` - 评估指南
- ✅ `README_MBJ_BANDGAP_MASKING_EVAL.md` - MBJ专用指南
- ✅ `README_COMPLETE_WORKFLOW.md` - 本文档

### 2. 依赖检查

确保已安装所需的Python包：

```bash
# 检查关键依赖
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import dgl; print('DGL:', dgl.__version__)"
python -c "import jarvis; print('JARVIS:', jarvis.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

### 3. 数据准备

确认数据集存在：

```bash
# 检查数据集目录
ls -la /public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap/

# 应该包含:
#   cif/              (CIF文件目录)
#   description.csv   (描述文件)
```

## 步骤1: 数据预处理

### 为什么需要预处理？

- **加载速度**: 原始数据加载需要10-30分钟，预处理后只需10-30秒（**100倍提速**）
- **一次预处理**: 预处理数据可用于多次评估和实验
- **必需步骤**: 文本遮挡评估要求使用预处理数据

### 1.1 编辑配置

```bash
vim preprocess_mbj_bandgap.sh
```

确认以下路径正确：
```bash
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
OUTPUT_DIR="/public/home/ghzhang/preprocessed_data"
```

### 1.2 运行预处理

```bash
chmod +x preprocess_mbj_bandgap.sh
./preprocess_mbj_bandgap.sh
```

**预计时间**: 10-30分钟（取决于数据集大小）

### 1.3 验证预处理结果

```bash
# 检查输出文件
ls -lh /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/

# 应该看到：
# train.pkl   (最大，~80% 数据)
# val.pkl     (中等，~10% 数据)
# test.pkl    (中等，~10% 数据)
# README.txt  (数据说明)
```

**验证数据内容**：
```python
import pickle

with open('/public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/test.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"测试集样本数: {len(test_data)}")
print(f"第一个样本: {test_data[0].keys()}")
# 应输出: dict_keys(['id', 'graph', 'line_graph', 'text', 'target'])
```

### 1.4 如果预处理失败

查看常见问题：
- [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) - 详细故障排除

常见错误：
1. **"找不到vocab_mappings.txt"** → 确保文件在当前目录或`crysmmnet-main/src/`
2. **"CIF目录不存在"** → 检查`ROOT_DIR`路径
3. **内存不足** → 使用更大内存的机器

## 步骤2: 文本遮挡评估

### 2.1 快速测试（推荐先运行）

验证所有配置正确：

```bash
chmod +x quick_test_masking.sh
./quick_test_masking.sh
```

这会：
- 只测试`random_token`策略
- 只使用3个遮挡率 (0%, 50%, 100%)
- 快速验证（~5分钟）

**如果成功**，你会看到：
```
✓ 测试成功！
结果保存在: ./test_masking_output
```

### 2.2 完整评估

快速测试成功后，运行完整评估：

```bash
chmod +x run_masking_eval_mbj_bandgap.sh
./run_masking_eval_mbj_bandgap.sh
```

这会：
- 测试所有**5种遮挡策略**：
  - `random_token`: 随机token遮挡
  - `random_word`: 随机单词遮挡
  - `random_chunk`: 连续文本块遮挡
  - `sentence`: 句子遮挡
  - `keep_keywords`: 只保留关键词
- 每种策略测试**11个遮挡率** (0%, 10%, ..., 100%)
- 自动生成对比报告

**预计时间**: 30-60分钟

### 2.3 监控进度

如果在后台运行：
```bash
# 后台运行
nohup ./run_masking_eval_mbj_bandgap.sh > masking_eval.log 2>&1 &

# 查看进度
tail -f masking_eval.log

# 查看哪个策略正在运行
ps aux | grep evaluate_text_masking
```

## 步骤3: 结果分析

### 3.1 查看单个策略结果

```bash
# 查看random_token策略的文本报告
cat masking_evaluation_mbj_bandgap/random_token/text_masking_report_random_token.txt
```

**报告内容**：
- 各遮挡率下的MAE、RMSE、R²
- 性能下降分析
- 最敏感的遮挡率区间

### 3.2 查看策略对比

```bash
# 查看所有策略的对比报告
cat masking_evaluation_mbj_bandgap/strategy_comparison_report.txt
```

**报告内容**：
- 鲁棒性排名
- 各策略的MAE增长率
- 关键发现和建议

### 3.3 可视化图表

```bash
# 列出所有生成的图表
find masking_evaluation_mbj_bandgap -name "*.png"

# 下载到本地查看
scp user@server:~/masking_evaluation_mbj_bandgap/*/*.png ./local_dir/
scp user@server:~/masking_evaluation_mbj_bandgap/*.png ./local_dir/
```

**图表类型**：
1. **单个策略图** (每个策略4张子图):
   - MAE vs 遮挡率
   - RMSE vs 遮挡率
   - R² vs 遮挡率
   - 归一化对比

2. **策略对比图** (4张子图):
   - MAE对比
   - RMSE对比
   - R²对比
   - 相对性能下降

### 3.4 数值数据

```bash
# CSV格式汇总表
cat masking_evaluation_mbj_bandgap/strategy_comparison_summary.csv

# JSON格式原始数据
cat masking_evaluation_mbj_bandgap/random_token/text_masking_results_random_token.json
```

## 完整命令参考

### 快速开始（推荐流程）

```bash
# 1. 数据预处理（只需一次）
./preprocess_mbj_bandgap.sh

# 2. 快速测试
./quick_test_masking.sh

# 3. 完整评估
./run_masking_eval_mbj_bandgap.sh

# 4. 查看结果
cat masking_evaluation_mbj_bandgap/strategy_comparison_report.txt
```

### 单个策略评估

如果只想测试某个特定策略：

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --output_dir ./eval_token_only
```

### 自定义遮挡率

只测试特定遮挡率：

```bash
python evaluate_text_masking.py \
    --checkpoint <checkpoint> \
    --preprocessed_dir <preprocessed_dir> \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.25 0.5 0.75 1.0  # 只测试这5个
```

### 生成策略对比

如果已经运行了所有策略，可以重新生成对比报告：

```bash
python compare_masking_strategies.py \
    --input_dir ./masking_evaluation_mbj_bandgap
```

## 故障排除

### 问题1: 预处理失败

**症状**: `preprocess_mbj_bandgap.sh`报错

**解决方案**:
1. 查看详细错误信息
2. 参考[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)
3. 检查数据集路径和文件权限

### 问题2: 快速测试失败

**症状**: `quick_test_masking.sh`报错

**常见原因**:
- ❌ 预处理数据不存在 → 先运行预处理
- ❌ Checkpoint路径错误 → 检查路径
- ❌ 依赖包缺失 → 安装所需包

**调试步骤**:
```bash
# 1. 检查预处理数据
ls -la /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/

# 2. 检查checkpoint
ls -la /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt

# 3. 测试Python导入
python -c "import torch, dgl, jarvis, transformers; print('All imports OK')"
```

### 问题3: 评估过程中内存不足

**症状**: `MemoryError` 或 `CUDA out of memory`

**解决方案**:
```bash
# 减小batch size
python evaluate_text_masking.py \
    ... \
    --batch_size 32  # 或更小
```

### 问题4: 结果文件缺失

**症状**: 找不到某些结果文件

**检查**:
```bash
# 查看评估是否完全成功
ls -la masking_evaluation_mbj_bandgap/

# 应该有5个策略目录
ls -la masking_evaluation_mbj_bandgap/*/
```

**重新运行失败的策略**:
```bash
python evaluate_text_masking.py \
    --checkpoint <checkpoint> \
    --preprocessed_dir <preprocessed_dir> \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy <failed_strategy> \
    --output_dir ./masking_evaluation_mbj_bandgap/<failed_strategy>
```

## 预期结果分析

### 性能指标解读

**MAE增长率**（从0%到100%遮挡）：
- < 50%: 优秀鲁棒性 ⭐⭐⭐
- 50-100%: 良好鲁棒性 ⭐⭐
- 100-200%: 一般鲁棒性 ⭐
- \> 200%: 较弱鲁棒性

**R²保持率**（100%遮挡 vs 0%遮挡）：
- \> 80%: 优秀
- 60-80%: 良好
- 40-60%: 一般
- < 40%: 较弱

### 关键发现

通过评估，你将了解：

1. **模型对文本的依赖程度**
   - 完全遮挡文本后，性能下降多少？
   - 结构信息能否补偿文本信息的缺失？

2. **最关键的文本信息类型**
   - 哪种遮挡策略影响最大？
   - Token级、单词级、还是句子级信息更重要？

3. **关键词的作用**
   - `keep_keywords`策略的表现如何？
   - 化学元素和结构术语是否足够支撑预测？

4. **Middle + Fine-grained Fusion的效果**
   - 融合机制是否有效整合了文本和结构信息？
   - 与简单融合相比，鲁棒性如何？

### 对比不同模型

如果你有多个模型（如Solution 1和Solution 2），可以对比它们的鲁棒性：

```bash
# 评估Solution 1
python evaluate_text_masking.py \
    --checkpoint ./results/solution1/best_model.pt \
    ... \
    --output_dir ./masking_eval_solution1

# 评估Solution 2
python evaluate_text_masking.py \
    --checkpoint ./results/solution2/best_model.pt \
    ... \
    --output_dir ./masking_eval_solution2

# 对比结果
diff masking_eval_solution1/strategy_comparison_report.txt \
     masking_eval_solution2/strategy_comparison_report.txt
```

## 时间估算

完整工作流程预计时间：

| 步骤 | 预计时间 | 备注 |
|------|---------|------|
| 数据预处理 | 10-30分钟 | 只需运行一次 |
| 快速测试 | 5-10分钟 | 验证配置 |
| 完整评估 | 30-60分钟 | 5种策略 × 11个遮挡率 |
| 结果分析 | 10-20分钟 | 查看报告和图表 |
| **总计** | **1-2小时** | 首次运行 |

**后续评估**（已有预处理数据）：
- 快速测试: 5-10分钟
- 完整评估: 30-60分钟

## 检查清单

### 开始之前 ✅

- [ ] 所有脚本已复制到服务器
- [ ] 数据集存在且可访问
- [ ] Checkpoint文件存在
- [ ] 依赖包已安装
- [ ] 有足够的磁盘空间（>5GB）
- [ ] 有足够的GPU内存（>8GB）

### 预处理完成后 ✅

- [ ] 生成了train.pkl、val.pkl、test.pkl
- [ ] 文件大小合理（不是0字节）
- [ ] README.txt包含正确的样本数量
- [ ] 可以用Python成功加载pkl文件

### 评估完成后 ✅

- [ ] 所有5个策略目录都存在
- [ ] 每个策略有3个文件（json、txt、png）
- [ ] 生成了策略对比文件
- [ ] 图表可以正常打开查看

## 相关文档

- 📘 **PREPROCESSING_GUIDE.md** - 数据预处理详细指南
- 📗 **TEXT_MASKING_EVALUATION_GUIDE.md** - 文本遮挡评估完整说明
- 📙 **README_MBJ_BANDGAP_MASKING_EVAL.md** - MBJ Band Gap专用指南
- 📕 **TRAINING_GUIDE.md** - 模型训练指南
- 📔 **FUSION_SOLUTIONS_GUIDE.md** - 融合策略选择指南

## 获取帮助

如遇到问题：

1. **查看日志**: 脚本会输出详细的错误信息和进度
2. **参考文档**: 查看相关指南中的故障排除部分
3. **验证步骤**: 按照检查清单逐项验证
4. **简化测试**: 从快速测试开始，逐步扩展

## 总结

完整的工作流程：

```bash
# === 第一次运行 ===
# 1. 预处理数据（只需一次）
./preprocess_mbj_bandgap.sh

# 2. 快速验证
./quick_test_masking.sh

# 3. 完整评估
./run_masking_eval_mbj_bandgap.sh

# 4. 查看结果
cat masking_evaluation_mbj_bandgap/strategy_comparison_report.txt

# === 后续运行 ===
# 可以直接从步骤2开始（跳过预处理）
```

祝评估顺利！🎉

如有任何问题，请参考相关文档或检查错误日志。
