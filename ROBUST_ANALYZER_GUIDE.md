# Robust Attention Analyzer 使用指南

## 📖 概述

全新的健壮注意力分析系统，专门设计用于处理各种边界情况，包括：
- ✅ 所有原子注意力相同的情况
- ✅ 多头注意力退化
- ✅ 注意力分布过于集中
- ✅ 代码版本不匹配
- ✅ 自动诊断和降级策略

## 🚀 快速开始

### 基础用法

```bash
python demo_robust_attention.py \
    --model_path /path/to/checkpoint.pt \
    --cif_path /path/to/structure.cif \
    --text "Material description text..." \
    --save_dir ./results
```

### 示例

```bash
python demo_robust_attention.py \
    --model_path /public/home/ghzhang/models/best_model.pt \
    --cif_path /public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap/cif/10.cif \
    --text "LiBa4Hf crystallizes cubic F-43m space group. structure consists clusters Ba4Hf framework." \
    --save_dir ./analysis_robust
```

## 🔍 功能特性

### 1. 自动质量诊断

系统会自动诊断注意力权重质量：

```
🔬 注意力权重质量诊断
================================================================================

1️⃣ 基本信息:
   - Attention heads: 8
   - Atoms: 6
   - Sequence length: 79

2️⃣ 多头注意力分析:
   - 平均头间相关性: 0.9998
   - 头多样性分数: 0.0002

3️⃣ 原子特异性分析:
   - 平均原子间相关性: 1.0000
   - 原子多样性分数: 0.0000

4️⃣ 注意力分布分析:
   - 平均熵: 2.8456
   - 最大可能熵: 4.3694

5️⃣ 诊断结论:
   - 质量评估: POOR
   - 发现问题:
      • 所有attention heads几乎相同（多头退化）
      • 所有原子的注意力模式几乎相同
   - 建议:
      • 建议使用全局分析而非逐原子分析
      • 检查GNN层输出的节点特征是否过于相似
      • 考虑减少GNN层数或添加残差连接
```

### 2. 自适应分析策略

#### 情况 A: 原子注意力正常（不同）

系统使用**逐原子分析**：

```
✅ 原子注意力模式正常，使用标准分析...

⚛️  逐原子注意力分析
================================================================================

Ba_0:
  - ba(1)              0.125678
  - barium             0.089234
  - framework          0.076543
  - cluster            0.065432
  - cubic              0.054321

Ba_1:
  - coordinate         0.134567
  - 12-coordinate      0.098765
  - framework          0.087654
  ...
```

生成可视化：`per_atom_attention.png`

#### 情况 B: 原子注意力相同

系统自动切换到**全局分析**：

```
⚠️  检测到原子注意力模式相同，使用全局分析策略...

📊 全局注意力模式分析
================================================================================

🔤 全局最重要的 15 个 Tokens:
Rank   Token                Importance   Category
------------------------------------------------------------
1      liba4hf              0.093750     Element
2      q6                   0.062500     Other
3      12-coordinate        0.041667     Crystallography
4      f-43m                0.031250     Crystallography
5      ba(1)                0.031250     Element
...
```

生成可视化：`global_attention_analysis.png`（包含4个子图）

### 3. 详细统计分析

无论使用哪种策略，都会提供统计信息：

```
📈 统计信息:
   - 注意力头数: 8
   - 原子数: 6
   - 序列长度: 79
   - 平均注意力: 0.012658
   - 注意力标准差: 0.023456
   - 稀疏度: 45.67%
```

## 📊 输出文件

### 全局分析模式

`global_attention_analysis.png` 包含：
1. **Top 10 Tokens柱状图** - 最重要的tokens及其权重
2. **Token类别分布饼图** - Element/Crystallography/Chemistry/Other
3. **最活跃Head的热图** - 显示该head的atom×token注意力模式
4. **注意力权重分布直方图** - 权重的统计分布

### 逐原子分析模式

`per_atom_attention.png` 包含：
- **热图矩阵** - 每个原子的top-10 attended tokens

## 🔧 作为Python模块使用

### 方法 1: 使用便捷函数

```python
from robust_attention_analyzer import run_complete_analysis

results = run_complete_analysis(
    model=model,
    g=graph,
    lg=line_graph,
    text=description,
    atoms_object=atoms,
    save_dir='./output'
)

# 访问结果
diagnosis = results['diagnosis']
statistics = results['statistics']

if diagnosis['use_alternative_analysis']:
    global_analysis = results['global_analysis']
    print(global_analysis['top_tokens'])
else:
    per_atom = results['per_atom_analysis']
    for atom_id, info in per_atom['atoms'].items():
        print(f"{atom_id}: {info['top_tokens']}")
```

### 方法 2: 直接使用分析器类

```python
from robust_attention_analyzer import RobustAttentionAnalyzer

analyzer = RobustAttentionAnalyzer(model, device='cuda')

# 1. 诊断质量
diagnosis = analyzer.diagnose_attention_quality(
    attention_weights,
    elements,
    verbose=True
)

# 2. 自适应分析
results = analyzer.analyze_with_fallback(
    attention_weights,
    atoms_object,
    text_tokens,
    save_dir='./output',
    top_k=15
)
```

## 💡 使用建议

### 对于训练新模型

1. **定期运行诊断**：
   ```bash
   # 每个epoch结束后
   python demo_robust_attention.py --model_path epoch_10.pt ...
   ```

2. **监控指标**：
   - `atom_diversity` > 0.1（原子注意力有差异）
   - `head_diversity` > 0.1（多头注意力有差异）
   - `entropy` > 2.0（注意力分布不太集中）

3. **根据诊断调整**：
   - 如果 `atom_diversity` 太低 → 检查GNN over-smoothing
   - 如果 `head_diversity` 太低 → 添加head diversity loss
   - 如果 `entropy` 太低 → 检查temperature scaling

### 对于分析现有模型

1. **首先运行诊断**：
   ```bash
   python demo_robust_attention.py ... --save_dir ./diagnosis
   ```

2. **查看质量评估**：
   - `GOOD`: 可以信任逐原子分析
   - `ACCEPTABLE`: 谨慎解读
   - `POOR`: 使用全局分析，问题可能在模型训练

3. **根据建议改进**：
   - 按照诊断输出的建议进行模型调整

## 🆚 与原系统的区别

| 特性 | 原系统 | Robust Analyzer |
|------|--------|----------------|
| 处理相同原子注意力 | ❌ 显示错误结果 | ✅ 自动切换到全局分析 |
| 质量诊断 | ❌ 无 | ✅ 5个维度的诊断 |
| 降级策略 | ❌ 无 | ✅ 自适应选择分析方法 |
| Token分类 | ⚠️  简单过滤 | ✅ 智能分类（Element/Crystallography/Chemistry/Other） |
| 可视化 | ⚠️  单一热图 | ✅ 多子图综合分析 |
| 统计分析 | ❌ 无 | ✅ 完整统计信息 |
| 错误处理 | ⚠️  可能崩溃 | ✅ 健壮的异常处理 |

## 🐛 故障排除

### 问题 1: 模块导入失败

```bash
ModuleNotFoundError: No module named 'robust_attention_analyzer'
```

**解决**：
```bash
# 确保在正确目录
cd /home/user/11.23

# 或者添加到 Python 路径
export PYTHONPATH=/home/user/11.23:$PYTHONPATH
```

### 问题 2: 所有分析都显示相同

**可能原因**：
1. 模型确实输出相同的注意力（见诊断输出）
2. 代码版本不匹配（检查 Missing keys/Unexpected keys）

**解决**：
```bash
# 查看完整的模型加载日志
python demo_robust_attention.py ... 2>&1 | grep -A5 "Missing keys\|Unexpected keys"
```

### 问题 3: 可视化图片质量差

**解决**：
修改 dpi 参数（默认300）：

```python
# 在 robust_attention_analyzer.py 中搜索：
plt.savefig(viz_path, dpi=300, bbox_inches='tight')

# 改为：
plt.savefig(viz_path, dpi=600, bbox_inches='tight')  # 更高分辨率
```

## 📚 进阶用法

### 自定义停用词

```python
analyzer = RobustAttentionAnalyzer(model, device='cuda')

# 添加自定义停用词
analyzer.stopwords.update({'custom', 'stopword', 'list'})

# 或完全替换
analyzer.stopwords = {'only', 'these', 'words'}
```

### 自定义Token分类

修改 `_categorize_token` 方法：

```python
def _categorize_token(self, token: str) -> str:
    # 添加自定义类别
    if 'band' in token.lower() or 'gap' in token.lower():
        return 'Electronic Property'

    # 调用原始分类
    return super()._categorize_token(token)
```

### 批量分析

```python
import glob

for cif_file in glob.glob('/path/to/cif/*.cif'):
    cif_name = Path(cif_file).stem
    results = run_complete_analysis(
        model, g, lg, text, atoms,
        save_dir=f'./batch_analysis/{cif_name}'
    )
```

## 📞 支持

如有问题，请查看：
1. `DIAGNOSTIC_GUIDE.md` - 原注意力诊断指南
2. `ROOT_CAUSE_ANALYSIS.md` - 根本原因分析
3. GitHub Issues

---

**最后更新**: 2025-11-23
**版本**: 1.0.0
**兼容性**: PyTorch 1.x+, DGL 0.9+, JARVIS-Tools
