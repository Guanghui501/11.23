# Checkpoint配置问题故障排除指南

## 问题描述

当运行文本遮挡评估时，出现以下错误：

```
ValueError: Checkpoint中没有找到model_config
```

## 原因分析

这个问题通常发生在：
1. **训练时没有保存model_config** - 旧版本的训练脚本可能只保存了模型权重
2. **Checkpoint格式不标准** - 使用了不同的字段名称（如`config`而非`model_config`）
3. **直接保存了模型** - 使用`torch.save(model, ...)`而非`torch.save({'model': model.state_dict(), ...}, ...)`

## 🛠️ 解决方案

### 方案1: 使用一体化修复脚本（推荐）

**最简单的方法** - 一个脚本自动完成所有步骤：

```bash
chmod +x fix_checkpoint_and_run.sh
./fix_checkpoint_and_run.sh
```

这个脚本会：
1. ✅ 检查checkpoint内容
2. ✅ 创建配置文件（如需要）
3. ✅ 运行快速测试验证
4. ✅ 询问是否运行完整评估

**适用场景**: 第一次运行评估，不确定checkpoint状态

---

### 方案2: 手动分步骤修复

如果你想了解每个步骤的细节：

#### 步骤1: 检查Checkpoint内容

```bash
python inspect_checkpoint.py \
    /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt
```

**输出示例**:
```
================================================================================
检查 Checkpoint: /path/to/best_test_model.pt
================================================================================

✓ Checkpoint加载成功

Checkpoint类型: dict
包含的键 (5 个):
--------------------------------------------------------------------------------
  • model                        : dict
      → 包含 1234 个参数
         - atom_embedding.weight
         - edge_embedding.weight
         - fc.weight
         ... (还有 1231 个)
  • optimizer                    : dict
  • epoch                        : int
      → 100
  • best_test_loss               : float
      → 0.1234

================================================================================
关键字段检查:
--------------------------------------------------------------------------------
✓ 找到模型权重: 'model'
✗ 未找到模型配置 (尝试过: ['model_config', 'config', 'args', 'model_args'])

⚠ 警告: 缺少模型配置！
   这可能导致评估脚本失败。
```

#### 步骤2: 创建配置文件

```bash
python create_config_from_checkpoint.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --output mbj_bandgap_model_config.json
```

**输出示例**:
```
================================================================================
从Checkpoint推断模型配置
================================================================================

Checkpoint: /path/to/best_test_model.pt

⚠ Checkpoint缺少model_config，尝试从模型结构推断...

推断的参数:
  alignn_layers: 4
  gcn_layers: 4
  hidden_features: 64
  use_cross_modal_attention: True
  cross_modal_hidden_dim: 64
  cross_modal_num_heads: 4
  use_middle_fusion: True
  middle_fusion_layers: [0, 1, 2]
  use_fine_grained_attention: True
  fusion_strategy: gated
  gated_fusion_type: dual_gate

⚠ 注意: 某些参数可能不准确，请检查并根据实际情况调整

================================================================================
模型配置:
================================================================================
  alignn_layers                       : 4
  gcn_layers                          : 4
  atom_input_features                 : 92
  edge_input_features                 : 80
  ...

✓ 配置已保存到: mbj_bandgap_model_config.json
```

#### 步骤3: 检查并编辑配置文件（可选）

```bash
vim mbj_bandgap_model_config.json
```

**重点检查**：
- `alignn_layers` 和 `gcn_layers` 是否正确
- `use_cross_modal_attention`, `use_middle_fusion`, `use_fine_grained_attention` 是否与你的训练设置一致
- `cross_modal_num_heads`, `middle_fusion_num_heads` 等参数

如果不确定，可以查看训练日志或训练脚本。

#### 步骤4: 使用配置文件运行评估

```bash
# 快速测试
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --config_file mbj_bandgap_model_config.json \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output
```

如果成功，继续完整评估：

```bash
# 修改run_masking_eval_mbj_bandgap.sh，添加--config_file参数
vim run_masking_eval_mbj_bandgap.sh

# 在python evaluate_text_masking.py命令中添加：
# --config_file mbj_bandgap_model_config.json \

# 然后运行
./run_masking_eval_mbj_bandgap.sh
```

---

### 方案3: 直接修改评估脚本（不推荐）

如果你确定模型配置，可以直接在代码中硬编码：

```python
# 在evaluate_text_masking.py中，找到加载配置的部分
# 添加一个默认配置

from models.alignn import ALIGNNConfig

# 如果checkpoint没有配置，使用默认配置
if model_config is None:
    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        edge_input_features=80,
        embedding_features=64,
        hidden_features=64,
        output_features=1,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=64,
        cross_modal_num_heads=4,
        use_middle_fusion=True,
        middle_fusion_layers=[0, 1, 2],
        use_fine_grained_attention=True,
        fusion_strategy='gated',
        gated_fusion_type='dual_gate',
        # ... 其他参数
    )
```

**注意**: 这种方法不推荐，因为参数可能不准确。

---

## 🔍 诊断工具

### 1. inspect_checkpoint.py

**用途**: 检查checkpoint包含哪些字段

**使用方法**:
```bash
python inspect_checkpoint.py /path/to/checkpoint.pt
```

**输出**:
- Checkpoint的所有键
- 模型权重信息
- 配置信息（如果有）
- 训练信息（epoch, loss等）

### 2. create_config_from_checkpoint.py

**用途**: 从checkpoint的模型结构推断配置

**使用方法**:
```bash
python create_config_from_checkpoint.py \
    --checkpoint /path/to/checkpoint.pt \
    --output config.json
```

**工作原理**:
- 分析模型的state_dict
- 查找特定的层名称和参数
- 推断模型架构参数
- 生成JSON配置文件

**限制**:
- 某些参数无法从权重推断（如num_heads）
- 需要手动验证推断的参数

### 3. fix_checkpoint_and_run.sh

**用途**: 一体化解决方案

**使用方法**:
```bash
./fix_checkpoint_and_run.sh
```

**功能**:
1. 检查checkpoint
2. 创建配置（如需要）
3. 快速测试
4. 完整评估（可选）

---

## ❓ 常见问题

### Q1: create_config_from_checkpoint.py 推断的参数准确吗？

**A**: 大部分参数是准确的，但以下参数可能需要手动检查：
- `cross_modal_num_heads` - 默认值为4
- `middle_fusion_num_heads` - 默认值为4
- `fine_grained_num_heads` - 默认值为4
- `middle_fusion_layers` - 推断的范围可能不准确

**建议**: 对比训练日志或训练脚本中的参数设置。

### Q2: 如果配置文件参数错误会怎样？

**A**:
- **模型结构参数错误** (如layers数量): 加载模型权重时会失败
- **其他参数错误** (如num_heads): 可能加载成功但评估结果不准确

**检测方法**:
```bash
# 尝试加载模型，看是否报错
python -c "
import torch
from models.alignn import ALIGNN, ALIGNNConfig
import json

with open('mbj_bandgap_model_config.json', 'r') as f:
    config_dict = json.load(f)

config = ALIGNNConfig(**config_dict)
model = ALIGNN(config)

checkpoint = torch.load('best_test_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'])

print('✓ 模型加载成功！配置正确。')
"
```

### Q3: 能否从训练脚本恢复配置？

**A**: 可以！如果你还有训练时使用的命令或脚本：

```bash
# 查看训练命令历史
history | grep train

# 或查看训练脚本
cat train_script.sh

# 或查看训练日志开头（通常会打印配置）
head -100 training.log
```

从这些信息中提取参数，手动创建配置文件。

### Q4: 为什么我的checkpoint没有保存model_config？

**A**: 可能原因：
1. 使用了旧版本的训练脚本
2. 训练脚本中没有包含config保存逻辑

**预防措施** (未来训练时):
```python
# 在训练脚本中，保存checkpoint时包含config
torch.save({
    'model': model.state_dict(),
    'model_config': model_config,  # ← 添加这行
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_loss': best_val_loss,
}, checkpoint_path)
```

---

## 📝 完整示例：处理MBJ Band Gap checkpoint

### 场景
你的checkpoint位于：
```
/public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt
```

### 解决步骤

```bash
# 1. 检查checkpoint
python inspect_checkpoint.py \
    /public/home/ghzhang/.../best_test_model.pt

# 2. 创建配置文件
python create_config_from_checkpoint.py \
    --checkpoint /public/home/ghzhang/.../best_test_model.pt \
    --output mbj_bandgap_config.json

# 3. （可选）根据训练日志检查配置
# 编辑 mbj_bandgap_config.json

# 4. 快速测试
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/.../best_test_model.pt \
    --config_file mbj_bandgap_config.json \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --masking_ratios 0.0 0.5 1.0 \
    --output_dir ./test_output

# 5. 如果成功，运行完整评估
# 方式A: 使用一体化脚本
./fix_checkpoint_and_run.sh

# 方式B: 手动运行每个策略
for strategy in random_token random_word random_chunk sentence keep_keywords; do
    python evaluate_text_masking.py \
        --checkpoint /public/home/ghzhang/.../best_test_model.pt \
        --config_file mbj_bandgap_config.json \
        --preprocessed_dir /public/home/ghzhang/preprocessed_data \
        --dataset jarvis \
        --property mbj_bandgap \
        --masking_strategy $strategy \
        --output_dir ./masking_eval/$strategy
done

# 6. 生成对比报告
python compare_masking_strategies.py --input_dir ./masking_eval
```

---

## 🎯 最佳实践

1. **首次评估**:
   - 使用 `fix_checkpoint_and_run.sh` 一体化脚本
   - 这样最省心，自动处理所有问题

2. **后续评估**:
   - 配置文件已创建，直接使用
   - 保存配置文件以备后用

3. **多个checkpoint**:
   - 为每个checkpoint创建单独的配置文件
   - 命名规范：`<model_name>_config.json`

4. **验证配置**:
   - 总是先运行快速测试
   - 确认配置正确再运行完整评估

5. **文档记录**:
   - 记录模型的训练参数
   - 保存配置文件到版本控制

---

## 🆘 仍然无法解决？

如果按照上述步骤仍然失败：

1. **检查错误日志**:
   - 仔细阅读完整的错误堆栈
   - 查找关键错误信息

2. **验证文件**:
   - Checkpoint文件是否损坏？
   - 配置文件JSON格式是否正确？

3. **测试简化版本**:
   ```python
   # 最小化测试脚本
   import torch
   checkpoint = torch.load('checkpoint.pt')
   print(checkpoint.keys())
   ```

4. **提供详细信息**:
   - Checkpoint的完整输出 (`inspect_checkpoint.py`)
   - 训练时使用的命令或脚本
   - 完整的错误日志

---

## 📚 相关文档

- **README_COMPLETE_WORKFLOW.md** - 完整评估工作流程
- **TEXT_MASKING_EVALUATION_GUIDE.md** - 评估详细指南
- **README_MBJ_BANDGAP_MASKING_EVAL.md** - MBJ Band Gap专用指南

祝你顺利解决问题！🎉
