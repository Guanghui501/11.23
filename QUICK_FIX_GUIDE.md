# ⚡ 快速修复指南 - 常见错误

## 🔴 错误1: Checkpoint配置问题

```
ValueError: Checkpoint中没有找到model_config
✗ random_token 评估失败
```

**状态**: ✅ 已修复（通过配置文件或自动推断）

---

## 🔴 错误2: 模型加载键不匹配

```
RuntimeError: Error(s) in loading state_dict for ALIGNN:
	Unexpected key(s) in state_dict: "fine_grained_attention.atom_position_embedding.weight".
```

**状态**: ✅ 已修复（自动宽松加载）

---

## 🔴 错误3: 数据加载类型错误

```
TypeError: list indices must be integers or slices, not str
```

**状态**: ✅ 已修复（自定义数据加载器）

---

## ✅ 快速解决方案（3分钟）

### 方案1: 一键修复（最简单）

```bash
chmod +x fix_checkpoint_and_run.sh
./fix_checkpoint_and_run.sh
```

**这个脚本会自动**：
1. ✅ 检查你的checkpoint
2. ✅ 创建配置文件
3. ✅ 运行快速测试
4. ✅ 询问是否完整评估

### 方案2: 手动两步（如果你想理解过程）

#### 步骤1: 创建配置文件

```bash
python create_config_from_checkpoint.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --output mbj_bandgap_config.json
```

#### 步骤2: 使用配置文件运行评估

```bash
python evaluate_text_masking.py \
    --checkpoint /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/SGA-V2.0/mbj/middle+fine/output_100epochs_42_bs64_sw_ju_middle_fg_proj_mbj_bandgap_quantext/mbj_bandgap/best_test_model.pt \
    --config_file mbj_bandgap_config.json \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --masking_strategy random_token \
    --output_dir ./test_output
```

## 📋 接下来做什么？

### 如果快速测试成功 ✅

运行完整评估：

```bash
# 方式A: 修改并运行批量脚本
# 编辑 run_masking_eval_mbj_bandgap.sh
# 在每个 python evaluate_text_masking.py 命令中添加：
# --config_file mbj_bandgap_config.json \

vim run_masking_eval_mbj_bandgap.sh
# 然后运行
./run_masking_eval_mbj_bandgap.sh

# 方式B: 使用一体化脚本（已包含配置文件处理）
./fix_checkpoint_and_run.sh
```

### 如果仍然失败 ❌

1. **检查预处理数据**：
   ```bash
   ls -la /public/home/ghzhang/preprocessed_data/jarvis/mbj_bandgap/
   # 应该看到: train.pkl, val.pkl, test.pkl
   ```

2. **检查配置文件**：
   ```bash
   cat mbj_bandgap_config.json
   # 查看参数是否合理
   ```

3. **查看详细诊断**：
   ```bash
   python inspect_checkpoint.py \
       /public/home/ghzhang/.../best_test_model.pt
   ```

4. **查看完整指南**：
   ```bash
   cat CHECKPOINT_CONFIG_TROUBLESHOOTING.md
   ```

## 🎯 为什么会出现这个问题？

你的checkpoint是在训练时创建的，但训练脚本**没有保存模型配置信息**（`model_config`）。

这是一个常见问题，特别是：
- 使用旧版本训练脚本
- 或训练脚本只保存了模型权重

## 🛠️ 已创建的工具

1. **inspect_checkpoint.py** - 检查checkpoint内容
2. **create_config_from_checkpoint.py** - 从checkpoint推断配置
3. **fix_checkpoint_and_run.sh** - 一体化解决方案
4. **CHECKPOINT_CONFIG_TROUBLESHOOTING.md** - 详细故障排除指南

## 📚 相关文档

- **CHECKPOINT_CONFIG_TROUBLESHOOTING.md** - 完整故障排除（强烈推荐阅读）
- **README_COMPLETE_WORKFLOW.md** - 完整评估工作流程
- **TEXT_MASKING_EVALUATION_GUIDE.md** - 评估详细指南

## ⏱️ 预计时间

- **方案1（一键修复）**: 约5分钟
- **方案2（手动）**: 约3分钟

## 💡 提示

- 配置文件只需创建**一次**
- 创建后可以重复使用
- 建议保存配置文件供将来使用

## ✨ 快速命令总结

```bash
# === 最快的解决方案 ===
./fix_checkpoint_and_run.sh

# === 或者手动两步 ===
# 1. 创建配置
python create_config_from_checkpoint.py \
    --checkpoint <your_checkpoint> \
    --output config.json

# 2. 运行评估
python evaluate_text_masking.py \
    --checkpoint <your_checkpoint> \
    --config_file config.json \
    --preprocessed_dir <preprocessed_dir> \
    --dataset jarvis \
    --property mbj_bandgap

# === 诊断工具 ===
# 检查checkpoint
python inspect_checkpoint.py <your_checkpoint>
```

祝你顺利解决问题！🎉
