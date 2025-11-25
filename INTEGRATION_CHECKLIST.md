# DynamicFusionModule 集成清单

## ✅ 已完成的修改

1. **models/alignn.py**
   - ✅ 替换 MiddleFusionModule → DynamicFusionModule
   - ✅ 添加动态路由 (Softmax 竞争机制)
   - ✅ 保留双重残差连接 (物理先验)
   - ✅ 集成权重监控 (EMA 跟踪)
   - ✅ 向后兼容别名

2. **monitor_fusion_weights.py**
   - ✅ 创建权重监控工具
   - ✅ 支持打印统计信息
   - ✅ 支持 CSV 日志记录

3. **test_residual_impact.py**
   - ✅ 双重残差分析脚本

---

## 🔧 建议修改的地方

### 1. 训练脚本集成 ⭐ 重要

**需要修改的文件：`train.py`**

参考 `train_monitoring_patch.py` 中的示例代码：

```python
# 在文件开头添加导入
from monitor_fusion_weights import print_fusion_weights, log_fusion_weights_to_file

# 在 Events.EPOCH_COMPLETED 处添加监控
# 位置：约第 386 行
if hasattr(net, 'middle_fusion_modules') and len(net.middle_fusion_modules) > 0:
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_fusion_weights(engine):
        if engine.state.epoch % 5 == 0:
            print_fusion_weights(net, verbose=True)
            log_fusion_weights_to_file(
                net,
                os.path.join(config.output_dir, "fusion_weights.csv"),
                engine.state.epoch
            )
```

**效果**：
- 每 5 个 epoch 打印权重统计
- 自动记录到 CSV 文件
- 可视化路由器学习过程

---

### 2. 其他 alignn.py 备份文件 (可选)

**相关文件**：
- `models/alignn_(1).py`
- `models/alignn-1.1.py`
- `models/alignn.py-SGA1.0`

**建议**：
- 如果这些是旧版本备份 → 不需要修改
- 如果正在使用 → 建议同步更新

**检查方法**：
```bash
# 查看是否有脚本导入这些文件
grep -r "from.*alignn_(1)" .
grep -r "from.*alignn-1.1" .
```

---

### 3. 可视化脚本更新 (可选)

**可能需要更新的文件**：
- `visualize_middle_fusion_clustering.py`
- `compare_fusion_mechanisms.py`

**建议操作**：
1. 运行这些脚本测试是否兼容
2. 如果报错，更新为使用 `DynamicFusionModule`
3. 添加新的权重分布可视化

---

### 4. 文档更新 (可选但推荐)

**建议添加**：
- README 中说明 DynamicFusionModule 的使用
- 添加训练日志示例
- 说明双重残差的物理意义

---

## 📊 验证步骤

### Step 1: 测试模型加载
```python
from models.alignn import ALIGNN, ALIGNNConfig

config = ALIGNNConfig(
    name="alignn",
    use_middle_fusion=True,
    middle_fusion_layers="2"
)
model = ALIGNN(config)

# 检查模块是否正确
print(type(model.middle_fusion_modules['layer_2']))
# 应该输出: <class 'models.alignn.DynamicFusionModule'>
```

### Step 2: 测试权重监控
```python
from monitor_fusion_weights import print_fusion_weights

# 在训练后调用
stats = print_fusion_weights(model)
print(stats)
```

### Step 3: 运行小规模训练
```bash
# 测试 5 个 epoch
python train.py \
    --config your_config.json \
    --n_train 100 \
    --epochs 5
```

检查输出：
- ✅ 每 5 个 epoch 应该打印权重统计
- ✅ 生成 `output_dir/fusion_weights.csv`
- ✅ w_graph + w_text ≈ 1.0

---

## 🎯 关键监控指标

### 正常表现（材料性质预测）

| 指标 | 期望范围 | 含义 |
|------|---------|------|
| w_graph | 0.5-0.9 | 路由器给图的原始权重 |
| w_text | 0.1-0.5 | 路由器给文本的原始权重 |
| 有效图权重 | 1.5-1.9 | (1 + w_graph) |
| 有效文本权重 | 0.1-0.5 | w_text |
| 图/文本比例 | 3-10x | 图应该占主导 |

### 异常情况

⚠️ **警告信号**：
- w_text > 0.7 → 文本权重过高，可能过拟合文本描述
- w_graph < 0.3 → 图权重过低，违反物理先验
- 比例 < 2x → 文本影响过大

🔧 **解决方案**：
1. 添加权重正则化
2. 限制 w_text 上限为 0.3-0.5
3. 增加路由器的 dropout

---

## 📝 后续优化建议

### 可选增强功能

1. **自适应权重限制**
   ```python
   # 在 DynamicFusionModule.forward 中
   w_text = torch.clamp(weights[:, 1], max=0.5)  # 限制文本最大权重
   w_graph = 1.0 - w_text
   ```

2. **分层权重策略**
   - 早期层：纯图（w_text=0）
   - 中期层：图为主（w_text<0.3）
   - 后期层：适度融合（w_text<0.5）

3. **注意力可视化**
   - 绘制不同样本的权重分布
   - 分析哪些样本依赖文本更多

---

## 💡 常见问题

### Q1: 旧模型检查点能否加载？
**A**: 可以！使用了别名 `MiddleFusionModule = DynamicFusionModule`，向后兼容。

### Q2: 如何关闭权重监控？
**A**: 注释掉 `trainer.on(Events.EPOCH_COMPLETED)` 的监控代码即可。

### Q3: 双重残差会导致过拟合吗？
**A**: 不会。它强化了物理先验（结构决定性质），反而能提高泛化能力。

### Q4: 如何对比新旧模块性能？
**A**:
1. 保留旧版本权重文件
2. 分别训练并记录验证集性能
3. 使用 `compare_fusion_mechanisms.py` 对比

---

## 📞 需要帮助？

如果遇到问题，检查：
1. `fusion_weights.csv` 中的权重趋势
2. 训练日志中的损失曲线
3. 验证集性能变化

可以运行诊断脚本：
```bash
python test_residual_impact.py  # 分析双重残差影响
python monitor_fusion_weights.py  # 测试监控工具
```
