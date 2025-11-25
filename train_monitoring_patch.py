"""
补丁：在 train.py 中添加 DynamicFusionModule 权重监控

使用方法：
1. 在 train.py 开头导入：
   from monitor_fusion_weights import print_fusion_weights, log_fusion_weights_to_file

2. 在 Events.EPOCH_COMPLETED 事件中添加监控：
   见下面的示例代码
"""

# ============ 添加到 train.py 文件开头的导入部分 ============
from monitor_fusion_weights import print_fusion_weights, log_fusion_weights_to_file


# ============ 添加到 train_dgl 函数中 (约第 386 行后) ============
def add_fusion_weight_monitoring(trainer, net, config):
    """Add DynamicFusionModule weight monitoring to training loop.

    Args:
        trainer: Ignite trainer engine
        net: The ALIGNN model
        config: Training configuration
    """

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_fusion_weights(engine):
        """Log fusion module weights after each epoch."""
        epoch = engine.state.epoch

        # Only log every N epochs to reduce overhead
        if epoch % 5 == 0:  # Log every 5 epochs
            print("\n" + "="*80)
            print(f"DynamicFusionModule Weight Statistics (Epoch {epoch})")
            print("="*80)

            # Print detailed statistics
            stats = print_fusion_weights(net, verbose=True)

            # Log to CSV file for plotting
            log_file = os.path.join(config.output_dir, "fusion_weights.csv")
            log_fusion_weights_to_file(net, log_file, epoch)

            print("="*80 + "\n")

    return trainer


# ============ 修改 train_dgl 函数 (约第 386 行处) ============
# 原代码:
#     handler = Checkpoint(
#         to_save,
#         DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
#         n_saved=2,
#         global_step_transform=lambda *_: trainer.state.epoch,
#     )
#     trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

# 修改为:
    handler = Checkpoint(
        to_save,
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        n_saved=2,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # 添加融合权重监控 (NEW!)
    if hasattr(net, 'middle_fusion_modules') and len(net.middle_fusion_modules) > 0:
        trainer = add_fusion_weight_monitoring(trainer, net, config)
        print("✅ DynamicFusionModule weight monitoring enabled")


# ============ 使用说明 ============
"""
执行后，训练过程中每 5 个 epoch 会输出：

================================================================================
DynamicFusionModule Weight Statistics (Epoch 50)
================================================================================
Updates: 15000

Router learned weights (from Softmax competition):
  w_graph: 0.6842
  w_text:  0.3158
  Sum:     1.0000 (should be ~1.0)

Effective weights (with double residual):
  Graph:  1.6842 (84.2%)
  Text:   0.3158 (15.8%)

Interpretation:
  ✅ Graph dominant (ratio: 5.33x)
     This is expected for material property prediction.
================================================================================

同时会生成 output_dir/fusion_weights.csv，可用于绘图：
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fusion_weights.csv')
plt.plot(df['epoch'], df['layer_2_w_graph'], label='w_graph')
plt.plot(df['epoch'], df['layer_2_w_text'], label='w_text')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.title('Router Weight Evolution During Training')
plt.savefig('fusion_weight_evolution.png')
```
"""
