# SLURM 批量训练作业使用指南

本指南说明如何使用SLURM作业系统进行批量晶体性质预测模型训练。

## 📁 文件说明

| 文件名 | 说明 |
|--------|------|
| `submit_slurm_batch.sh` | SLURM批处理脚本（主脚本） |
| `slurm_submit.sh` | 作业提交辅助工具 |
| `slurm_monitor.sh` | 作业监控脚本 |
| `logs/` | SLURM日志输出目录 |

## 🚀 快速开始

### 1. 配置SLURM脚本

在提交作业前，**必须**修改 `submit_slurm_batch.sh` 中的以下内容：

```bash
# 修改分区名称（根据你的集群配置）
#SBATCH --partition=gpu              # 改为你的GPU分区名

# 配置环境激活（二选一）
# 选项1: 使用conda环境
source ~/.bashrc
conda activate your_env_name         # 改为你的环境名

# 选项2: 使用module系统
module load cuda/11.8                # 根据需要修改CUDA版本
module load python/3.9
```

### 2. 检查资源配置

确认以下SLURM资源配置符合你的集群规范：

```bash
#SBATCH --gres=gpu:1                 # 每个任务1个GPU
#SBATCH --cpus-per-task=24           # 24个CPU核心
#SBATCH --mem=64G                    # 64GB内存
#SBATCH --time=48:00:00              # 最大48小时
```

### 3. 提交作业

```bash
# 方式1: 使用辅助脚本（推荐）
chmod +x slurm_submit.sh
./slurm_submit.sh

# 方式2: 直接提交
sbatch submit_slurm_batch.sh
```

## 📊 任务配置

当前配置会提交 **6个训练任务**（作业数组0-5）：

| 任务ID | 属性 | 随机种子 | 输出目录 |
|--------|------|----------|----------|
| 0 | bulk_modulus_kv | 42 | output_100epochs_42_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext |
| 1 | bulk_modulus_kv | 7 | output_100epochs_7_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext |
| 2 | bulk_modulus_kv | 123 | output_100epochs_123_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext |
| 3 | shear_modulus_gv | 42 | output_100epochs_42_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext |
| 4 | shear_modulus_gv | 7 | output_100epochs_7_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext |
| 5 | shear_modulus_gv | 123 | output_100epochs_123_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext |

## 🔍 监控作业

### 实时监控面板

```bash
chmod +x slurm_monitor.sh

# 单次查看
./slurm_monitor.sh

# 每5秒自动刷新
watch -n 5 './slurm_monitor.sh'
```

### 查看作业状态

```bash
# 查看你的所有作业
squeue -u $USER

# 查看特定作业（假设作业ID为12345）
squeue -j 12345

# 查看详细信息
scontrol show job 12345
```

### 查看训练日志

```bash
# 查看特定任务的实时日志（例如：作业ID=12345, 任务ID=0）
tail -f logs/slurm_12345_0.out

# 查看错误日志
tail -f logs/slurm_12345_0.err

# 列出所有日志文件
ls -lh logs/
```

## 🛠 作业管理

### 取消作业

```bash
# 取消整个作业数组
scancel 12345

# 取消特定的数组任务（例如：只取消任务3）
scancel 12345_3

# 取消你的所有作业
scancel -u $USER
```

### 重新提交失败的任务

如果某个特定任务失败，可以单独重新提交：

```bash
# 只提交任务3（shear_modulus_gv, seed=42）
sbatch --array=3 submit_slurm_batch.sh
```

## 📈 查看结果

训练完成后，每个任务的结果保存在对应的输出目录：

```bash
# 查看所有输出目录
ls -d output_100epochs_*

# 查看特定任务的结果
ls -lh output_100epochs_42_bs64_sw_ju_onlymiddle_bulk_modulus_kv_quantext/

# 典型的输出文件：
# ├── best_model.pth           # 最佳模型权重
# ├── training_log.csv         # 训练日志
# ├── config.json              # 配置文件
# └── predictions/             # 预测结果
```

## 🔧 自定义配置

### 修改训练参数

在 `submit_slurm_batch.sh` 中修改Python命令的参数：

```bash
python train_with_cross_modal_attention.py \
    --batch_size 128 \                      # 改为128
    --epochs 200 \                          # 改为200轮
    --learning_rate 1e-3 \                  # 调整学习率
    # ... 其他参数
```

### 添加更多属性或种子

在 `submit_slurm_batch.sh` 中修改数组定义：

```bash
# 添加更多属性
PROPERTIES=("bulk_modulus_kv" "shear_modulus_gv" "formation_energy_peratom")

# 添加更多随机种子
RANDOM_SEEDS=(42 7 123 456 789)

# 同时更新作业数组范围
#SBATCH --array=0-14               # 3属性 × 5种子 = 15个任务 (0-14)
```

## ⚠️ 注意事项

1. **环境配置**: 确保Python环境中已安装所有依赖包
2. **数据路径**: 确认 `--root_dir` 路径正确且可访问
3. **GPU资源**: 确认集群GPU分区名称和资源配置
4. **磁盘空间**: 每个任务约需10-20GB空间，确保有足够磁盘空间
5. **运行时间**: 根据数据集大小调整 `--time` 参数

## 🆘 常见问题

### Q1: 作业一直处于PENDING状态
**A**: 可能原因：
- GPU资源不足，等待资源释放
- 分区名称错误
- 资源请求超出限制

检查方法：
```bash
squeue -j <JOB_ID> -o "%A %T %R"  # 查看等待原因
```

### Q2: 作业立即失败
**A**: 检查错误日志：
```bash
cat logs/slurm_<JOB_ID>_<TASK_ID>.err
```

常见问题：
- Python环境未激活
- 缺少依赖包
- 数据路径不存在

### Q3: 如何增加或减少并行任务数
**A**: SLURM作业数组会自动管理并发。如果想控制同时运行的任务数：
```bash
#SBATCH --array=0-5%2              # 最多同时运行2个任务
```

### Q4: 训练中断后如何继续
**A**: 如果支持断点续训，在训练脚本中添加 `--resume` 选项，并修改SLURM脚本：
```bash
python train_with_cross_modal_attention.py \
    --resume "$OUTPUT_DIR/last_checkpoint.pth" \
    # ... 其他参数
```

## 📞 集群特定配置

不同的HPC集群可能有不同的要求，常见配置示例：

### 示例1: 指定GPU类型
```bash
#SBATCH --gres=gpu:v100:1          # 使用V100 GPU
#SBATCH --gres=gpu:a100:1          # 使用A100 GPU
```

### 示例2: 指定账户
```bash
#SBATCH --account=your_account     # 计费账户
```

### 示例3: 邮件通知
```bash
#SBATCH --mail-type=END,FAIL       # 结束或失败时发送邮件
#SBATCH --mail-user=your@email.com # 邮箱地址
```

## 📚 相关文档

- [SLURM官方文档](https://slurm.schedmd.com/documentation.html)
- [SLURM作业数组指南](https://slurm.schedmd.com/job_array.html)
- [训练脚本参数说明](train_with_cross_modal_attention.py)

---

**创建日期**: 2025-12-08
**作者**: Crystal Property Prediction Team
