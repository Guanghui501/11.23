# SLURM 快速开始指南

## ⚡ 三步启动

### 1️⃣ 修改配置 (首次使用)
```bash
vi submit_slurm_batch.sh
```
修改以下两处：
- `#SBATCH --partition=gpu` → 改为你的分区名
- 取消注释环境激活命令（conda或module）

### 2️⃣ 提交作业
```bash
./slurm_submit.sh
```

### 3️⃣ 监控进度
```bash
# 实时监控
watch -n 5 './slurm_monitor.sh'

# 或者查看日志
tail -f logs/slurm_*_0.out
```

---

## 📋 常用命令速查

| 操作 | 命令 |
|------|------|
| 提交作业 | `./slurm_submit.sh` 或 `sbatch submit_slurm_batch.sh` |
| 查看所有作业 | `squeue -u $USER` |
| 监控面板 | `./slurm_monitor.sh` |
| 取消所有作业 | `scancel -u $USER` |
| 取消特定作业 | `scancel <JOB_ID>` |
| 查看日志 | `tail -f logs/slurm_<JOB_ID>_<TASK_ID>.out` |
| 查看错误 | `tail -f logs/slurm_<JOB_ID>_<TASK_ID>.err` |

---

## 🎯 作业配置一览

**训练任务**: 6个 (2个属性 × 3个种子)

| 任务ID | 属性 | 种子 |
|--------|------|------|
| 0-2 | bulk_modulus_kv | 42, 7, 123 |
| 3-5 | shear_modulus_gv | 42, 7, 123 |

**资源配置**:
- GPU: 1个/任务
- CPU: 24核/任务
- 内存: 64GB/任务
- 时间: 最多48小时

---

## 🔧 快速修改

### 添加更多属性
```bash
# 在 submit_slurm_batch.sh 中修改
PROPERTIES=("bulk_modulus_kv" "shear_modulus_gv" "new_property")
RANDOM_SEEDS=(42 7 123)

# 更新作业数组
#SBATCH --array=0-8    # 3属性 × 3种子 = 9任务
```

### 修改训练参数
```bash
# 在 submit_slurm_batch.sh 的 python 命令中修改
--batch_size 128 \
--epochs 200 \
--learning_rate 1e-3 \
```

### 限制并发任务数
```bash
# 最多同时运行2个任务
#SBATCH --array=0-5%2
```

---

## ❓ 问题排查

### 作业一直PENDING?
```bash
squeue -j <JOB_ID> -o "%A %T %R"  # 查看原因
```
常见原因：资源不足、分区名错误、权限问题

### 作业秒退?
```bash
cat logs/slurm_<JOB_ID>_0.err    # 查看错误
```
常见原因：环境未激活、缺少依赖、路径错误

### 如何重跑失败的任务?
```bash
sbatch --array=3 submit_slurm_batch.sh  # 只跑任务3
```

---

## 📊 新旧方法对比

| 特性 | 原bash脚本 | SLURM系统 |
|------|-----------|----------|
| 并行方式 | 串行执行 | 真正并行 |
| 资源管理 | 手动 | 自动调度 |
| 作业管理 | 进程ID | 统一作业ID |
| 日志管理 | 混合在一起 | 分离stdout/stderr |
| 失败重试 | 手动 | 简单命令 |
| 集群支持 | ❌ | ✅ |
| 资源监控 | 基本 | 完善 |

---

## 📚 详细文档

查看完整文档: `cat SLURM_README.md`

---

**提示**: 首次使用请阅读 `SLURM_README.md` 了解详细配置
