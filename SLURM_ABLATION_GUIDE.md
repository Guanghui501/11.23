# SLURM 消融实验提交指南

## 🔴 问题：需要sudo密码

SLURM作业提交**不应该**需要sudo密码。如果遇到这个问题，通常是因为：

1. ❌ 脚本尝试在没有权限的目录创建文件
2. ❌ 使用了需要root权限的命令
3. ❌ Conda环境配置问题
4. ❌ Module系统需要特殊权限

---

## ✅ 解决方案：使用无sudo版本

我已经创建了一个**不需要任何特殊权限**的脚本：`submit_ablation_no_sudo.sh`

### 关键改进

1. **工作目录配置**：使用你有权限的目录
   ```bash
   WORK_DIR="${HOME}/ablation_experiments"
   ```

2. **提前创建目录**：在sbatch外部创建，避免权限问题
   ```bash
   mkdir -p "$output_dir" 2>/dev/null
   ```

3. **改进的Conda激活**：尝试多个可能的路径
   ```bash
   source ~/.bashrc || source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
   ```

4. **可选的module purge**：如果不需要可以注释掉

---

## 🚀 使用方法

### 步骤1: 修改配置（如果需要）

打开脚本，修改以下配置：

```bash
# 工作目录（确保你有写权限）
WORK_DIR="${HOME}/ablation_experiments"
# 或者使用绝对路径
WORK_DIR="/public/home/ghzhang/ablation_experiments"

# 数据集路径
DATA_ROOT="/public/home/ghzhang/crysmmnet-main-2/dataset"

# Conda环境名称
CONDA_ENV="sganet"

# 属性和种子
PROPERTIES=("shear_modulus_gv")
RANDOM_SEEDS=(7)
```

### 步骤2: 赋予执行权限

```bash
chmod +x submit_ablation_no_sudo.sh
```

### 步骤3: 运行脚本

```bash
./submit_ablation_no_sudo.sh
```

### 步骤4: 查看提交结果

脚本会输出类似：

```
==========================================
所有作业提交完成！
==========================================

提交的作业列表（串行执行顺序）:
  1. 作业ID: 123456 - Baseline: No Fine-grained + With Middle fusion
  2. 作业ID: 123457 - Fine-grained + Projection + Middle fusion
  3. 作业ID: 123458 - Fine-grained + Projection + Cross-modal, No Middle fusion
  4. 作业ID: 123459 - Fine-grained + Projection + Cross-modal + Middle fusion (Full)
```

---

## 🔧 如果仍然需要sudo密码

### 诊断步骤

#### 1. 检查是否真的需要sudo

```bash
# 查看完整错误信息
./submit_ablation_no_sudo.sh 2>&1 | tee submission.log
cat submission.log
```

#### 2. 测试目录权限

```bash
# 测试是否能创建工作目录
mkdir -p ~/ablation_experiments
echo "Test" > ~/ablation_experiments/test.txt
rm ~/ablation_experiments/test.txt

# 如果失败，改用其他目录
WORK_DIR="/scratch/$USER/ablation"  # 或其他你有权限的目录
```

#### 3. 测试Conda激活

```bash
# 测试conda是否需要sudo
source ~/.bashrc
conda activate sganet

# 如果需要初始化
conda init bash
source ~/.bashrc
```

#### 4. 检查sbatch权限

```bash
# 测试是否能提交作业
sbatch <<EOF
#!/bin/bash
#SBATCH -J test_job
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -o test-%j.out

echo "Test job running"
sleep 10
EOF

# 如果提示需要权限，联系集群管理员
```

---

## 📋 脚本功能说明

### 4个消融实验配置

| 配置 | Fine-grained | Projection | Cross-modal | Middle Fusion |
|------|-------------|------------|-------------|---------------|
| **1. Baseline** | ❌ | ❌ | ❌ | ✅ |
| **2. FG+Proj+Middle** | ✅ | ✅ | ❌ | ✅ |
| **3. FG+Proj+Cross** | ✅ | ✅ | ✅ | ❌ |
| **4. Full** | ✅ | ✅ | ✅ | ✅ |

### 串行执行

脚本使用SLURM依赖链，确保任务按顺序执行：

```
任务1 (Baseline) → 任务2 (FG+Proj+Middle) → 任务3 (FG+Proj+Cross) → 任务4 (Full)
```

每个任务只有在前一个任务**成功完成**后才会开始。

---

## 🛠️ 常见问题

### Q1: 如何取消作业链？

```bash
# 取消所有你的作业
scancel -u $USER

# 或取消特定作业（脚本会输出作业ID）
scancel 123456 123457 123458 123459
```

### Q2: 如何查看作业状态？

```bash
# 查看所有作业
squeue -u $USER

# 查看详细信息
squeue -u $USER -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'

# 查看依赖关系
squeue -u $USER -o '%.18i %.30j %.8T %.10r'
```

### Q3: 如何查看作业输出？

```bash
# 输出文件位置
ls -lh ~/ablation_experiments/output_*/

# 查看实时输出
tail -f ~/ablation_experiments/output_100epochs_7_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext/train_*-*.out

# 查看错误信息
tail -f ~/ablation_experiments/output_100epochs_7_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext/train_*-*.err
```

### Q4: 如何并行而不是串行执行？

如果想要4个配置同时运行（而不是依次执行），修改脚本：

```bash
# 在每个submit_job调用中，将 "$PREV_JOB_ID" 改为 ""
JOB_ID=$(submit_job "$JOB_NAME" "$OUTPUT_DIR" "$PROPERTY" "$SEED" \
                   "$CONFIG_1_FG" "$CONFIG_1_PROJ" "$CONFIG_1_MIDDLE" "$CONFIG_1_CROSS" "")
                   #                                                                      ^^
                   #                                                       空字符串 = 无依赖
```

但注意：并行执行需要4个GPU！

### Q5: 脚本中的module purge报错怎么办？

如果你的集群不使用module系统，注释掉这行：

```bash
# 在heredoc中找到这行并注释
# # 清理模块环境
# module purge
```

---

## 📊 预期输出结构

```
~/ablation_experiments/
├── output_100epochs_7_bs64_sw_ju_onlymiddle_shear_modulus_gv_quantext/
│   ├── train_shear_modulus_gv_seed7_baseline-123456.out
│   ├── train_shear_modulus_gv_seed7_baseline-123456.err
│   ├── best_model.pt
│   ├── checkpoint_epoch_*.pt
│   └── history_*.json
├── output_100epochs_7_bs64_sw_ju_middle_fg_proj_shear_modulus_gv_quantext/
│   ├── train_shear_modulus_gv_seed7_fg_proj_middle-123457.out
│   └── ...
├── output_100epochs_7_bs64_sw_ju_fg_proj_crossmodal_nomiddle_shear_modulus_gv_quantext/
│   └── ...
└── output_100epochs_7_bs64_sw_ju_fg_proj_crossmodal_middle_shear_modulus_gv_quantext/
    └── ...
```

---

## ✅ 验证清单

提交前检查：

- [ ] 修改了 `WORK_DIR` 为你有权限的目录
- [ ] 确认 `DATA_ROOT` 路径正确
- [ ] 确认 `CONDA_ENV` 名称正确
- [ ] 测试过 conda activate 不需要sudo
- [ ] 测试过 sbatch 命令可用
- [ ] 脚本有执行权限 (`chmod +x`)

提交后检查：

- [ ] 所有4个作业都成功提交（看到作业ID）
- [ ] 作业在队列中（`squeue -u $USER`）
- [ ] 输出目录已创建
- [ ] 第一个作业开始运行后有输出文件

---

## 📞 需要帮助？

如果仍然遇到问题，提供以下信息：

```bash
# 1. 完整错误信息
./submit_ablation_no_sudo.sh 2>&1 | tee error.log
cat error.log

# 2. 权限测试
ls -ld ~
ls -ld ~/ablation_experiments 2>/dev/null || echo "目录不存在"
mkdir -p ~/test_permissions && echo "✓ 可以创建目录" || echo "✗ 无法创建目录"

# 3. SLURM配置
sinfo
sbatch --version
squeue -u $USER

# 4. Conda环境
conda env list
conda activate sganet && echo "✓ Conda激活成功" || echo "✗ Conda激活失败"
```

---

**祝实验顺利！🎉**
