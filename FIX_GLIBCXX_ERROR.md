# 修复 GLIBCXX_3.4.30 错误

## 🔴 错误信息

```
OSError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
(required by /public/home/ghzhang/dgl/build/libdgl.so)
```

## 原因

你的 DGL 库是用较新的 GCC 编译的，需要 GLIBCXX_3.4.30 版本，但系统的 libstdc++ 版本较旧。

---

## ✅ 解决方案（3种方法）

### 方法1: 使用 conda 环境的 libstdc++（最简单，推荐）

在运行任何 Python 脚本前，先设置环境变量：

```bash
# 激活 conda 环境
conda activate MatMMFuse

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 然后运行脚本
python extract_test_set_from_csv.py ...
```

**或者使用包装脚本（已自动处理）：**

```bash
# 已更新的一键脚本会自动设置环境
./quick_extract_and_eval.sh

# 或者使用专门的提取脚本
./run_extract_with_fix.sh
```

---

### 方法2: 在 conda 环境中安装/更新 libstdc++

```bash
# 激活 conda 环境
conda activate MatMMFuse

# 安装最新的 libstdc++
conda install -c conda-forge libstdcxx-ng

# 验证安装
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.30
```

---

### 方法3: 将环境变量永久添加到 conda 环境

```bash
# 激活环境
conda activate MatMMFuse

# 创建激活脚本
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF

# 创建停用脚本
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh << 'EOF'
#!/bin/sh
unset LD_LIBRARY_PATH
EOF

# 重新激活环境使其生效
conda deactivate
conda activate MatMMFuse

# 验证
echo $LD_LIBRARY_PATH
```

之后每次激活 MatMMFuse 环境时，LD_LIBRARY_PATH 会自动设置。

---

## 🚀 快速运行（推荐）

我已经更新了所有脚本，自动处理 GLIBCXX 问题。

### 选项A: 使用自动修复脚本

```bash
cd /public/home/ghzhang/11.23

# 拉取最新代码
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 激活环境（必须）
conda activate MatMMFuse

# 运行（会自动设置 LD_LIBRARY_PATH）
./quick_extract_and_eval.sh
```

### 选项B: 只提取测试集

```bash
conda activate MatMMFuse
./run_extract_with_fix.sh
```

### 选项C: 手动设置后运行

```bash
conda activate MatMMFuse
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python extract_test_set_from_csv.py \
    --predictions_csv /public/home/ghzhang/crysmmnet-main-2/src/coGN/band-shuangyanma/111my/output_100epochs_42_bs128_sw_ju_onlymiddle/mbj_bandgap/predictions_best_test_model_test.csv \
    --preprocessed_dir /public/home/ghzhang/preprocessed_data \
    --dataset jarvis \
    --property mbj_bandgap \
    --output_dir ./corrected_test_set
```

---

## 🔍 诊断工具

检查你的环境是否有问题：

```bash
chmod +x check_glibcxx.sh
./check_glibcxx.sh
```

这会显示：
- 系统 libstdc++ 支持的版本
- conda 环境 libstdc++ 支持的版本
- 是否包含所需的 GLIBCXX_3.4.30
- 推荐的解决方案

---

## ❓ 故障排除

### 问题1: conda 环境没有 libstdc++.so.6

**症状：**
```
✗ 未找到 conda 环境中的 libstdc++
```

**解决：**
```bash
conda install -c conda-forge libstdcxx-ng
```

### 问题2: conda 环境的 libstdc++ 也没有 GLIBCXX_3.4.30

**症状：**
```
✗ Conda 环境也不包含 GLIBCXX_3.4.30
```

**解决：**
```bash
# 更新 conda
conda update conda

# 重新安装最新版 libstdc++
conda install -c conda-forge libstdcxx-ng --force-reinstall

# 验证
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX | tail -5
```

### 问题3: 仍然报错

**可能原因：** DGL 可能链接了错误的库

**临时解决方案 - 使用系统 Python（如果服务器有）：**
```bash
# 查看可用的 Python 模块
module avail python

# 加载系统 Python
module load python/3.10  # 或其他版本

# 查看是否有预装的包
python -c "import dgl; print(dgl.__version__)"

# 如果有，使用系统 Python 运行
python extract_test_set_from_csv.py ...
```

**或者重新安装 DGL：**
```bash
conda activate MatMMFuse

# 使用 conda 安装 DGL（推荐）
conda install -c dglteam dgl

# 或使用 pip
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

---

## 📋 验证修复

运行以下命令验证问题已解决：

```bash
conda activate MatMMFuse
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -c "import dgl; print('DGL version:', dgl.__version__)"
```

如果没有报错，说明修复成功！

---

## 📝 总结

**最简单的方法：**

```bash
conda activate MatMMFuse
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./quick_extract_and_eval.sh
```

**永久解决（推荐）：**

按照"方法3"将环境变量添加到 conda 激活脚本中，之后就不需要每次手动设置了。

---

所有脚本（`quick_extract_and_eval.sh` 和 `run_extract_with_fix.sh`）已经包含了自动修复逻辑，直接运行即可！
