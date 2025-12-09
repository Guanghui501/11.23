# 修复 Pydantic v2 导入错误

## 🔴 错误信息

```python
ImportError: cannot import name 'Literal' from 'pydantic.typing'
(/public/home/ghzhang/.conda/envs/MatMMFuse/lib/python3.10/site-packages/pydantic/typing.py)
```

## 原因

在 **Pydantic v2** 中，`pydantic.typing` 模块已被移除。`Literal` 应该直接从 Python 标准库的 `typing` 模块导入。

### 变化对比

| Pydantic v1 (旧版) | Pydantic v2 (新版) |
|-------------------|-------------------|
| `from pydantic.typing import Literal` | `from typing import Literal` |
| `from pydantic.typing import Optional` | `from typing import Optional` |

---

## ✅ 已自动修复

我已经修复了 `models/alignn.py` 文件：

### 修改前
```python
from typing import Tuple, Union
from pydantic.typing import Literal  # ✗ Pydantic v2 不支持
```

### 修改后
```python
from typing import Tuple, Union, Literal  # ✓ 使用标准库
```

---

## 🚀 验证修复

运行以下命令验证修复是否成功：

```bash
cd /public/home/ghzhang/11.23
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 测试导入
python -c "from models.alignn import ALIGNN, ALIGNNConfig; print('✓ 导入成功')"
```

---

## 🔧 如果其他文件也有问题

如果你在其他 Python 文件中也遇到同样的错误，使用自动修复脚本：

```bash
# 运行自动修复脚本
python fix_pydantic_imports.py
```

这个脚本会：
1. ✅ 搜索所有包含 `from pydantic.typing import` 的文件
2. ✅ 自动替换为 `from typing import`
3. ✅ 创建备份文件 (.bak)
4. ✅ 验证修复结果

---

## 🔍 其他 Pydantic v2 兼容性问题

如果修复后仍有其他 Pydantic 相关错误，可能需要更多更改：

### 常见兼容性问题

#### 1. BaseModel 配置更改

**Pydantic v1:**
```python
class Config:
    arbitrary_types_allowed = True
```

**Pydantic v2:**
```python
model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### 2. 验证器装饰器更改

**Pydantic v1:**
```python
from pydantic import validator

@validator('field_name')
def validate_field(cls, v):
    return v
```

**Pydantic v2:**
```python
from pydantic import field_validator

@field_validator('field_name')
def validate_field(cls, v):
    return v
```

#### 3. JSON 方法更改

**Pydantic v1:**
```python
model.dict()
model.json()
```

**Pydantic v2:**
```python
model.model_dump()
model.model_dump_json()
```

---

## 🛠️ 替代方案：降级 Pydantic（如果需要）

如果修复后仍有兼容性问题，可以临时降级到 Pydantic v1：

```bash
conda activate MatMMFuse

# 降级到 Pydantic v1
pip install 'pydantic<2.0'

# 或指定具体版本
pip install pydantic==1.10.13

# 验证版本
python -c "import pydantic; print(pydantic.__version__)"
```

**注意**：降级可能影响其他依赖 Pydantic v2 的包。

---

## 📋 检查当前 Pydantic 版本

```bash
python -c "import pydantic; print('Pydantic version:', pydantic.__version__)"
```

输出示例：
- `Pydantic version: 2.x.x` → 需要使用 `from typing import Literal`
- `Pydantic version: 1.x.x` → 可以使用 `from pydantic.typing import Literal`

---

## ✅ 总结

**问题已解决！** 代码已修复为兼容 Pydantic v2。

现在可以正常运行所有脚本：

```bash
cd /public/home/ghzhang/11.23

# 拉取最新修复
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 激活环境
conda activate MatMMFuse

# 运行评估
./quick_extract_and_eval.sh
```

---

## 📞 需要帮助？

如果遇到其他 Pydantic 相关错误，提供以下信息：

1. **Pydantic 版本**:
   ```bash
   python -c "import pydantic; print(pydantic.__version__)"
   ```

2. **完整错误信息**:
   ```bash
   python your_script.py 2>&1 | tee error.log
   ```

3. **导入测试**:
   ```bash
   python -c "from models.alignn import ALIGNN; print('OK')"
   ```

---

**修复完成！🎉**
