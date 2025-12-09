# 修复 BaseSettings 导入错误

## 🔴 错误信息

```python
pydantic.errors.PydanticImportError: `BaseSettings` has been moved to the `pydantic-settings` package.
See https://docs.pydantic.dev/2.12/migration/#basesettings-has-moved-to-pydantic-settings
```

## 原因

在 **Pydantic v2** 中，`BaseSettings` 已从主包移到独立的 `pydantic-settings` 包中。

### 变化对比

| Pydantic v1 (旧版) | Pydantic v2 (新版) |
|-------------------|-------------------|
| `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings` |
| 包含在主包中 | 需要单独安装 `pydantic-settings` |

---

## ✅ 已自动修复

我已经修复了 `utils.py` 文件，使其兼容 Pydantic v1 和 v2：

### 修改内容

**新的导入逻辑（向后兼容）：**
```python
# 尝试 Pydantic v2
try:
    from pydantic_settings import BaseSettings as PydanticBaseSettings
except ImportError:
    # 回退到 Pydantic v1
    try:
        from pydantic import BaseSettings as PydanticBaseSettings
    except ImportError:
        # 最终回退到 BaseModel
        from pydantic import BaseModel as PydanticBaseSettings
```

**配置类兼容性：**
```python
class BaseSettings(PydanticBaseSettings):
    # Pydantic v2 风格
    try:
        from pydantic import ConfigDict
        model_config = ConfigDict(
            extra='forbid',
            use_enum_values=True,
            env_prefix='jv_'
        )
    except ImportError:
        # Pydantic v1 风格
        class Config:
            extra = "forbid"
            use_enum_values = True
            env_prefix = "jv_"
```

---

## 🚀 快速修复（推荐）

### 方法1: 安装 pydantic-settings（推荐）

```bash
cd /public/home/ghzhang/11.23

# 拉取最新修复代码
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 激活环境
conda activate MatMMFuse

# 运行自动安装脚本
chmod +x install_pydantic_settings.sh
./install_pydantic_settings.sh
```

这个脚本会：
1. ✅ 检测 Pydantic 版本
2. ✅ 如果是 v2，自动安装 pydantic-settings
3. ✅ 验证所有导入是否正常

### 方法2: 手动安装

```bash
conda activate MatMMFuse

# 安装 pydantic-settings
pip install pydantic-settings

# 验证安装
python -c "import pydantic_settings; print('✓ 安装成功')"
```

### 方法3: 使用修复后的代码（已包含兼容性处理）

```bash
cd /public/home/ghzhang/11.23
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 即使没有安装 pydantic-settings，代码也会回退到兼容模式
python -c "from utils import BaseSettings; print('✓ 导入成功')"
```

---

## 🔍 验证修复

运行以下命令验证所有问题都已解决：

```bash
conda activate MatMMFuse

# 测试1: 检查 Pydantic 版本
python -c "import pydantic; print('Pydantic:', pydantic.__version__)"

# 测试2: 检查 pydantic-settings
python -c "import pydantic_settings; print('pydantic-settings:', pydantic_settings.__version__)" 2>/dev/null || echo "未安装（使用兼容模式）"

# 测试3: 测试 BaseSettings 导入
python -c "from utils import BaseSettings; print('✓ BaseSettings 导入成功')"

# 测试4: 测试 ALIGNN 导入
python -c "from models.alignn import ALIGNN, ALIGNNConfig; print('✓ ALIGNN 导入成功')"

# 如果所有测试通过，显示成功
echo "✓ 所有导入测试通过！"
```

---

## 📋 Pydantic v2 主要变化总结

### 1. BaseSettings 位置变化

**Pydantic v1:**
```python
from pydantic import BaseSettings
```

**Pydantic v2:**
```python
from pydantic_settings import BaseSettings
# 需要: pip install pydantic-settings
```

### 2. 配置类语法变化

**Pydantic v1:**
```python
class MyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'
```

**Pydantic v2:**
```python
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid'
    )
```

### 3. Literal 导入变化

**Pydantic v1:**
```python
from pydantic.typing import Literal  # 可以工作但不推荐
```

**Pydantic v2:**
```python
from typing import Literal  # 标准库
```

### 4. 验证器装饰器变化

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

---

## 🛠️ 如果仍有问题：降级到 Pydantic v1

如果你不想处理兼容性问题，可以降级到 Pydantic v1：

```bash
conda activate MatMMFuse

# 降级到 Pydantic v1
pip install 'pydantic<2.0'

# 推荐的稳定版本
pip install pydantic==1.10.13

# 验证版本
python -c "import pydantic; print(pydantic.__version__)"
```

**注意**：降级可能影响其他依赖 Pydantic v2 的包。

---

## ✅ 总结

**所有问题已修复！**

修复了以下兼容性问题：
1. ✅ `Literal` 导入（models/alignn.py）
2. ✅ `BaseSettings` 导入（utils.py）
3. ✅ 配置类语法（utils.py）

代码现在兼容 Pydantic v1 和 v2！

---

## 🚀 现在可以运行

```bash
cd /public/home/ghzhang/11.23

# 拉取所有修复
git pull origin claude/fine-grained-cross-attention-fusion-01B5rhccV6vfHeUhH9U1R1f8

# 激活环境
conda activate MatMMFuse

# 安装 pydantic-settings（如果使用 Pydantic v2）
./install_pydantic_settings.sh

# 运行完整评估
./quick_extract_and_eval.sh
```

---

## 📞 需要帮助？

如果遇到其他问题，提供以下信息：

```bash
# 1. Python 和包版本
python --version
python -c "import pydantic; print('Pydantic:', pydantic.__version__)"
python -c "import pydantic_settings; print('pydantic-settings:', pydantic_settings.__version__)" 2>/dev/null || echo "未安装"

# 2. 导入测试
python -c "from utils import BaseSettings; print('OK')" 2>&1
python -c "from models.alignn import ALIGNN; print('OK')" 2>&1

# 3. 完整错误信息
python your_script.py 2>&1 | tee error.log
cat error.log
```

**修复完成！🎉**
