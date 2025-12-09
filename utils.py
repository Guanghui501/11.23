"""Shared pydantic settings configuration."""
import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

# Pydantic v2 compatibility: BaseSettings moved to pydantic-settings
try:
    # Try Pydantic v2 (requires pydantic-settings package)
    from pydantic_settings import BaseSettings as PydanticBaseSettings
except ImportError:
    try:
        # Fallback to Pydantic v1
        from pydantic import BaseSettings as PydanticBaseSettings
    except ImportError:
        # If BaseSettings not available, use BaseModel as fallback
        from pydantic import BaseModel as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    # Pydantic v2 uses model_config instead of nested Config class
    try:
        from pydantic import ConfigDict
        model_config = ConfigDict(
            extra='forbid',
            use_enum_values=True,
            env_prefix='jv_'
        )
    except ImportError:
        # Pydantic v1 style config
        class Config:
            """Configure BaseSettings behavior."""
            extra = "forbid"
            use_enum_values = True
            env_prefix = "jv_"


def plot_learning_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val
