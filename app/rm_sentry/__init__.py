"""RMUL 3v3 哨兵强化学习项目包。

最低 Python 版本：**3.10**
    rmul_3v3_env.py 与 train_ppo.py 使用了 PEP 604 的 `X | None` 注解语法，
    该语法自 Python 3.10 起才支持。

    - Ubuntu 22.04 自带 Python 3.10.6 ✅ 正好满足
    - Ubuntu 24.04 自带 3.12 ✅
    - 若用 3.9 或更低，会在导入时报 SyntaxError（下面给出更清晰的提示）
"""

import sys as _sys

if _sys.version_info < (3, 10):  # pragma: no cover
    raise RuntimeError(
        f"本项目需要 Python 3.10 及以上，当前为 "
        f"{_sys.version_info.major}.{_sys.version_info.minor}。\n"
        f"原因：rmul_3v3_env.py / train_ppo.py 使用了 PEP 604 的 `X | None` 语法。\n"
        f"处理：Ubuntu 22.04 自带 3.10 可满足；若用的是 3.9，请装 python3.10 或用 "
        f"`bash setup_env.sh` 让脚本自行建 venv。"
    )

from rm_sentry.rmul_3v3_env import (  # noqa: E402
    BatchedRMUL3v3,
    RMUL3v3Config,
    S,
    E,
)

try:
    from rm_sentry.rmul_3v3_env import RMUL3v3Env  # noqa: E402
except ImportError:  # gymnasium 未安装
    RMUL3v3Env = None

__all__ = ["BatchedRMUL3v3", "RMUL3v3Config", "S", "E", "RMUL3v3Env"]
__version__ = "0.2.0"
