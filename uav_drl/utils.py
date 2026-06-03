# -*- coding: utf-8 -*-
"""通用工具函数。"""

from __future__ import annotations

import random

import numpy as np
import torch

from .config import DEFAULT_SEED


def fix_seed(seed: int = DEFAULT_SEED) -> None:
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def optional_point_3d(
    x: float | None,
    y: float | None,
    z: float | None,
    name: str,
    default_z: float,
) -> tuple[float, float, float] | None:
    """解析命令行中的三维起点/目标点。

    如果 x/y/z 都不提供，则返回 None，表示由环境随机采样。
    如果提供了 x/y 但没有提供 z，则使用 default_z，兼容旧的二维输入习惯。
    """
    if x is None and y is None and z is None:
        return None
    if x is None or y is None:
        raise SystemExit(f"--{name}-x and --{name}-y must be provided together.")
    if z is None:
        z = default_z
    return (float(x), float(y), float(z))


# 兼容旧名字。新代码请使用 optional_point_3d。
optional_point = optional_point_3d
