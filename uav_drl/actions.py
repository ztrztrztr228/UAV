# -*- coding: utf-8 -*-
"""无人机三维离散动作空间定义。"""

from __future__ import annotations

import numpy as np


def _direction_name(dx: int, dy: int, dz: int) -> str:
    """把三维方向分量转换成可读动作名称。"""
    parts: list[str] = []
    if dy > 0:
        parts.append("north")
    elif dy < 0:
        parts.append("south")

    if dx > 0:
        parts.append("east")
    elif dx < 0:
        parts.append("west")

    if dz > 0:
        parts.append("up")
    elif dz < 0:
        parts.append("down")

    return "_".join(parts)


def _make_3d_action_space() -> tuple[tuple[str, ...], np.ndarray]:
    """生成 26 个三维移动方向 + 1 个悬停动作。

    三维网格中 dx, dy, dz 都可以取 -1、0、1。去掉 (0,0,0) 后得到
    26 个邻接方向，包括水平移动、垂直移动、斜向上升/下降等。
    每个方向都会归一化为单位向量，保证所有方向乘以 step_length 后
    实际飞行距离一致。
    """
    names: list[str] = []
    directions: list[np.ndarray] = []

    for dz in (0, 1, -1):
        for dy in (0, 1, -1):
            for dx in (1, 0, -1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                vector = np.asarray([dx, dy, dz], dtype=np.float32)
                vector = vector / np.linalg.norm(vector)
                names.append(_direction_name(dx, dy, dz))
                directions.append(vector)

    names.append("hover")
    directions.append(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    return tuple(names), np.asarray(directions, dtype=np.float32)


# ACTION_NAMES[i] 与 ACTION_DIRECTIONS[i] 一一对应。
ACTION_NAMES, ACTION_DIRECTIONS = _make_3d_action_space()

# 最后一个动作固定为悬停，奖励函数会用它判断是否需要扣悬停惩罚。
HOVER_ACTION_INDEX = len(ACTION_NAMES) - 1
