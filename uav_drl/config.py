# -*- coding: utf-8 -*-
"""三维地图、障碍物和环境参数配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


DEFAULT_SEED = 2026


@dataclass(frozen=True)
class BoxObstacle:
    """三维长方体建筑物/禁飞区。

    坐标含义：
        xmin, ymin, zmin: 长方体最小角点；
        xmax, ymax, zmax: 长方体最大角点。

    对小区场景来说，建筑物可以理解为从地面 z=0 向上延伸的长方体。
    """

    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    name: str = "obstacle"

    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """判断三维点是否进入障碍物，margin 用来留出无人机半径。"""
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
            and self.zmin - margin <= z <= self.zmax + margin
        )

    def distance_to(self, point: np.ndarray) -> float:
        """计算三维点到长方体的最短欧氏距离。"""
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        dx = max(self.xmin - x, 0.0, x - self.xmax)
        dy = max(self.ymin - y, 0.0, y - self.ymax)
        dz = max(self.zmin - z, 0.0, z - self.zmax)
        return math.sqrt(dx * dx + dy * dy + dz * dz)


# 兼容旧文档/旧代码中的名称：原来的 RectObstacle 现在等价于三维 BoxObstacle。
RectObstacle = BoxObstacle


def default_community_obstacles() -> list[BoxObstacle]:
    """默认三维小区地图中的建筑物/禁飞区。

    默认地图水平范围为 100 x 100，高度为 30。下面的楼栋从地面 z=0
    向上延伸到不同高度，形成三维避障环境。
    """

    return [
        BoxObstacle(18, 12, 0, 34, 43, 18, "building_a"),
        BoxObstacle(47, 8, 0, 63, 30, 14, "building_b"),
        BoxObstacle(72, 43, 0, 88, 76, 24, "building_c"),
        BoxObstacle(12, 61, 0, 43, 77, 12, "building_d"),
        BoxObstacle(51, 56, 0, 64, 91, 20, "building_e"),
        BoxObstacle(36, 39, 0, 45, 53, 28, "tower"),
    ]


@dataclass
class UAVEnvConfig:
    """三维无人机环境参数配置。"""

    # 水平地图范围 x=[0,map_width], y=[0,map_height]。
    map_width: float = 100.0
    map_height: float = 100.0

    # 垂直高度范围 z=[0,map_altitude]。
    map_altitude: float = 30.0

    # 每执行一个动作的三维飞行距离。
    step_length: float = 2.0

    # 每个 episode 最大步数。
    max_steps: int = 320

    # 第一阶段轨迹规划后处理参数。
    trajectory_dt: float = 1.0
    max_speed: float = 8.0
    max_acceleration: float = 3.0
    smoothing_iterations: int = 1

    # 距离目标点小于该三维半径，认为到达目标。
    goal_radius: float = 3.0

    # 无人机半径，用于三维碰撞检测。
    uav_radius: float = 0.7

    # 安全距离半径，靠近建筑物或边界时扣分。
    safety_radius: float = 4.0

    # 26 方向三维雷达探测距离和采样分辨率。
    lidar_range: float = 24.0
    lidar_resolution: float = 0.8

    # 线段碰撞检测采样间隔。
    collision_resolution: float = 0.4

    # 随机起点和目标点之间的最小三维距离。
    min_start_goal_distance: float = 35.0

    # 奖励函数权重。
    progress_reward_scale: float = 6.0
    distance_penalty_scale: float = 0.25
    step_penalty: float = 0.03
    hover_penalty: float = 0.08
    turn_penalty_scale: float = 0.04
    proximity_penalty_scale: float = 1.0
    altitude_penalty_scale: float = 0.01
    goal_reward: float = 140.0
    collision_penalty: float = -140.0
    timeout_penalty: float = -30.0

    # 默认三维建筑物列表。
    obstacles: list[BoxObstacle] = field(default_factory=default_community_obstacles)


def config_to_dict(config: UAVEnvConfig) -> dict[str, object]:
    """把环境配置转换成可以写入 checkpoint 的普通 dict。"""
    data = vars(config).copy()
    data["obstacles"] = [vars(obstacle).copy() for obstacle in config.obstacles]
    return data
