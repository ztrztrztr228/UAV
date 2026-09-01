# -*- coding: utf-8 -*-
"""三维地图、障碍物和环境参数配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np


DEFAULT_SEED = 2026

# ==================== 场景原始数据 ====================
# 吴泾试飞场局部 ENU 坐标元数据。经纬度仅用于记录场景基准，仿真内部全部使用米。
WUJING_ORIGIN_LON_GCJ02 = 121.4480000
WUJING_ORIGIN_LAT_GCJ02 = 31.0680000
WUJING_BUILDING_INFLATION_M = 8.0

# id, 名称, 中心东向坐标, 中心北向坐标, 东西尺寸, 南北尺寸, 假设高度, 置信度。
# B09/B10 是旧厂房群整体包络，暂按 25 m；其余仓库/厂房暂按 15 m。
WUJING_BUILDING_ESTIMATES: tuple[tuple[str, str, float, float, float, float, float, str], ...] = (
    ("B01", "西北长条厂房", 77.4, 226.8, 69.1, 46.0, 15.0, "medium"),
    ("B02", "西侧设备或厂房", 80.0, 176.0, 74.2, 35.8, 15.0, "medium"),
    ("B03", "中部小型白顶库", 200.2, 130.8, 53.7, 34.8, 15.0, "medium"),
    ("B04", "东北大型白顶库", 379.2, 193.3, 150.9, 62.4, 15.0, "medium"),
    ("B05", "东北中型厂房A", 335.8, 135.4, 74.2, 46.0, 15.0, "medium"),
    ("B06", "东北中型厂房B", 421.4, 135.4, 71.6, 46.0, 15.0, "medium"),
    ("B07", "西南白顶厂房A", 120.9, 22.3, 74.2, 58.8, 15.0, "medium"),
    ("B08", "西南白顶厂房B", 197.6, 22.8, 69.1, 57.8, 15.0, "medium"),
    ("B09", "东南旧厂房群A", 357.5, 26.1, 76.7, 51.1, 25.0, "low"),
    ("B10", "东南旧厂房群B", 438.1, 32.5, 48.6, 79.3, 25.0, "low"),
)


# ==================== 障碍物几何模型 ====================
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


# ==================== 吴泾场景障碍物构建 ====================
def wujing_airfield_obstacles(inflation: float = WUJING_BUILDING_INFLATION_M) -> list[BoxObstacle]:
    """根据公开影像估算值构建吴泾试飞场建筑物包络框。

    ``inflation`` 只在水平面向外膨胀，用于覆盖约 0.51 m/px 影像量测误差和
    建筑轮廓不确定性。高度不是实测值：B01-B08 假设 15 m，B09-B10 假设
    25 m。这些障碍物只能用于仿真，不能直接作为真实飞行安全边界。
    """
    if inflation < 0.0:
        raise ValueError("Building inflation must be non-negative.")

    obstacles: list[BoxObstacle] = []
    for building_id, name, x, y, width, depth, height, confidence in WUJING_BUILDING_ESTIMATES:
        half_width = width / 2.0 + inflation
        half_depth = depth / 2.0 + inflation
        obstacles.append(
            BoxObstacle(
                xmin=x - half_width,
                ymin=y - half_depth,
                zmin=0.0,
                xmax=x + half_width,
                ymax=y + half_depth,
                zmax=height,
                name=f"{building_id}_{name}_{confidence}",
            )
        )
    return obstacles


# 保留旧函数名，避免已有调用失效；默认场景已经切换为吴泾试飞场。
default_community_obstacles = wujing_airfield_obstacles


# ==================== 统一环境参数 ====================
@dataclass
class UAVEnvConfig:
    """三维无人机环境参数配置。"""

    scene_name: str = "wujing_airfield_estimated"
    coordinate_system: str = "local_enu_from_gcj02_origin"
    origin_lon_gcj02: float = WUJING_ORIGIN_LON_GCJ02
    origin_lat_gcj02: float = WUJING_ORIGIN_LAT_GCJ02
    obstacle_inflation: float = WUJING_BUILDING_INFLATION_M

    # 局部坐标范围：x=[map_x_min, map_x_min+map_width]，y 同理。
    # y 下界必须为负数，因为 B07/B08/B10 的估算包络跨过局部原点南侧。
    map_x_min: float = 0.0
    map_y_min: float = -30.0
    map_width: float = 500.0
    map_height: float = 310.0

    # 垂直高度范围 z=[0,map_altitude]。
    map_altitude: float = 50.0

    # 第一阶段参数，仅为旧调用和轨迹偏差默认值保留。
    step_length: float = 2.0

    # 每个 episode 最大步数。
    max_steps: int = 320

    # 第二阶段离散时间动力学参数。速度单位为 m/s，加速度单位为 m/s²，
    # jerk 单位为 m/s³；下降速度以正的幅值保存。
    trajectory_dt: float = 0.5
    max_horizontal_speed: float = 23.0
    max_speed: float = 23.18
    max_climb_speed: float = 2.85
    max_descent_speed: float = 1.65
    max_climb_angle_deg: float = 90.0

    # 15.5 m/s² 是日志瞬时峰值；正常飞行控制采用 3 m/s²（给定 3--5
    # m/s² 区间的保守端），制动使用实测最大减速度 3.09 m/s²。
    max_acceleration: float = 15.5
    normal_acceleration: float = 3.0
    max_deceleration: float = 3.09

    # 轨迹约束采用平滑后峰值；原始日志峰值单独保留用于追溯。
    max_jerk: float = 78.0
    raw_max_jerk: float = 142.0
    goal_speed_tolerance: float = 1.0
    smoothing_iterations: int = 1

    # 距离目标点小于该三维半径，认为到达目标。
    goal_radius: float = 3.0

    # 无人机半径，用于三维碰撞检测。
    uav_radius: float = 0.7

    # 安全距离半径，靠近建筑物或边界时扣分。
    safety_radius: float = 4.0

    # 26 方向三维雷达探测距离和采样分辨率。40 m 让常用飞行速度下的
    # 制动风险更早进入状态，同时避免把扫描成本提高到最大制动距离对应的水平。
    lidar_range: float = 40.0
    lidar_resolution: float = 0.8

    # 线段碰撞检测采样间隔。
    collision_resolution: float = 0.4

    # 随机起点和目标点之间的最小三维距离。
    min_start_goal_distance: float = 35.0

    # v3 奖励只保留四个密集项：归一化目标进度、时间成本、制动安全风险和 jerk。
    # 进度先除以单步最大位移并裁剪到 [-1, 1]，因此终止奖惩不会再被米制进度淹没。
    reward_shaping_version: int = 3
    progress_reward_scale: float = 1.0
    step_penalty: float = 0.01
    safety_risk_penalty_scale: float = 1.0
    jerk_penalty_scale: float = 0.02
    goal_reward: float = 50.0
    collision_penalty: float = -50.0
    timeout_penalty: float = -20.0

    # 默认三维建筑物列表。
    obstacles: list[BoxObstacle] = field(default_factory=wujing_airfield_obstacles)

    @property
    def climb_angle_at_max_horizontal_speed_deg(self) -> float:
        """最大水平速度与最大上升速度同时出现时的航迹爬升角。"""
        return math.degrees(math.atan2(self.max_climb_speed, self.max_horizontal_speed))


# ==================== Checkpoint 配置序列化 ====================
def config_to_dict(config: UAVEnvConfig) -> dict[str, object]:
    """把环境配置转换成可以写入 checkpoint 的普通 dict。"""
    data = vars(config).copy()
    data["obstacles"] = [vars(obstacle).copy() for obstacle in config.obstacles]
    return data


def config_from_dict(data: dict[str, object]) -> UAVEnvConfig:
    """从 checkpoint 中保存的普通字典恢复环境配置。"""
    if not isinstance(data, dict):
        raise TypeError("config data must be a dictionary.")
    allowed_names = {item.name for item in fields(UAVEnvConfig)}
    values = {name: value for name, value in data.items() if name in allowed_names}
    obstacle_values = values.get("obstacles")
    if obstacle_values is not None:
        if not isinstance(obstacle_values, (list, tuple)):
            raise ValueError("config obstacles must be a list of box dictionaries.")
        obstacles: list[BoxObstacle] = []
        for item in obstacle_values:
            if isinstance(item, BoxObstacle):
                obstacles.append(item)
            elif isinstance(item, dict):
                obstacles.append(BoxObstacle(**item))
            else:
                raise ValueError("each config obstacle must be a box dictionary.")
        values["obstacles"] = obstacles
    return UAVEnvConfig(**values)
