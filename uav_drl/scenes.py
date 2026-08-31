# -*- coding: utf-8 -*-
"""四个可训练地图的注册表和障碍物适配器。"""

from __future__ import annotations

from dataclasses import dataclass

from standalone_maps.geometry import SceneMap
from standalone_maps.lanxianghu_villa_map import SCENE as LANXIANGHU_SCENE
from standalone_maps.sanming_garden_map import SCENE as SANMING_SCENE
from standalone_maps.spring_garden_phase2_map import SCENE as SPRING_SCENE

from .config import BoxObstacle, UAVEnvConfig, wujing_airfield_obstacles


# ==================== 可训练场景统一接口 ====================
@dataclass(frozen=True)
class TrainingScene:
    """训练入口使用的地图元数据。"""

    key: str
    display_name: str
    scene_name: str
    origin_gcj02: tuple[float, float]
    bounds: tuple[float, float, float, float]
    max_altitude: float
    default_obstacle_inflation: float
    default_start_xy: tuple[float, float]
    source_scene: SceneMap | None = None

    @property
    def map_width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def map_height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def obstacles(self, inflation: float | None = None) -> list[BoxObstacle]:
        """构建训练环境使用的轴对齐三维障碍物。"""
        margin = self.default_obstacle_inflation if inflation is None else float(inflation)
        if margin < 0.0:
            raise ValueError("Obstacle inflation must be non-negative.")
        if self.source_scene is None:
            return wujing_airfield_obstacles(margin)
        return _scene_map_to_box_obstacles(self.source_scene, margin)

    def make_config(
        self,
        obstacle_inflation: float | None = None,
        **overrides: object,
    ) -> UAVEnvConfig:
        """用地图默认值和可选覆盖项生成环境配置。"""
        margin = self.default_obstacle_inflation if obstacle_inflation is None else float(obstacle_inflation)
        values: dict[str, object] = {
            "scene_name": self.scene_name,
            "coordinate_system": "local_enu_from_gcj02_origin",
            "origin_lon_gcj02": self.origin_gcj02[0],
            "origin_lat_gcj02": self.origin_gcj02[1],
            "map_x_min": self.bounds[0],
            "map_y_min": self.bounds[1],
            "map_width": self.map_width,
            "map_height": self.map_height,
            "map_altitude": self.max_altitude,
            "obstacle_inflation": margin,
            "obstacles": self.obstacles(margin),
        }
        values.update({name: value for name, value in overrides.items() if value is not None})
        return UAVEnvConfig(**values)

    def default_start(self, config: UAVEnvConfig) -> tuple[float, float, float]:
        """返回该地图预先检查过的默认起点。"""
        return (self.default_start_xy[0], self.default_start_xy[1], config.uav_radius + 1e-3)


# ==================== 住宅区地图到训练障碍物的适配 ====================
def _scene_map_to_box_obstacles(scene: SceneMap, inflation: float) -> list[BoxObstacle]:
    """把旋转矩形建筑转换为保守的轴对齐包围盒。"""
    scene.validate()
    obstacles: list[BoxObstacle] = []
    for building in scene.buildings:
        footprint = building.footprint()
        xs = [point[0] for point in footprint]
        ys = [point[1] for point in footprint]
        obstacles.append(
            BoxObstacle(
                xmin=min(xs) - inflation,
                ymin=min(ys) - inflation,
                zmin=0.0,
                xmax=max(xs) + inflation,
                ymax=max(ys) + inflation,
                zmax=building.height,
                name=f"{building.building_id}_{scene.slug}_{building.confidence}",
            )
        )
    return obstacles


# 用原始 SceneMap 的元数据构造住宅区训练场景。
def _from_scene_map(
    key: str,
    scene: SceneMap,
    default_start_xy: tuple[float, float] = (15.0, 15.0),
) -> TrainingScene:
    return TrainingScene(
        key=key,
        display_name=scene.name,
        scene_name=scene.slug,
        origin_gcj02=scene.origin_gcj02,
        bounds=scene.bounds,
        max_altitude=scene.max_altitude,
        default_obstacle_inflation=2.0,
        default_start_xy=default_start_xy,
        source_scene=scene,
    )


# ==================== 四个场景注册表 ====================
SCENES: dict[str, TrainingScene] = {
    "wujing_airfield": TrainingScene(
        key="wujing_airfield",
        display_name="吴泾试飞场",
        scene_name="wujing_airfield_estimated",
        origin_gcj02=(121.4480000, 31.0680000),
        bounds=(0.0, -30.0, 500.0, 280.0),
        max_altitude=50.0,
        default_obstacle_inflation=8.0,
        default_start_xy=(250.0, 125.0),
    ),
    "lanxianghu_villa": _from_scene_map("lanxianghu_villa", LANXIANGHU_SCENE),
    "sanming_garden": _from_scene_map("sanming_garden", SANMING_SCENE),
    "spring_garden_phase2": _from_scene_map("spring_garden_phase2", SPRING_SCENE),
}


# ==================== 场景查询接口 ====================
def available_scene_keys() -> tuple[str, ...]:
    return tuple(SCENES)


def get_training_scene(key: str) -> TrainingScene:
    try:
        return SCENES[key]
    except KeyError as exc:
        choices = ", ".join(available_scene_keys())
        raise ValueError(f"Unknown scene {key!r}; choose one of: {choices}.") from exc


__all__ = ["SCENES", "TrainingScene", "available_scene_keys", "get_training_scene"]
