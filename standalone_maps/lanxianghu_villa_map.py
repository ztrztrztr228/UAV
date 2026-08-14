"""兰香湖贰号东/西区估算地图，可经场景适配器接入训练环境。"""

from __future__ import annotations

import math
from pathlib import Path

from standalone_maps.geometry import Building, SceneMap, render_scene, save_scene


BUILDING_TYPES = (
    (23.0, 15.0),
    (21.0, 17.0),
    (22.0, 14.0),
    (19.0, 16.0),
    (16.0, 13.0),
)


def _curved_row(
    prefix: str,
    count: int,
    x_start: float,
    x_end: float,
    y_base: float,
    curve: float,
    type_offset: int,
) -> list[Building]:
    buildings: list[Building] = []
    span = x_end - x_start
    for index in range(count):
        ratio = 0.5 if count == 1 else index / (count - 1)
        normalized = ratio * 2.0 - 1.0
        x = x_start + span * ratio
        y = y_base + curve * normalized * normalized
        slope = 4.0 * curve * normalized / span
        yaw = math.degrees(math.atan(slope))
        length, width = BUILDING_TYPES[(index + type_offset) % len(BUILDING_TYPES)]
        buildings.append(
            Building(
                building_id=f"{prefix}_{index + 1:02d}",
                center_x=x,
                center_y=y,
                length=length,
                width=width,
                height=10.0,
                yaw_deg=yaw,
                confidence="low",
                source_note="Amap satellite layout + project PDF size classes",
            )
        )
    return buildings


def build_scene() -> SceneMap:
    buildings: list[Building] = []

    # 东区五条弧形排布，弧度和数量按高德卫星图逐排近似。
    east_rows = (
        ("east_north", 12, 382.0, 696.0, 520.0, 25.0),
        ("east_row2", 13, 374.0, 704.0, 415.0, 20.0),
        ("east_row3", 14, 365.0, 710.0, 310.0, 18.0),
        ("east_row4", 15, 355.0, 716.0, 205.0, 16.0),
        ("east_south", 16, 345.0, 722.0, 95.0, 20.0),
    )
    for row_index, row in enumerate(east_rows):
        buildings.extend(_curved_row(*row, type_offset=row_index))

    # 西区沿水系呈扇形排布，建筑朝向随弧线变化。
    west_rows = (
        ("west_north", 7, 82.0, 282.0, 480.0, -22.0),
        ("west_row2", 8, 68.0, 294.0, 382.0, -18.0),
        ("west_row3", 9, 55.0, 306.0, 286.0, -15.0),
        ("west_row4", 10, 42.0, 312.0, 190.0, -12.0),
        ("west_south", 11, 30.0, 318.0, 92.0, -10.0),
    )
    for row_index, row in enumerate(west_rows):
        buildings.extend(_curved_row(*row, type_offset=row_index + 2))

    return SceneMap(
        name="兰香湖贰号东区与西区",
        slug="lanxianghu_villa_estimated",
        # 原点按高德 z=17 卫星瓦片和 100 m 比例尺估算为场景西南角。
        origin_gcj02=(121.4636, 31.0172),
        origin_description="estimated southwest corner; GCJ-02; not surveyed",
        bounds=(0.0, 0.0, 760.0, 620.0),
        max_altitude=30.0,
        buildings=tuple(buildings),
        source_urls=(
            "https://ditu.amap.com/place/B0KDPC7Y95",
            "https://sh.fang.anjuke.com/loupan/522629.html",
        ),
        notes=(
            "Buildings follow the east/west curved-row topology visible in Amap satellite imagery.",
            "Five PDF footprint classes are cycled across rows; every building height is assumed 10 m.",
            "Gaps of 5 m or less should be closed as no-fly corridors when integrating with a planner.",
        ),
    )


SCENE = build_scene()


if __name__ == "__main__":
    output = Path("outputs/standalone_maps")
    save_scene(SCENE, output)
    render_scene(SCENE, output / f"{SCENE.slug}.png")
