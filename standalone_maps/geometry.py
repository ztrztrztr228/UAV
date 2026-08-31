"""独立地图模块共用的二维轮廓、三维挤出、导出和预览工具。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


# ==================== 单栋旋转矩形建筑 ====================
@dataclass(frozen=True)
class Building:
    """带朝向的矩形建筑轮廓，坐标单位为米。"""

    building_id: str
    center_x: float
    center_y: float
    length: float
    width: float
    height: float
    yaw_deg: float = 0.0
    confidence: str = "estimated"
    source_note: str = "network_map_estimate"

    def footprint(self) -> tuple[tuple[float, float], ...]:
        """返回逆时针排列的四个二维角点。"""
        # 先构造建筑自身坐标系中的四角，再按 yaw 旋转并平移到场景坐标。
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        angle = math.radians(self.yaw_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        corners: list[tuple[float, float]] = []
        for local_x, local_y in (
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        ):
            x = self.center_x + local_x * cos_a - local_y * sin_a
            y = self.center_y + local_x * sin_a + local_y * cos_a
            corners.append((x, y))
        return tuple(corners)

    def to_geojson_feature(self) -> dict[str, object]:
        ring = [list(point) for point in self.footprint()]
        ring.append(ring[0])
        return {
            "type": "Feature",
            "properties": {
                "building_id": self.building_id,
                "height_m": self.height,
                "yaw_deg": self.yaw_deg,
                "confidence": self.confidence,
                "source_note": self.source_note,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        }


# ==================== 完整局部 ENU 场景 ====================
@dataclass(frozen=True)
class SceneMap:
    """一个可独立检查和导出的局部 ENU 住宅区地图。"""

    name: str
    slug: str
    origin_gcj02: tuple[float, float]
    origin_description: str
    bounds: tuple[float, float, float, float]
    max_altitude: float
    buildings: tuple[Building, ...]
    source_urls: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def validate(self) -> None:
        # 依次检查地图边界、建筑编号唯一性、尺寸和所有角点是否在场景内。
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin < xmax and ymin < ymax and self.max_altitude > 0.0):
            raise ValueError(f"Invalid scene bounds for {self.slug}.")
        if not self.buildings:
            raise ValueError(f"Scene {self.slug} has no buildings.")
        seen: set[str] = set()
        for building in self.buildings:
            if building.building_id in seen:
                raise ValueError(f"Duplicate building id: {building.building_id}")
            seen.add(building.building_id)
            if min(building.length, building.width, building.height) <= 0.0:
                raise ValueError(f"Non-positive dimension: {building.building_id}")
            if building.height >= self.max_altitude:
                raise ValueError(f"Building reaches scene ceiling: {building.building_id}")
            for x, y in building.footprint():
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    raise ValueError(f"Building outside map bounds: {building.building_id}")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        data = asdict(self)
        data["coordinate_system"] = "local_enu_meters"
        data["data_quality"] = "network_map_estimate_not_for_real_flight"
        return data

    def to_geojson(self) -> dict[str, object]:
        self.validate()
        return {
            "type": "FeatureCollection",
            "name": self.slug,
            "coordinate_system": "local_enu_meters",
            "origin_gcj02": list(self.origin_gcj02),
            "features": [building.to_geojson_feature() for building in self.buildings],
        }


# ==================== 地图数据导出 ====================
def save_scene(scene: SceneMap, output_dir: Path) -> tuple[Path, Path]:
    """保存场景参数 JSON 和局部坐标 GeoJSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{scene.slug}.json"
    geojson_path = output_dir / f"{scene.slug}.geojson"
    json_path.write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    geojson_path.write_text(json.dumps(scene.to_geojson(), ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, geojson_path


# ==================== 地图二维/三维预览 ====================
def render_scene(scene: SceneMap, output_path: Path) -> None:
    """生成二维轮廓和三维挤出并列预览图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scene.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 6.5))
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")
    xmin, ymin, xmax, ymax = scene.bounds

    # 左图绘制旋转平面轮廓，右图把同一轮廓按建筑高度向上挤出。
    for index, building in enumerate(scene.buildings):
        footprint = building.footprint()
        color = "#64748b" if building.confidence == "medium" else "#94a3b8"
        ax2d.add_patch(Polygon(footprint, closed=True, facecolor=color, edgecolor="#1e293b", alpha=0.78))
        if len(scene.buildings) <= 40:
            ax2d.text(building.center_x, building.center_y, building.building_id, fontsize=5, ha="center")

        bottom = [(x, y, 0.0) for x, y in footprint]
        top = [(x, y, building.height) for x, y in footprint]
        faces: list[Sequence[tuple[float, float, float]]] = [bottom, top]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        ax3d.add_collection3d(
            Poly3DCollection(faces, facecolor=color, edgecolor="#334155", linewidths=0.3, alpha=0.68)
        )

    ax2d.set_title(f"{scene.slug} - local footprint estimate")
    ax2d.set_xlim(xmin, xmax)
    ax2d.set_ylim(ymin, ymax)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x east (m)")
    ax2d.set_ylabel("y north (m)")
    ax2d.grid(alpha=0.2)

    ax3d.set_title("3D extrusion")
    ax3d.set_xlim(xmin, xmax)
    ax3d.set_ylim(ymin, ymax)
    ax3d.set_zlim(0.0, scene.max_altitude)
    ax3d.set_xlabel("x east (m)")
    ax3d.set_ylabel("y north (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.view_init(elev=28, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
