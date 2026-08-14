"""导出并预览三个独立住宅区地图。"""

from pathlib import Path

from standalone_maps.geometry import render_scene, save_scene
from standalone_maps.lanxianghu_villa_map import SCENE as VILLA_SCENE
from standalone_maps.sanming_garden_map import SCENE as SANMING_SCENE
from standalone_maps.spring_garden_phase2_map import SCENE as SPRING_SCENE


def main() -> None:
    output = Path("outputs/standalone_maps")
    for scene in (VILLA_SCENE, SANMING_SCENE, SPRING_SCENE):
        json_path, geojson_path = save_scene(scene, output)
        preview_path = output / f"{scene.slug}.png"
        render_scene(scene, preview_path)
        print(f"{scene.name}: {len(scene.buildings)} buildings")
        print(f"  {json_path}")
        print(f"  {geojson_path}")
        print(f"  {preview_path}")


if __name__ == "__main__":
    main()
