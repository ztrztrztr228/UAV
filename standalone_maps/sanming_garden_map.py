"""沧源路755弄三明花园估算地图，可经场景适配器接入训练环境。"""

from __future__ import annotations

from pathlib import Path

from standalone_maps.geometry import Building, SceneMap, render_scene, save_scene


TEMPLATES = {
    "single": (23.0, 10.0),
    "double": (30.0, 12.0),
    "triple": (46.0, 12.0),
    "quad": (56.0, 15.0),
}


def _row(
    y: float,
    groups: tuple[tuple[str, str], ...],
    gap: float = 20.0,
    yaw_deg: float = 0.0,
) -> list[Building]:
    lengths = [TEMPLATES[kind][0] for _, kind in groups]
    total = sum(lengths) + gap * (len(groups) - 1)
    cursor = (520.0 - total) / 2.0
    result: list[Building] = []
    for (building_id, kind), length in zip(groups, lengths):
        width = TEMPLATES[kind][1]
        result.append(
            Building(
                building_id=building_id,
                center_x=cursor + length / 2.0,
                center_y=y,
                length=length,
                width=width,
                height=20.0,
                yaw_deg=yaw_deg,
                confidence="medium",
                source_note="Amap satellite/block numbers + project PDF group dimensions",
            )
        )
        cursor += length + gap
    return result


def build_scene() -> SceneMap:
    buildings: list[Building] = []
    buildings.extend(
        _row(
            275.0,
            (("20-24", "triple"), ("46-54", "quad"), ("66-68", "double"),
             ("78-82", "triple"), ("98-100", "double"), ("120-124", "triple")),
            gap=18.0,
            yaw_deg=-1.0,
        )
    )
    buildings.extend(
        _row(
            225.0,
            (("18", "single"), ("40-46", "quad"), ("62-64", "double"),
             ("72-76", "triple"), ("94-96", "double"), ("114-118", "triple")),
            gap=20.0,
        )
    )
    buildings.extend(
        _row(
            170.0,
            (("12", "single"), ("56", "single"), ("58-60", "double"),
             ("88-90", "double"), ("110-112", "double")),
            gap=28.0,
            yaw_deg=1.0,
        )
    )
    buildings.extend(
        _row(
            115.0,
            (("4-6", "double"), ("26-30", "triple"), ("32", "single"),
             ("34", "single"), ("36-38", "double"), ("84-86", "double"),
             ("102-106", "triple")),
            gap=17.0,
        )
    )
    buildings.extend(
        _row(
            58.0,
            (("2-10", "quad"), ("14-16", "double"), ("21-29", "quad"),
             ("31-39", "quad"), ("41-55", "quad"), ("57-61", "triple")),
            gap=19.0,
            yaw_deg=-0.8,
        )
    )

    return SceneMap(
        name="三明花园",
        slug="sanming_garden_estimated",
        origin_gcj02=(121.4264, 31.0161),
        origin_description="estimated southwest corner from Amap satellite tiles; GCJ-02",
        bounds=(0.0, 0.0, 520.0, 320.0),
        max_altitude=40.0,
        buildings=tuple(buildings),
        source_urls=(
            "https://ditu.amap.com/place/B00155QPC9",
            "https://www.jia.com/zxq/shanghai/lp-18857/xqcs/",
        ),
        notes=(
            "Address verified as Cangyuan Road 755 Lane, even numbers 2-124.",
            "Rows and building-number groups follow the public map; dimensions follow the project PDF templates.",
            "All buildings use a conservative assumed height of 20 m.",
        ),
    )


SCENE = build_scene()


if __name__ == "__main__":
    output = Path("outputs/standalone_maps")
    save_scene(SCENE, output)
    render_scene(SCENE, output / f"{SCENE.slug}.png")
