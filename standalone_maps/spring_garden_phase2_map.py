"""长宁区春天花园二期估算地图，可经场景适配器接入训练环境。"""

from __future__ import annotations

from pathlib import Path

from standalone_maps.geometry import Building, SceneMap, render_scene, save_scene


STRIP_TYPES = {
    "long": (73.0, 13.0),
    "long_wide": (71.0, 15.0),
    "medium": (63.0, 15.0),
    "short": (54.0, 12.0),
}


def _strip_row(
    y: float,
    groups: tuple[tuple[str, str], ...],
    gap: float,
    yaw_deg: float,
) -> list[Building]:
    lengths = [STRIP_TYPES[kind][0] for _, kind in groups]
    total = sum(lengths) + gap * (len(groups) - 1)
    cursor = (520.0 - total) / 2.0
    result: list[Building] = []
    for (building_id, kind), length in zip(groups, lengths):
        width = STRIP_TYPES[kind][1]
        result.append(
            Building(
                building_id=building_id,
                center_x=cursor + length / 2.0,
                center_y=y,
                length=length,
                width=width,
                height=60.0,
                yaw_deg=yaw_deg,
                confidence="low",
                source_note="Amap satellite/block numbers + project PDF strip dimensions",
            )
        )
        cursor += length + gap
    return result


def build_scene() -> SceneMap:
    buildings: list[Building] = []
    buildings.extend(_strip_row(315.0, (("73", "long"), ("76", "long_wide"), ("71", "short")), 44.0, -3.0))
    buildings.extend(_strip_row(265.0, (("69-66", "long_wide"), ("67-63", "medium")), 70.0, -1.5))
    buildings.extend(_strip_row(215.0, (("58-55", "long"), ("30-26", "medium")), 64.0, 0.5))
    buildings.extend(
        _strip_row(
            165.0,
            (("64-56", "long_wide"), ("53-47", "medium"), ("22-18", "short"), ("31-25", "short")),
            28.0,
            -0.5,
        )
    )
    buildings.extend(
        _strip_row(
            112.0,
            (("48-42", "long"), ("45-41", "medium"), ("16-10", "medium"), ("23-17", "short")),
            24.0,
            1.0,
        )
    )
    buildings.extend(
        _strip_row(
            58.0,
            (("40-34", "long"), ("39-37", "short"), ("6-2", "medium"), ("15-9", "medium"), ("7-1", "short")),
            18.0,
            0.0,
        )
    )

    return SceneMap(
        name="春天花园二期",
        slug="spring_garden_phase2_estimated",
        origin_gcj02=(121.4016, 31.2151),
        origin_description="estimated southwest corner; GCJ-02; POI center cross-checked online",
        bounds=(0.0, 0.0, 520.0, 350.0),
        max_altitude=80.0,
        buildings=tuple(buildings),
        source_urls=(
            "https://ditu.amap.com/place/B0JB31HTS6",
            "https://house.leju.com/sh/1658/xinxi/",
            "https://www.poi86.com/poi/amap2/116793246.html",
        ),
        notes=(
            "Public sources place the compound at Loushanguan Road 999 Lane in Changning District.",
            "Connected high-rise strips are modeled as single obstacles following the public map topology.",
            "The project PDF reports 50-60 m; 60 m is used conservatively because per-building heights are unavailable.",
        ),
    )


SCENE = build_scene()


if __name__ == "__main__":
    output = Path("outputs/standalone_maps")
    save_scene(SCENE, output)
    render_scene(SCENE, output / f"{SCENE.slug}.png")
