from __future__ import annotations

import unittest

from standalone_maps.lanxianghu_villa_map import SCENE as VILLA_SCENE
from standalone_maps.sanming_garden_map import SCENE as SANMING_SCENE
from standalone_maps.spring_garden_phase2_map import SCENE as SPRING_SCENE


class StandaloneMapTests(unittest.TestCase):
    def test_all_scenes_are_valid(self) -> None:
        for scene in (VILLA_SCENE, SANMING_SCENE, SPRING_SCENE):
            scene.validate()
            self.assertEqual(len(scene.to_geojson()["features"]), len(scene.buildings))

    def test_expected_building_counts(self) -> None:
        self.assertEqual(len(VILLA_SCENE.buildings), 115)
        self.assertEqual(len(SANMING_SCENE.buildings), 30)
        self.assertEqual(len(SPRING_SCENE.buildings), 20)

    def test_height_assumptions_are_scene_specific(self) -> None:
        self.assertEqual({building.height for building in VILLA_SCENE.buildings}, {10.0})
        self.assertEqual({building.height for building in SANMING_SCENE.buildings}, {20.0})
        self.assertEqual({building.height for building in SPRING_SCENE.buildings}, {60.0})


if __name__ == "__main__":
    unittest.main()
