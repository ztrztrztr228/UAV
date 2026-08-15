from __future__ import annotations

import unittest

import numpy as np

from uav_drl.environment import UAVPathPlanningEnv
from uav_drl.scenes import available_scene_keys, get_training_scene


EXPECTED_OBSTACLE_COUNTS = {
    "wujing_airfield": 10,
    "lanxianghu_villa": 115,
    "sanming_garden": 30,
    "spring_garden_phase2": 20,
}


class TrainingSceneTests(unittest.TestCase):
    def test_all_four_scenes_build_training_configs(self) -> None:
        self.assertEqual(set(available_scene_keys()), set(EXPECTED_OBSTACLE_COUNTS))
        for key, expected_count in EXPECTED_OBSTACLE_COUNTS.items():
            scene = get_training_scene(key)
            config = scene.make_config()
            self.assertEqual(config.scene_name, scene.scene_name)
            self.assertEqual(len(config.obstacles), expected_count)
            self.assertAlmostEqual(config.map_width, scene.map_width)
            self.assertAlmostEqual(config.map_height, scene.map_height)
            self.assertAlmostEqual(config.map_altitude, scene.max_altitude)

    def test_default_starts_are_inside_bounds_and_collision_free(self) -> None:
        for key in available_scene_keys():
            scene = get_training_scene(key)
            config = scene.make_config()
            env = UAVPathPlanningEnv(config=config, seed=2026)
            start = np.asarray(scene.default_start(config), dtype=np.float32)
            self.assertFalse(env._point_in_collision(start), key)

    def test_residential_rotation_is_conservatively_enclosed(self) -> None:
        for key in ("lanxianghu_villa", "sanming_garden", "spring_garden_phase2"):
            scene = get_training_scene(key)
            self.assertIsNotNone(scene.source_scene)
            obstacles = scene.obstacles(inflation=2.0)
            assert scene.source_scene is not None
            for building, obstacle in zip(scene.source_scene.buildings, obstacles):
                for x, y in building.footprint():
                    self.assertLessEqual(obstacle.xmin, x)
                    self.assertGreaterEqual(obstacle.xmax, x)
                    self.assertLessEqual(obstacle.ymin, y)
                    self.assertGreaterEqual(obstacle.ymax, y)


if __name__ == "__main__":
    unittest.main()
