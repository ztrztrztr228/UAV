from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from uav_drl.actions import ACTION_NAMES
from uav_drl.config import BoxObstacle, UAVEnvConfig, wujing_airfield_obstacles
from uav_drl.environment import UAVPathPlanningEnv
from uav_drl.validation import validate_timed_trajectory


def make_config(**overrides: object) -> UAVEnvConfig:
    values: dict[str, object] = {
        "map_x_min": 0.0,
        "map_y_min": 0.0,
        "map_width": 40.0,
        "map_height": 40.0,
        "map_altitude": 30.0,
        "obstacles": [],
        "trajectory_dt": 0.5,
        "max_speed": 2.0,
        "max_acceleration": 2.0,
        "max_jerk": 4.0,
        "goal_radius": 0.05,
        "goal_speed_tolerance": 2.0,
        "max_steps": 20,
    }
    values.update(overrides)
    return UAVEnvConfig(**values)


class DynamicsEnvironmentTests(unittest.TestCase):
    def test_measured_default_dynamics_parameters(self) -> None:
        config = UAVEnvConfig()
        self.assertAlmostEqual(config.max_horizontal_speed, 23.0)
        self.assertAlmostEqual(config.max_speed, 23.18)
        self.assertAlmostEqual(config.max_climb_speed, 2.85)
        self.assertAlmostEqual(config.max_descent_speed, 1.65)
        self.assertAlmostEqual(config.max_climb_angle_deg, 90.0)
        self.assertAlmostEqual(config.climb_angle_at_max_horizontal_speed_deg, 7.1, places=1)
        self.assertAlmostEqual(config.max_acceleration, 15.5)
        self.assertAlmostEqual(config.normal_acceleration, 3.0)
        self.assertAlmostEqual(config.max_deceleration, 3.09)
        self.assertAlmostEqual(config.max_jerk, 78.0)
        self.assertAlmostEqual(config.raw_max_jerk, 142.0)
        self.assertAlmostEqual(config.lidar_range, 40.0)
        self.assertEqual(config.reward_shaping_version, 3)

    def test_velocity_projection_applies_horizontal_vertical_and_combined_limits(self) -> None:
        env = UAVPathPlanningEnv(UAVEnvConfig(obstacles=[]))
        ascent = env._limit_velocity(np.asarray([30.0, 0.0, 10.0]))
        descent = env._limit_velocity(np.asarray([0.0, 0.0, -10.0]))
        self.assertLessEqual(float(np.linalg.norm(ascent[:2])), 23.0 + 1e-6)
        self.assertLessEqual(float(ascent[2]), 2.85 + 1e-6)
        self.assertLessEqual(float(np.linalg.norm(ascent)), 23.18 + 1e-6)
        self.assertGreaterEqual(float(descent[2]), -1.65 - 1e-6)

    def test_evaluation_goal_can_be_sampled_near_a_building(self) -> None:
        obstacle = BoxObstacle(18.0, 8.0, 0.0, 22.0, 12.0, 15.0, "test_building")
        env = UAVPathPlanningEnv(
            make_config(obstacles=[obstacle], min_start_goal_distance=10.0)
        )
        env.reset(
            start=(5, 5, 5),
            seed=123,
            goal_near_obstacle_probability=1.0,
            goal_near_obstacle_min_clearance=2.0,
            goal_near_obstacle_max_clearance=8.0,
        )
        self.assertEqual(env.goal_sampling_mode, "near_obstacle")
        self.assertGreaterEqual(env.goal_obstacle_clearance, 2.0)
        self.assertLessEqual(env.goal_obstacle_clearance, 8.0)
        self.assertFalse(env._point_in_collision(env.goal))

    def test_residential_curriculum_limits_distance_altitude_and_building_side(self) -> None:
        obstacle = BoxObstacle(18.0, 8.0, 0.0, 22.0, 12.0, 15.0, "test_building")
        env = UAVPathPlanningEnv(
            make_config(obstacles=[obstacle], min_start_goal_distance=10.0)
        )
        start = np.asarray([5.0, 5.0, 5.0], dtype=np.float32)
        env.reset(
            start=start,
            seed=321,
            goal_near_obstacle_probability=1.0,
            goal_near_obstacle_min_clearance=2.0,
            goal_near_obstacle_max_clearance=8.0,
            goal_max_start_distance=25.0,
            goal_altitude_min=5.0,
            goal_altitude_max=10.0,
            goal_near_obstacle_horizontal_only=True,
        )

        horizontal_clearance = env._nearest_obstacle_horizontal_clearance(env.goal)
        self.assertEqual(env.goal_sampling_mode, "near_obstacle")
        self.assertLessEqual(float(np.linalg.norm(env.goal - start)), 25.0)
        self.assertGreaterEqual(float(env.goal[2]), 5.0)
        self.assertLessEqual(float(env.goal[2]), 10.0)
        self.assertGreaterEqual(horizontal_clearance, 2.0)
        self.assertLessEqual(horizontal_clearance, 8.0)

    def test_vectorized_collision_check_matches_scalar_definition(self) -> None:
        obstacle = BoxObstacle(18.0, 8.0, 0.0, 22.0, 12.0, 15.0, "test_building")
        env = UAVPathPlanningEnv(make_config(obstacles=[obstacle], uav_radius=0.5))
        points = np.asarray(
            [
                [10.0, 10.0, 10.0],
                [18.0, 10.0, 5.0],
                [22.4, 12.4, 15.4],
                [22.6, 12.6, 15.6],
                [0.4, 10.0, 10.0],
                [39.5, 20.0, 20.0],
            ],
            dtype=np.float32,
        )
        expected = []
        for point in points:
            outside_flyable_bounds = bool(
                np.any(point < env.map_min + env.config.uav_radius)
                or np.any(point > env.map_max - env.config.uav_radius)
            )
            expected.append(
                outside_flyable_bounds
                or obstacle.contains(point, margin=env.config.uav_radius)
            )

        self.assertEqual(env._points_in_collision(points).tolist(), expected)

    def test_near_obstacle_goal_sampling_arguments_are_validated(self) -> None:
        env = UAVPathPlanningEnv(make_config())
        with self.assertRaises(ValueError):
            env.reset(goal_near_obstacle_probability=1.1)
        with self.assertRaises(ValueError):
            env.reset(
                goal_near_obstacle_probability=1.0,
                goal_near_obstacle_min_clearance=10.0,
                goal_near_obstacle_max_clearance=2.0,
            )

    def test_normalized_reward_has_only_distinct_dense_components(self) -> None:
        env = UAVPathPlanningEnv(make_config(goal_radius=0.01, goal_speed_tolerance=0.0))
        env.reset(start=(10, 10, 10), goal=(30, 10, 15))

        east = ACTION_NAMES.index("accelerate_east")
        _, _, _, info = env.step(east)
        components = info["reward_components"]
        self.assertEqual(set(components), {"progress", "step", "jerk", "safety_risk", "total"})
        self.assertGreater(components["progress"], 0.0)
        self.assertLessEqual(abs(components["progress"]), env.config.progress_reward_scale)

        env.position = np.asarray([1.0, 10.0, 10.0], dtype=np.float32)
        env.velocity = np.asarray([env.config.max_speed, 0.0, 0.0], dtype=np.float32)
        env.acceleration = np.zeros(3, dtype=np.float32)
        env._shaped_reward(
            progress=100.0,
            previous_acceleration=np.zeros(3, dtype=np.float32),
        )
        self.assertAlmostEqual(env.last_reward_components["progress"], 1.0)
        self.assertLess(env.last_reward_components["safety_risk"], 0.0)
        self.assertGreater(env.last_reward_diagnostics["safety_risk"], 0.0)

    def test_collision_penalty_dominates_one_step_progress(self) -> None:
        env = UAVPathPlanningEnv(make_config())
        env.reset(start=(0.71, 10.0, 10.0), goal=(30.0, 10.0, 10.0))
        _, reward, done, info = env.step(ACTION_NAMES.index("accelerate_west"))

        self.assertTrue(done)
        self.assertEqual(info["event"], "collision")
        self.assertEqual(reward, env.config.collision_penalty)
        self.assertGreater(abs(reward), env.config.progress_reward_scale)

    def test_wujing_building_estimates_are_inflated_and_height_assumptions_applied(self) -> None:
        obstacles = wujing_airfield_obstacles(inflation=8.0)
        self.assertEqual(len(obstacles), 10)
        self.assertAlmostEqual(obstacles[0].xmin, 34.85)
        self.assertAlmostEqual(obstacles[0].ymax, 257.8)
        self.assertEqual(obstacles[0].zmax, 15.0)
        self.assertAlmostEqual(obstacles[-1].ymin, -15.15)
        self.assertEqual(obstacles[-1].zmax, 25.0)

    def test_negative_map_origin_is_supported(self) -> None:
        env = UAVPathPlanningEnv(make_config(map_y_min=-20.0))
        state = env.reset(start=(10, -10, 10), goal=(30, -10, 10))
        self.assertAlmostEqual(float(state[1]), 0.25, places=6)
        self.assertFalse(env._point_in_collision(np.asarray([10.0, -10.0, 10.0])))

    def test_state_and_action_include_dynamics(self) -> None:
        env = UAVPathPlanningEnv(make_config())
        state = env.reset(start=(10, 10, 10), goal=(30, 10, 10))
        self.assertEqual(env.state_dim, 50)
        self.assertEqual(state.shape, (50,))
        self.assertTrue(np.all(np.isfinite(state[-4:])))
        self.assertIn("accelerate_east", ACTION_NAMES)
        self.assertEqual(ACTION_NAMES[-1], "coast")

    def test_safe_action_mask_rejects_acceleration_toward_close_building(self) -> None:
        obstacle = BoxObstacle(18.0, 8.0, 0.0, 22.0, 12.0, 15.0, "test_building")
        env = UAVPathPlanningEnv(make_config(obstacles=[obstacle], max_speed=10.0))
        env.reset(start=(14, 10, 5), goal=(30, 10, 5))
        env.velocity = np.asarray([5.0, 0.0, 0.0], dtype=np.float32)
        env.acceleration = np.zeros(3, dtype=np.float32)

        mask = env.safe_action_mask()

        self.assertEqual(mask.shape, (env.num_actions,))
        self.assertTrue(np.any(mask))
        self.assertFalse(mask[ACTION_NAMES.index("accelerate_east")])
        self.assertTrue(mask[ACTION_NAMES.index("accelerate_west")])

    def test_velocity_is_integrated_and_capped(self) -> None:
        env = UAVPathPlanningEnv(make_config(goal_radius=0.01, goal_speed_tolerance=0.0))
        env.reset(start=(10, 10, 10), goal=(30, 10, 10))
        east = ACTION_NAMES.index("accelerate_east")
        for _ in range(5):
            _, _, done, _ = env.step(east)
            self.assertFalse(done)
        self.assertLessEqual(np.linalg.norm(env.velocity), env.config.max_speed + 1e-6)
        self.assertLessEqual(np.linalg.norm(env.acceleration), env.config.max_acceleration + 1e-6)
        self.assertEqual(len(env.trajectory), len(env.velocity_trajectory))
        self.assertEqual(len(env.trajectory), len(env.acceleration_trajectory))

    def test_goal_requires_low_enough_terminal_speed(self) -> None:
        config = make_config(goal_speed_tolerance=0.1)
        env = UAVPathPlanningEnv(config)
        env.reset(start=(10, 10, 10), goal=(10.25, 10, 10))
        east = ACTION_NAMES.index("accelerate_east")
        _, _, done, info = env.step(east)
        self.assertFalse(done)
        self.assertEqual(info["event"], "running")
        self.assertAlmostEqual(float(info["speed"]), 1.0, places=6)

    def test_direct_rl_trajectory_passes_full_validation(self) -> None:
        env = UAVPathPlanningEnv(make_config())
        env.reset(start=(10, 10, 10), goal=(10.25, 10, 10))
        env.step(ACTION_NAMES.index("accelerate_east"))
        trajectory = env.timed_trajectory()
        result = validate_timed_trajectory(
            env,
            env.trajectory,
            trajectory,
            deviation_tolerance=1e-8,
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.dynamics_consistent)
        self.assertTrue(result.goal_reached)
        self.assertAlmostEqual(result.max_jerk, 4.0, places=6)

        invalid = replace(
            trajectory,
            acceleration=trajectory.acceleration * 2.0,
            acceleration_norm=trajectory.acceleration_norm * 2.0,
            max_acceleration=trajectory.max_acceleration * 2.0,
        )
        invalid_result = validate_timed_trajectory(
            env,
            env.trajectory,
            invalid,
            deviation_tolerance=1e-8,
        )
        self.assertFalse(invalid_result.passed)
        self.assertFalse(invalid_result.acceleration_limit_satisfied)


if __name__ == "__main__":
    unittest.main()
