from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from uav_drl.actions import ACTION_NAMES
from uav_drl.config import UAVEnvConfig, wujing_airfield_obstacles
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

    def test_velocity_projection_applies_horizontal_vertical_and_combined_limits(self) -> None:
        env = UAVPathPlanningEnv(UAVEnvConfig(obstacles=[]))
        ascent = env._limit_velocity(np.asarray([30.0, 0.0, 10.0]))
        descent = env._limit_velocity(np.asarray([0.0, 0.0, -10.0]))
        self.assertLessEqual(float(np.linalg.norm(ascent[:2])), 23.0 + 1e-6)
        self.assertLessEqual(float(ascent[2]), 2.85 + 1e-6)
        self.assertLessEqual(float(np.linalg.norm(ascent)), 23.18 + 1e-6)
        self.assertGreaterEqual(float(descent[2]), -1.65 - 1e-6)

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
        self.assertEqual(env.state_dim, 46)
        self.assertEqual(state.shape, (46,))
        self.assertIn("accelerate_east", ACTION_NAMES)
        self.assertEqual(ACTION_NAMES[-1], "coast")

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
