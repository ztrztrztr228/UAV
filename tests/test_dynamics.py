from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from uav_drl.actions import ACTION_NAMES
from uav_drl.config import UAVEnvConfig
from uav_drl.environment import UAVPathPlanningEnv
from uav_drl.validation import validate_timed_trajectory


def make_config(**overrides: object) -> UAVEnvConfig:
    values: dict[str, object] = {
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
