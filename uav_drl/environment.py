# -*- coding: utf-8 -*-
"""包含速度和加速度状态的无人机三维动力学强化学习环境。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .actions import ACTION_DIRECTIONS, ACTION_NAMES, COAST_ACTION_INDEX
from .config import DEFAULT_SEED, UAVEnvConfig
from .trajectory import TimedTrajectory, dynamics_samples_to_trajectory


class UAVPathPlanningEnv:
    """使用点质量模型的三维轨迹规划环境。

    动作为 26 个单位方向上的最大加速度和一个零加速度动作。每一步先计算
    ``v[k+1] = clip(v[k] + a[k] * dt)``，再用梯形积分更新位置。46 维状态由
    位置、目标相对量、航向、速度、加速度、动力学裕度、雷达和时间进度组成。
    """

    def __init__(self, config: UAVEnvConfig | None = None, seed: int = DEFAULT_SEED) -> None:
        self.config = config or UAVEnvConfig()
        self._validate_config()
        self.rng = np.random.default_rng(seed)
        self.num_actions = len(ACTION_NAMES)
        self.lidar_dim = self.num_actions - 1
        self.state_dim = 3 + 3 + 1 + 3 + 3 + 3 + 3 + self.lidar_dim + 1
        self.map_size = np.asarray(
            [self.config.map_width, self.config.map_height, self.config.map_altitude],
            dtype=np.float32,
        )
        self.max_distance = float(np.linalg.norm(self.map_size))

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory: list[np.ndarray] = []
        self.velocity_trajectory: list[np.ndarray] = []
        self.acceleration_trajectory: list[np.ndarray] = []
        self.reset()

    def _validate_config(self) -> None:
        for name in ("trajectory_dt", "max_speed", "max_acceleration", "max_jerk"):
            if float(getattr(self.config, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.config.goal_speed_tolerance < 0.0:
            raise ValueError("goal_speed_tolerance must be non-negative.")

    def reset(
        self,
        start: Sequence[float] | None = None,
        goal: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = self._sample_free_point() if start is None else self._validate_free_point(start, "start")
        self.goal = self._sample_goal_far_from(self.start) if goal is None else self._validate_free_point(goal, "goal")
        self.position = self.start.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory = [self.position.copy()]
        self.velocity_trajectory = [self.velocity.copy()]
        self.acceleration_trajectory = [self.acceleration.copy()]
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._get_state(), 0.0, True, self._info("already_done")
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}], got {action}.")

        dt = self.config.trajectory_dt
        previous_position = self.position.copy()
        previous_velocity = self.velocity.copy()
        previous_acceleration = self.acceleration.copy()
        previous_distance = self.distance_to_goal()
        commanded_acceleration = ACTION_DIRECTIONS[action] * self.config.max_acceleration

        unconstrained_velocity = previous_velocity + commanded_acceleration * dt
        unconstrained_speed = float(np.linalg.norm(unconstrained_velocity))
        if unconstrained_speed > self.config.max_speed:
            next_velocity = unconstrained_velocity * (self.config.max_speed / unconstrained_speed)
        else:
            next_velocity = unconstrained_velocity
        effective_acceleration = (next_velocity - previous_velocity) / dt
        candidate = previous_position + 0.5 * (previous_velocity + next_velocity) * dt

        self.steps += 1
        if self._dynamics_segment_in_collision(
            previous_position,
            previous_velocity,
            effective_acceleration,
            dt,
        ):
            self.done = True
            reward = self.config.collision_penalty
            reward -= self.config.distance_penalty_scale * (previous_distance / self.max_distance)
            return self._get_state(), float(reward), True, self._info(
                "collision",
                collision=True,
                reward=reward,
                commanded_acceleration=commanded_acceleration,
                speed_clipped=unconstrained_speed > self.config.max_speed,
            )

        self.position = candidate.astype(np.float32)
        self.velocity = next_velocity.astype(np.float32)
        self.acceleration = effective_acceleration.astype(np.float32)
        segment_length = float(np.linalg.norm(self.position - previous_position))
        self.path_length += segment_length
        self.trajectory.append(self.position.copy())
        self.velocity_trajectory.append(self.velocity.copy())
        self.acceleration_trajectory.append(self.acceleration.copy())

        distance = self.distance_to_goal()
        progress = previous_distance - distance
        reward = self._shaped_reward(
            action,
            progress,
            distance,
            previous_acceleration,
            max(0.0, unconstrained_speed - self.config.max_speed),
        )
        speed = float(np.linalg.norm(self.velocity))
        reached_goal = distance <= self.config.goal_radius and speed <= self.config.goal_speed_tolerance
        timeout = self.steps >= self.config.max_steps
        event = "running"
        if reached_goal:
            self.done = True
            event = "goal"
            time_bonus = 1.0 - self.steps / max(1, self.config.max_steps)
            reward += self.config.goal_reward + 25.0 * max(0.0, time_bonus)
        elif timeout:
            self.done = True
            event = "timeout"
            reward += self.config.timeout_penalty

        return self._get_state(), float(reward), bool(self.done), self._info(
            event,
            success=reached_goal,
            reward=reward,
            progress=progress,
            commanded_acceleration=commanded_acceleration,
            speed_clipped=unconstrained_speed > self.config.max_speed,
        )

    def timed_trajectory(self) -> TimedTrajectory:
        """返回 RL 动力学积分直接产生的等时间轨迹点。"""
        return dynamics_samples_to_trajectory(
            self.trajectory,
            self.velocity_trajectory,
            self.acceleration_trajectory,
            self.config.trajectory_dt,
        )

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _shaped_reward(
        self,
        action: int,
        progress: float,
        distance: float,
        previous_acceleration: np.ndarray,
        clipped_speed: float,
    ) -> float:
        reward = self.config.progress_reward_scale * progress
        reward -= self.config.distance_penalty_scale * (distance / self.max_distance)
        reward -= self.config.step_penalty

        speed = float(np.linalg.norm(self.velocity))
        acceleration_norm = float(np.linalg.norm(self.acceleration))
        jerk = float(np.linalg.norm(self.acceleration - previous_acceleration)) / self.config.trajectory_dt
        reward -= self.config.acceleration_penalty_scale * acceleration_norm / self.config.max_acceleration
        reward -= self.config.jerk_penalty_scale * min(2.0, jerk / self.config.max_jerk)
        reward -= self.config.speed_penalty_scale * (speed / self.config.max_speed) ** 2
        reward -= self.config.speed_clip_penalty_scale * clipped_speed / self.config.max_speed

        delta = self.goal - self.position
        if speed > 1e-6 and np.linalg.norm(delta) > 1e-6:
            alignment = float(np.dot(self.velocity / speed, delta / np.linalg.norm(delta)))
            reward += self.config.velocity_alignment_reward_scale * alignment
        if action == COAST_ACTION_INDEX and speed < 1e-3:
            reward -= self.config.hover_penalty

        clearance = self._nearest_clearance(self.position)
        if clearance < self.config.safety_radius:
            unsafe_ratio = (self.config.safety_radius - clearance) / self.config.safety_radius
            reward -= self.config.proximity_penalty_scale * unsafe_ratio
        stopping_distance = speed * speed / (2.0 * self.config.max_acceleration)
        if stopping_distance > clearance:
            risk = min(2.0, (stopping_distance - clearance) / self.config.safety_radius)
            reward -= self.config.braking_risk_penalty_scale * risk

        altitude_ratio = self.position[2] / self.config.map_altitude
        reward -= self.config.altitude_penalty_scale * float(altitude_ratio)
        return float(reward)

    def _get_state(self) -> np.ndarray:
        delta = self.goal - self.position
        distance = float(np.linalg.norm(delta))
        heading = delta / distance if distance > 1e-9 else np.zeros(3, dtype=np.float32)
        speed_ratio = float(np.linalg.norm(self.velocity)) / self.config.max_speed
        acceleration_ratio = float(np.linalg.norm(self.acceleration)) / self.config.max_acceleration
        stopping_distance = float(np.linalg.norm(self.velocity)) ** 2 / (2.0 * self.config.max_acceleration)
        clearance = self._nearest_clearance(self.position)
        braking_margin = np.clip((clearance - stopping_distance) / self.config.lidar_range, -1.0, 1.0)
        state = np.concatenate(
            [
                self.position / self.map_size,
                delta / self.map_size,
                np.asarray([distance / self.max_distance], dtype=np.float32),
                heading.astype(np.float32),
                self.velocity / self.config.max_speed,
                self.acceleration / self.config.max_acceleration,
                np.asarray([speed_ratio, acceleration_ratio, braking_margin], dtype=np.float32),
                self._lidar_scan(),
                np.asarray([self.steps / max(1, self.config.max_steps)], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def _lidar_scan(self) -> np.ndarray:
        readings: list[float] = []
        for direction in ACTION_DIRECTIONS[:-1]:
            distance = self.config.lidar_range
            sample_count = max(2, int(math.ceil(distance / self.config.lidar_resolution)))
            for i in range(1, sample_count + 1):
                probe_distance = min(distance, i * self.config.lidar_resolution)
                if self._point_in_collision(self.position + direction * probe_distance):
                    distance = probe_distance
                    break
            readings.append(distance / self.config.lidar_range)
        return np.asarray(readings, dtype=np.float32)

    def _sample_goal_far_from(self, start: np.ndarray) -> np.ndarray:
        for _ in range(10_000):
            point = self._sample_free_point()
            if np.linalg.norm(point - start) >= self.config.min_start_goal_distance:
                return point
        raise RuntimeError("Could not sample a valid goal far from start.")

    def _sample_free_point(self) -> np.ndarray:
        low = np.full(3, self.config.uav_radius, dtype=np.float32)
        high = self.map_size - self.config.uav_radius
        for _ in range(10_000):
            point = self.rng.uniform(low, high).astype(np.float32)
            if not self._point_in_collision(point):
                return point
        raise RuntimeError("Could not sample a valid free point.")

    def _validate_free_point(self, point: Sequence[float], name: str) -> np.ndarray:
        array = np.asarray(point, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"{name} must have three values x/y/z, got {point}.")
        if self._point_in_collision(array):
            raise ValueError(f"{name} point {array.tolist()} is outside the map or inside an obstacle.")
        return array

    def _segment_in_collision(self, start: np.ndarray, end: np.ndarray) -> bool:
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / self.config.collision_resolution)) + 1)
        return any(self._point_in_collision(start + t * (end - start)) for t in np.linspace(0.0, 1.0, count))

    def _dynamics_segment_in_collision(
        self,
        start: np.ndarray,
        initial_velocity: np.ndarray,
        acceleration: np.ndarray,
        duration: float,
    ) -> bool:
        """沿常加速度抛物线采样，避免只检查端点连线漏掉碰撞。"""
        end_speed = float(np.linalg.norm(initial_velocity + acceleration * duration))
        estimated_length = max(float(np.linalg.norm(initial_velocity)), end_speed) * duration
        count = max(2, int(math.ceil(estimated_length / self.config.collision_resolution)) + 1)
        for t in np.linspace(0.0, duration, count):
            point = start + initial_velocity * t + 0.5 * acceleration * t * t
            if self._point_in_collision(point):
                return True
        return False

    def _point_in_collision(self, point: np.ndarray) -> bool:
        x, y, z = map(float, point)
        radius = self.config.uav_radius
        if not (radius <= x <= self.config.map_width - radius):
            return True
        if not (radius <= y <= self.config.map_height - radius):
            return True
        if not (radius <= z <= self.config.map_altitude - radius):
            return True
        return any(obstacle.contains(point, margin=radius) for obstacle in self.config.obstacles)

    def _nearest_clearance(self, point: np.ndarray) -> float:
        x, y, z = map(float, point)
        boundary_clearance = min(
            x, y, z,
            self.config.map_width - x,
            self.config.map_height - y,
            self.config.map_altitude - z,
        )
        obstacle_clearance = min(
            (obstacle.distance_to(point) for obstacle in self.config.obstacles),
            default=float("inf"),
        )
        return max(0.0, min(boundary_clearance, obstacle_clearance) - self.config.uav_radius)

    def _info(
        self,
        event: str,
        success: bool = False,
        collision: bool = False,
        reward: float = 0.0,
        progress: float = 0.0,
        commanded_acceleration: np.ndarray | None = None,
        speed_clipped: bool = False,
    ) -> dict[str, object]:
        return {
            "event": event,
            "success": bool(success),
            "collision": bool(collision),
            "distance_to_goal": self.distance_to_goal(),
            "clearance": self._nearest_clearance(self.position),
            "path_length": self.path_length,
            "steps": self.steps,
            "time": self.steps * self.config.trajectory_dt,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "speed": float(np.linalg.norm(self.velocity)),
            "acceleration": self.acceleration.copy(),
            "acceleration_norm": float(np.linalg.norm(self.acceleration)),
            "commanded_acceleration": None if commanded_acceleration is None else commanded_acceleration.copy(),
            "speed_clipped": bool(speed_clipped),
            "goal": self.goal.copy(),
            "reward": float(reward),
            "progress": float(progress),
        }
