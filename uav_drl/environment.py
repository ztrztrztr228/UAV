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

    动作为 26 个单位方向上的正常飞行加速度和一个零加速度动作。每一步先计算
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
        self.map_min = np.asarray(
            [self.config.map_x_min, self.config.map_y_min, 0.0],
            dtype=np.float32,
        )
        self.map_max = self.map_min + self.map_size
        self.max_distance = float(np.linalg.norm(self.map_size))

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(3, dtype=np.float32)
        self.goal_sampling_mode = "uninitialized"
        self.goal_obstacle_clearance = float("inf")
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory: list[np.ndarray] = []
        self.velocity_trajectory: list[np.ndarray] = []
        self.acceleration_trajectory: list[np.ndarray] = []
        self.last_reward_components: dict[str, float] = {}
        self.last_reward_diagnostics: dict[str, float] = {}
        self.reset()

    def _validate_config(self) -> None:
        for name in ("map_width", "map_height", "map_altitude"):
            if float(getattr(self.config, name)) <= 2.0 * self.config.uav_radius:
                raise ValueError(f"{name} must be greater than twice the UAV radius.")
        for name in (
            "trajectory_dt",
            "max_horizontal_speed",
            "max_speed",
            "max_climb_speed",
            "max_descent_speed",
            "max_acceleration",
            "normal_acceleration",
            "max_deceleration",
            "max_jerk",
            "raw_max_jerk",
        ):
            if float(getattr(self.config, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.config.max_climb_angle_deg <= 90.0:
            raise ValueError("max_climb_angle_deg must be in (0, 90].")
        if self.config.goal_speed_tolerance < 0.0:
            raise ValueError("goal_speed_tolerance must be non-negative.")
        if self.config.reward_shaping_version < 1:
            raise ValueError("reward_shaping_version must be positive.")
        non_negative_names = (
            "turn_penalty_scale",
            "turn_speed_threshold",
            "detour_penalty_scale",
            "altitude_penalty_scale",
            "extra_altitude_penalty_scale",
            "extra_altitude_margin",
            "goal_altitude_penalty_scale",
            "vertical_speed_guidance_scale",
        )
        for name in non_negative_names:
            if float(getattr(self.config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.config.goal_guidance_distance <= 0.0:
            raise ValueError("goal_guidance_distance must be positive.")
        if self.config.vertical_guidance_time <= 0.0:
            raise ValueError("vertical_guidance_time must be positive.")

    def reset(
        self,
        start: Sequence[float] | None = None,
        goal: Sequence[float] | None = None,
        seed: int | None = None,
        goal_near_obstacle_probability: float = 0.0,
        goal_near_obstacle_min_clearance: float = 2.0,
        goal_near_obstacle_max_clearance: float = 12.0,
    ) -> np.ndarray:
        if not 0.0 <= goal_near_obstacle_probability <= 1.0:
            raise ValueError("goal_near_obstacle_probability must be in [0, 1].")
        if goal_near_obstacle_min_clearance < 0.0:
            raise ValueError("goal_near_obstacle_min_clearance must be non-negative.")
        if goal_near_obstacle_max_clearance < goal_near_obstacle_min_clearance:
            raise ValueError(
                "goal_near_obstacle_max_clearance must not be below the minimum clearance."
            )
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start = self._sample_free_point() if start is None else self._validate_free_point(start, "start")
        if goal is not None:
            self.goal = self._validate_free_point(goal, "goal")
            self.goal_sampling_mode = "specified"
        elif self.config.obstacles and self.rng.random() < goal_near_obstacle_probability:
            near_goal = self._sample_goal_near_obstacle(
                self.start,
                goal_near_obstacle_min_clearance,
                goal_near_obstacle_max_clearance,
            )
            if near_goal is None:
                self.goal = self._sample_goal_far_from(self.start)
                self.goal_sampling_mode = "uniform_fallback"
            else:
                self.goal = near_goal
                self.goal_sampling_mode = "near_obstacle"
        else:
            self.goal = self._sample_goal_far_from(self.start)
            self.goal_sampling_mode = "uniform"
        self.goal_obstacle_clearance = self._nearest_obstacle_clearance(self.goal)
        self.position = self.start.copy()
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory = [self.position.copy()]
        self.velocity_trajectory = [self.velocity.copy()]
        self.acceleration_trajectory = [self.acceleration.copy()]
        self.last_reward_components = {}
        self.last_reward_diagnostics = {}
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
        commanded_acceleration = self._limit_commanded_acceleration(
            ACTION_DIRECTIONS[action] * self.config.normal_acceleration,
            previous_velocity,
        )
        acceleration_delta = commanded_acceleration - previous_acceleration
        max_acceleration_delta = self.config.max_jerk * dt
        acceleration_delta_norm = float(np.linalg.norm(acceleration_delta))
        if acceleration_delta_norm > max_acceleration_delta:
            commanded_acceleration = previous_acceleration + acceleration_delta * (
                max_acceleration_delta / acceleration_delta_norm
            )

        unconstrained_velocity = previous_velocity + commanded_acceleration * dt
        next_velocity = self._limit_velocity(unconstrained_velocity)
        clipped_velocity = float(np.linalg.norm(unconstrained_velocity - next_velocity))
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
            self.last_reward_components = {
                "collision": float(self.config.collision_penalty),
                "distance": float(-self.config.distance_penalty_scale * previous_distance / self.max_distance),
                "total": float(reward),
            }
            self.last_reward_diagnostics = {}
            return self._get_state(), float(reward), True, self._info(
                "collision",
                collision=True,
                reward=reward,
                commanded_acceleration=commanded_acceleration,
                speed_clipped=clipped_velocity > 1e-9,
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
            previous_velocity,
            previous_acceleration,
            segment_length,
            clipped_velocity,
        )
        speed = float(np.linalg.norm(self.velocity))
        reached_goal = distance <= self.config.goal_radius and speed <= self.config.goal_speed_tolerance
        timeout = self.steps >= self.config.max_steps
        event = "running"
        if reached_goal:
            self.done = True
            event = "goal"
            time_bonus = 1.0 - self.steps / max(1, self.config.max_steps)
            goal_bonus = self.config.goal_reward + 25.0 * max(0.0, time_bonus)
            reward += goal_bonus
            self.last_reward_components["goal"] = float(goal_bonus)
        elif timeout:
            self.done = True
            event = "timeout"
            reward += self.config.timeout_penalty
            self.last_reward_components["timeout"] = float(self.config.timeout_penalty)
        self.last_reward_components["total"] = float(reward)

        return self._get_state(), float(reward), bool(self.done), self._info(
            event,
            success=reached_goal,
            reward=reward,
            progress=progress,
            commanded_acceleration=commanded_acceleration,
            speed_clipped=clipped_velocity > 1e-9,
        )

    def _limit_commanded_acceleration(
        self,
        acceleration: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Apply normal-flight, peak, and measured deceleration limits."""
        limit = min(self.config.normal_acceleration, self.config.max_acceleration)
        if np.linalg.norm(velocity) > 1e-9 and float(np.dot(acceleration, velocity)) < 0.0:
            limit = min(limit, self.config.max_deceleration)
        norm = float(np.linalg.norm(acceleration))
        if norm > limit:
            return acceleration * (limit / norm)
        return acceleration

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Project velocity onto horizontal, vertical, climb-angle, and 3D limits."""
        limited = np.asarray(velocity, dtype=np.float64).copy()
        horizontal_speed = float(np.linalg.norm(limited[:2]))
        if horizontal_speed > self.config.max_horizontal_speed:
            limited[:2] *= self.config.max_horizontal_speed / horizontal_speed
            horizontal_speed = self.config.max_horizontal_speed

        limited[2] = np.clip(
            limited[2],
            -self.config.max_descent_speed,
            self.config.max_climb_speed,
        )
        if limited[2] > 0.0 and self.config.max_climb_angle_deg < 90.0:
            angle_vertical_limit = horizontal_speed * math.tan(
                math.radians(self.config.max_climb_angle_deg)
            )
            limited[2] = min(limited[2], angle_vertical_limit)

        speed = float(np.linalg.norm(limited))
        if speed > self.config.max_speed:
            limited *= self.config.max_speed / speed
        return limited.astype(np.float32)

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

    def _reference_route_altitude(self, position: np.ndarray) -> float:
        """Return linearly interpolated start-to-goal altitude at the current XY projection."""
        horizontal_route = self.goal[:2] - self.start[:2]
        route_length_sq = float(np.dot(horizontal_route, horizontal_route))
        if route_length_sq <= 1e-9:
            return float(max(self.start[2], self.goal[2]))
        route_ratio = float(
            np.clip(
                np.dot(position[:2] - self.start[:2], horizontal_route) / route_length_sq,
                0.0,
                1.0,
            )
        )
        return float(self.start[2] + route_ratio * (self.goal[2] - self.start[2]))

    def _allowed_route_altitude(self, position: np.ndarray) -> float:
        """Return free altitude corridor including local obstacle-overflight clearance."""
        margin = self.config.extra_altitude_margin
        allowed_altitude = self._reference_route_altitude(position) + margin
        horizontal_margin = self.config.safety_radius + self.config.uav_radius
        x, y = float(position[0]), float(position[1])
        for obstacle in self.config.obstacles:
            if (
                obstacle.xmin - horizontal_margin <= x <= obstacle.xmax + horizontal_margin
                and obstacle.ymin - horizontal_margin <= y <= obstacle.ymax + horizontal_margin
            ):
                obstacle_clearance_altitude = obstacle.zmax + self.config.uav_radius + margin
                allowed_altitude = max(allowed_altitude, obstacle_clearance_altitude)
        return float(min(allowed_altitude, self.map_max[2] - self.config.uav_radius))

    def _goal_guidance_terms(self) -> tuple[float, float, float, float]:
        """Return approach weight, altitude error, desired vz, and vertical-speed error ratio."""
        horizontal_distance = float(np.linalg.norm((self.goal - self.position)[:2]))
        approach_weight = float(
            np.clip(1.0 - horizontal_distance / self.config.goal_guidance_distance, 0.0, 1.0)
        )
        altitude_error = float(self.goal[2] - self.position[2])
        desired_vertical_speed = float(
            np.clip(
                altitude_error / self.config.vertical_guidance_time,
                -self.config.max_descent_speed,
                self.config.max_climb_speed,
            )
        )
        vertical_speed_scale = max(self.config.max_climb_speed, self.config.max_descent_speed)
        vertical_speed_error_ratio = abs(float(self.velocity[2]) - desired_vertical_speed) / vertical_speed_scale
        return approach_weight, altitude_error, desired_vertical_speed, vertical_speed_error_ratio

    def _shaped_reward(
        self,
        action: int,
        progress: float,
        distance: float,
        previous_velocity: np.ndarray,
        previous_acceleration: np.ndarray,
        segment_length: float,
        clipped_speed: float,
    ) -> float:
        components: dict[str, float] = {
            "progress": float(self.config.progress_reward_scale * progress),
            "distance": float(-self.config.distance_penalty_scale * distance / self.max_distance),
            "step": float(-self.config.step_penalty),
        }

        speed = float(np.linalg.norm(self.velocity))
        acceleration_norm = float(np.linalg.norm(self.acceleration))
        jerk = float(np.linalg.norm(self.acceleration - previous_acceleration)) / self.config.trajectory_dt
        acceleration_scale = min(self.config.normal_acceleration, self.config.max_acceleration)
        components["acceleration"] = float(
            -self.config.acceleration_penalty_scale * acceleration_norm / acceleration_scale
        )
        components["jerk"] = float(
            -self.config.jerk_penalty_scale * min(2.0, jerk / self.config.max_jerk)
        )
        components["speed"] = float(-self.config.speed_penalty_scale * (speed / self.config.max_speed) ** 2)
        components["speed_clip"] = float(
            -self.config.speed_clip_penalty_scale * clipped_speed / self.config.max_speed
        )

        useful_progress = max(0.0, progress)
        detour_distance = max(0.0, segment_length - useful_progress)
        components["detour"] = float(-self.config.detour_penalty_scale * detour_distance)

        previous_speed = float(np.linalg.norm(previous_velocity))
        turn_angle_ratio = 0.0
        if min(previous_speed, speed) >= self.config.turn_speed_threshold:
            heading_cosine = float(
                np.clip(np.dot(previous_velocity, self.velocity) / (previous_speed * speed), -1.0, 1.0)
            )
            turn_angle_ratio = math.acos(heading_cosine) / math.pi
        components["turn"] = float(-self.config.turn_penalty_scale * turn_angle_ratio)

        delta = self.goal - self.position
        if speed > 1e-6 and np.linalg.norm(delta) > 1e-6:
            alignment = float(np.dot(self.velocity / speed, delta / np.linalg.norm(delta)))
            components["velocity_alignment"] = float(
                self.config.velocity_alignment_reward_scale * alignment
            )
        else:
            components["velocity_alignment"] = 0.0
        if action == COAST_ACTION_INDEX and speed < 1e-3:
            components["hover"] = float(-self.config.hover_penalty)

        clearance = self._nearest_clearance(self.position)
        if clearance < self.config.safety_radius:
            unsafe_ratio = (self.config.safety_radius - clearance) / self.config.safety_radius
            components["proximity"] = float(-self.config.proximity_penalty_scale * unsafe_ratio)
        stopping_distance = speed * speed / (2.0 * self.config.max_deceleration)
        if stopping_distance > clearance:
            risk = min(2.0, (stopping_distance - clearance) / self.config.safety_radius)
            components["braking_risk"] = float(-self.config.braking_risk_penalty_scale * risk)

        altitude_ratio = self.position[2] / self.config.map_altitude
        components["absolute_altitude"] = float(-self.config.altitude_penalty_scale * altitude_ratio)
        allowed_altitude = self._allowed_route_altitude(self.position)
        extra_altitude = max(0.0, float(self.position[2]) - allowed_altitude)
        components["extra_altitude"] = float(
            -self.config.extra_altitude_penalty_scale * extra_altitude
        )

        approach_weight, altitude_error, desired_vz, vertical_speed_error_ratio = self._goal_guidance_terms()
        components["goal_altitude"] = float(
            -self.config.goal_altitude_penalty_scale * approach_weight * abs(altitude_error)
        )
        components["vertical_speed_guidance"] = float(
            -self.config.vertical_speed_guidance_scale * approach_weight * vertical_speed_error_ratio
        )

        reward = float(sum(components.values()))
        components["total"] = reward
        self.last_reward_components = components
        self.last_reward_diagnostics = {
            "allowed_altitude": float(allowed_altitude),
            "extra_altitude_m": float(extra_altitude),
            "turn_angle_ratio": float(turn_angle_ratio),
            "detour_distance_m": float(detour_distance),
            "goal_approach_weight": float(approach_weight),
            "goal_altitude_error_m": float(altitude_error),
            "desired_vertical_speed": float(desired_vz),
        }
        return float(reward)

    def _get_state(self) -> np.ndarray:
        delta = self.goal - self.position
        distance = float(np.linalg.norm(delta))
        heading = delta / distance if distance > 1e-9 else np.zeros(3, dtype=np.float32)
        speed_ratio = float(np.linalg.norm(self.velocity)) / self.config.max_speed
        acceleration_scale = min(self.config.normal_acceleration, self.config.max_acceleration)
        acceleration_ratio = float(np.linalg.norm(self.acceleration)) / acceleration_scale
        stopping_distance = float(np.linalg.norm(self.velocity)) ** 2 / (2.0 * self.config.max_deceleration)
        clearance = self._nearest_clearance(self.position)
        braking_margin = np.clip((clearance - stopping_distance) / self.config.lidar_range, -1.0, 1.0)
        state = np.concatenate(
            [
                (self.position - self.map_min) / self.map_size,
                delta / self.map_size,
                np.asarray([distance / self.max_distance], dtype=np.float32),
                heading.astype(np.float32),
                self.velocity / self.config.max_speed,
                self.acceleration / acceleration_scale,
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

    def _sample_goal_near_obstacle(
        self,
        start: np.ndarray,
        min_clearance: float,
        max_clearance: float,
    ) -> np.ndarray | None:
        """Sample a free goal in a requested shell around any building."""
        for _ in range(10_000):
            point = self._sample_free_point()
            if np.linalg.norm(point - start) < self.config.min_start_goal_distance:
                continue
            clearance = self._nearest_obstacle_clearance(point)
            if min_clearance <= clearance <= max_clearance:
                return point
        return None

    def _sample_free_point(self) -> np.ndarray:
        low = self.map_min + self.config.uav_radius
        high = self.map_max - self.config.uav_radius
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
        radius = self.config.uav_radius
        if np.any(point < self.map_min + radius) or np.any(point > self.map_max - radius):
            return True
        return any(obstacle.contains(point, margin=radius) for obstacle in self.config.obstacles)

    def _nearest_clearance(self, point: np.ndarray) -> float:
        boundary_clearance = float(min(np.min(point - self.map_min), np.min(self.map_max - point)))
        obstacle_clearance = min(
            (obstacle.distance_to(point) for obstacle in self.config.obstacles),
            default=float("inf"),
        )
        return max(0.0, min(boundary_clearance, obstacle_clearance) - self.config.uav_radius)

    def _nearest_obstacle_clearance(self, point: np.ndarray) -> float:
        """Return UAV-surface clearance to the nearest building, excluding map boundaries."""
        obstacle_distance = min(
            (obstacle.distance_to(point) for obstacle in self.config.obstacles),
            default=float("inf"),
        )
        return max(0.0, obstacle_distance - self.config.uav_radius)

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
            "goal_sampling_mode": self.goal_sampling_mode,
            "goal_obstacle_clearance": (
                float(self.goal_obstacle_clearance)
                if np.isfinite(self.goal_obstacle_clearance)
                else None
            ),
            "reward": float(reward),
            "progress": float(progress),
            "reward_components": self.last_reward_components.copy(),
            "reward_diagnostics": self.last_reward_diagnostics.copy(),
        }
