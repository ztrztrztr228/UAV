# -*- coding: utf-8 -*-
"""包含速度和加速度状态的无人机三维动力学强化学习环境。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .actions import ACTION_DIRECTIONS, ACTION_NAMES, COAST_ACTION_INDEX
from .config import DEFAULT_SEED, UAVEnvConfig
from .trajectory import TimedTrajectory, dynamics_samples_to_trajectory


# ==================== 三维强化学习环境 ====================
class UAVPathPlanningEnv:
    """使用点质量模型的三维轨迹规划环境。

    动作为 26 个单位方向上的正常飞行加速度和一个零加速度动作。每一步先计算
    ``v[k+1] = clip(v[k] + a[k] * dt)``，再用梯形积分更新位置。50 维状态由
    位置、目标相对量、航向、速度、加速度、动力学裕度、雷达、时间进度和
    提前制动特征组成。新增特征追加在旧 46 维状态末尾，以兼容旧 checkpoint。
    """

    def __init__(self, config: UAVEnvConfig | None = None, seed: int = DEFAULT_SEED) -> None:
        # 读取并校验环境参数，同时确定动作数、状态维度和地图归一化尺度。
        self.config = config or UAVEnvConfig()
        self._validate_config()
        self.rng = np.random.default_rng(seed)
        self.num_actions = len(ACTION_NAMES)
        self.lidar_dim = self.num_actions - 1
        self.early_braking_dim = 4
        self.state_dim = 3 + 3 + 1 + 3 + 3 + 3 + 3 + self.lidar_dim + 1 + self.early_braking_dim
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

        # 把障碍物边界预先整理成数组，供雷达和碰撞检测批量计算。
        self._obstacle_mins = np.asarray(
            [[item.xmin, item.ymin, item.zmin] for item in self.config.obstacles],
            dtype=np.float32,
        ).reshape(-1, 3)
        self._obstacle_maxs = np.asarray(
            [[item.xmax, item.ymax, item.zmax] for item in self.config.obstacles],
            dtype=np.float32,
        ).reshape(-1, 3)
        self._last_lidar_scan = np.ones(self.lidar_dim, dtype=np.float32)

        # 保存当前 episode 的动力学状态、任务状态和完整轨迹历史。
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

    # ==================== 参数合法性检查 ====================
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
            "lidar_range",
            "lidar_resolution",
            "collision_resolution",
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
            "progress_reward_scale",
            "step_penalty",
            "safety_risk_penalty_scale",
            "jerk_penalty_scale",
            "goal_reward",
        )
        for name in non_negative_names:
            if float(getattr(self.config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.config.collision_penalty > 0.0 or self.config.timeout_penalty > 0.0:
            raise ValueError("terminal failure penalties must be non-positive.")

    def reset(
        self,
        start: Sequence[float] | None = None,
        goal: Sequence[float] | None = None,
        seed: int | None = None,
        goal_near_obstacle_probability: float = 0.0,
        goal_near_obstacle_min_clearance: float = 2.0,
        goal_near_obstacle_max_clearance: float = 12.0,
        goal_max_start_distance: float | None = None,
        goal_altitude_min: float | None = None,
        goal_altitude_max: float | None = None,
        goal_near_obstacle_horizontal_only: bool = False,
    ) -> np.ndarray:
        # 先检查课程学习传入的概率、净空、距离和高度范围。
        if not 0.0 <= goal_near_obstacle_probability <= 1.0:
            raise ValueError("goal_near_obstacle_probability must be in [0, 1].")
        if goal_near_obstacle_min_clearance < 0.0:
            raise ValueError("goal_near_obstacle_min_clearance must be non-negative.")
        if goal_near_obstacle_max_clearance < goal_near_obstacle_min_clearance:
            raise ValueError(
                "goal_near_obstacle_max_clearance must not be below the minimum clearance."
            )
        altitude_min = (
            self.config.uav_radius
            if goal_altitude_min is None
            else float(goal_altitude_min)
        )
        altitude_max = (
            self.config.map_altitude - self.config.uav_radius
            if goal_altitude_max is None
            else float(goal_altitude_max)
        )
        if altitude_min < self.config.uav_radius or altitude_max > (
            self.config.map_altitude - self.config.uav_radius
        ):
            raise ValueError("Goal altitude curriculum must stay inside the flyable map altitude.")
        if altitude_max < altitude_min:
            raise ValueError("goal_altitude_max must not be below goal_altitude_min.")
        if (
            goal_max_start_distance is not None
            and goal_max_start_distance < self.config.min_start_goal_distance
        ):
            raise ValueError(
                "goal_max_start_distance must not be below min_start_goal_distance."
            )
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 起点可由用户指定或随机生成；目标按“指定、建筑附近、均匀”优先级采样。
        self.start = self._sample_free_point() if start is None else self._validate_free_point(start, "start")
        if goal is not None:
            self.goal = self._validate_free_point(goal, "goal")
            self.goal_sampling_mode = "specified"
        elif self.config.obstacles and self.rng.random() < goal_near_obstacle_probability:
            near_goal = self._sample_goal_near_obstacle(
                self.start,
                goal_near_obstacle_min_clearance,
                goal_near_obstacle_max_clearance,
                max_start_distance=goal_max_start_distance,
                altitude_min=altitude_min,
                altitude_max=altitude_max,
                horizontal_only=goal_near_obstacle_horizontal_only,
            )
            if near_goal is None:
                self.goal = self._sample_goal_far_from(
                    self.start,
                    max_start_distance=goal_max_start_distance,
                    altitude_min=altitude_min,
                    altitude_max=altitude_max,
                )
                self.goal_sampling_mode = "uniform_fallback"
            else:
                self.goal = near_goal
                self.goal_sampling_mode = "near_obstacle"
        else:
            self.goal = self._sample_goal_far_from(
                self.start,
                max_start_distance=goal_max_start_distance,
                altitude_min=altitude_min,
                altitude_max=altitude_max,
            )
            self.goal_sampling_mode = "uniform"
        self.goal_obstacle_clearance = self._nearest_obstacle_clearance(self.goal)

        # 每个 episode 都从静止状态重新开始，并清空累计统计和轨迹缓存。
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

    # ==================== 单步环境交互 ====================
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return self._get_state(), 0.0, True, self._info("already_done")
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}], got {action}.")

        previous_position = self.position.copy()
        previous_velocity = self.velocity.copy()
        previous_acceleration = self.acceleration.copy()
        previous_distance = self.distance_to_goal()
        (
            commanded_acceleration,
            next_velocity,
            effective_acceleration,
            candidate,
            clipped_velocity,
        ) = self._predict_action_dynamics(action)
        dt = self.config.trajectory_dt

        # 沿本步真实抛物线轨迹检查碰撞；碰撞时不提交候选状态，直接终止回合。
        self.steps += 1
        if self._dynamics_segment_in_collision(
            previous_position,
            previous_velocity,
            effective_acceleration,
            dt,
        ):
            self.done = True
            reward = self.config.collision_penalty
            self.last_reward_components = {
                "collision": float(self.config.collision_penalty),
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

        # 安全时提交新的位置、速度和加速度，并累计路径长度和时序样本。
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
            progress,
            previous_acceleration,
        )
        speed = float(np.linalg.norm(self.velocity))
        reached_goal = distance <= self.config.goal_radius and speed <= self.config.goal_speed_tolerance
        timeout = self.steps >= self.config.max_steps

        # 到达目标还要求末速度足够低；否则按最大步数判断超时。
        event = "running"
        if reached_goal:
            self.done = True
            event = "goal"
            reward += self.config.goal_reward
            self.last_reward_components["goal"] = float(self.config.goal_reward)
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

    # ==================== 动力学预测与约束 ====================
    def _predict_action_dynamics(
        self,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Predict one action with exactly the same dynamics used by :meth:`step`."""
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}], got {action}.")

        # 先将离散动作变成加速度命令，再依次施加加速度、jerk 和速度约束。
        dt = self.config.trajectory_dt
        commanded_acceleration = self._limit_commanded_acceleration(
            ACTION_DIRECTIONS[action] * self.config.normal_acceleration,
            self.velocity,
        )
        acceleration_delta = commanded_acceleration - self.acceleration
        max_acceleration_delta = self.config.max_jerk * dt
        acceleration_delta_norm = float(np.linalg.norm(acceleration_delta))
        if acceleration_delta_norm > max_acceleration_delta:
            commanded_acceleration = self.acceleration + acceleration_delta * (
                max_acceleration_delta / acceleration_delta_norm
            )

        unconstrained_velocity = self.velocity + commanded_acceleration * dt
        next_velocity = self._limit_velocity(unconstrained_velocity)
        clipped_velocity = float(np.linalg.norm(unconstrained_velocity - next_velocity))
        effective_acceleration = (next_velocity - self.velocity) / dt
        # 使用梯形积分更新位置，使位置与前后时刻速度保持一致。
        candidate = self.position + 0.5 * (self.velocity + next_velocity) * dt
        return (
            commanded_acceleration.astype(np.float32),
            next_velocity.astype(np.float32),
            effective_acceleration.astype(np.float32),
            candidate.astype(np.float32),
            clipped_velocity,
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
        # 约束顺序为水平速度、升降速度、爬升角，最后再限制三维合速度。
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

    # ==================== 奖励函数 ====================
    def _shaped_reward(
        self,
        progress: float,
        previous_acceleration: np.ndarray,
    ) -> float:
        # 米制进度按单步最大位移归一化并裁剪，避免它淹没终止奖惩。
        max_step_distance = max(self.config.max_speed * self.config.trajectory_dt, 1e-6)
        normalized_progress = float(np.clip(progress / max_step_distance, -1.0, 1.0))
        components: dict[str, float] = {
            "progress": float(self.config.progress_reward_scale * normalized_progress),
            "step": float(-self.config.step_penalty),
        }

        speed = float(np.linalg.norm(self.velocity))
        jerk = float(np.linalg.norm(self.acceleration - previous_acceleration)) / self.config.trajectory_dt
        jerk_ratio = float(np.clip(jerk / self.config.max_jerk, 0.0, 1.0))
        components["jerk"] = float(
            -self.config.jerk_penalty_scale * jerk_ratio
        )

        # 用一个连续风险项统一原先重复的净空和制动风险惩罚。
        clearance = self._nearest_clearance(self.position)
        stopping_distance = speed * speed / (2.0 * self.config.max_deceleration)
        required_clearance = self.config.safety_radius + stopping_distance
        safety_risk = 0.0
        if clearance < required_clearance:
            safety_risk = float(
                np.clip(
                    (required_clearance - clearance) / max(required_clearance, 1e-6),
                    0.0,
                    1.0,
                )
            )
        components["safety_risk"] = float(
            -self.config.safety_risk_penalty_scale * safety_risk
        )

        reward = float(sum(components.values()))
        components["total"] = reward
        self.last_reward_components = components
        self.last_reward_diagnostics = {
            "progress_m": float(progress),
            "normalized_progress": float(normalized_progress),
            "clearance_m": float(clearance),
            "stopping_distance_m": float(stopping_distance),
            "required_clearance_m": float(required_clearance),
            "safety_risk": float(safety_risk),
            "jerk_ratio": float(jerk_ratio),
        }
        return float(reward)

    # ==================== 50 维状态构造 ====================
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
        lidar_scan = self._lidar_scan()
        self._last_lidar_scan = lidar_scan.copy()
        early_braking_state = self._early_braking_state(lidar_scan, stopping_distance)
        # 拼接顺序固定；旧 checkpoint 迁移依赖新增制动特征位于末尾。
        state = np.concatenate(
            [
                (self.position - self.map_min) / self.map_size,
                delta / self.map_size,
                np.asarray([distance / self.max_distance], dtype=np.float32),
                heading.astype(np.float32),
                self.velocity / self.config.max_speed,
                self.acceleration / acceleration_scale,
                np.asarray([speed_ratio, acceleration_ratio, braking_margin], dtype=np.float32),
                lidar_scan,
                np.asarray([self.steps / max(1, self.config.max_steps)], dtype=np.float32),
                early_braking_state,
            ]
        )
        return state.astype(np.float32)

    def _early_braking_state(
        self,
        lidar_scan: np.ndarray,
        stopping_distance: float | None = None,
    ) -> np.ndarray:
        """Return explicit stopping-distance, forward-clearance, margin and TTC features."""
        speed = float(np.linalg.norm(self.velocity))
        if stopping_distance is None:
            stopping_distance = speed * speed / (2.0 * self.config.max_deceleration)

        if speed > 1e-6:
            velocity_direction = self.velocity / speed
            forward_index = int(np.argmax(ACTION_DIRECTIONS[:-1] @ velocity_direction))
            forward_distance = float(lidar_scan[forward_index] * self.config.lidar_range)
            time_to_collision = forward_distance / speed
        else:
            forward_distance = self.config.lidar_range
            time_to_collision = 10.0

        stopping_distance_ratio = float(
            np.clip(stopping_distance / max(self.config.lidar_range, 1e-6), 0.0, 4.0) / 4.0
        )
        forward_distance_ratio = float(
            np.clip(forward_distance / max(self.config.lidar_range, 1e-6), 0.0, 1.0)
        )
        forward_braking_margin = float(
            np.clip(
                (forward_distance - stopping_distance) / max(self.config.lidar_range, 1e-6),
                -1.0,
                1.0,
            )
        )
        time_to_collision_ratio = float(np.clip(time_to_collision / 10.0, 0.0, 1.0))
        return np.asarray(
            [
                stopping_distance_ratio,
                forward_distance_ratio,
                forward_braking_margin,
                time_to_collision_ratio,
            ],
            dtype=np.float32,
        )

    # ==================== 安全动作屏蔽 ====================
    def safe_action_mask(
        self,
        safety_buffer: float = 0.5,
        worsening_tolerance: float = 0.25,
    ) -> np.ndarray:
        """Return actions that avoid immediate collision and do not worsen braking risk.

        When the vehicle already has insufficient stopping distance, actions are kept only
        if they maintain or improve the predicted braking margin. At least one action is
        always returned so action selection remains well-defined in an unavoidable state.
        """
        current_speed = float(np.linalg.norm(self.velocity))
        current_stopping_distance = current_speed * current_speed / (
            2.0 * self.config.max_deceleration
        )
        lidar_scan = self._last_lidar_scan
        if lidar_scan.shape != (self.lidar_dim,):
            lidar_scan = self._lidar_scan()
        if current_speed > 1e-6:
            current_direction = self.velocity / current_speed
            current_forward_index = int(np.argmax(ACTION_DIRECTIONS[:-1] @ current_direction))
            current_forward_clearance = float(
                lidar_scan[current_forward_index] * self.config.lidar_range
            )
        else:
            current_forward_clearance = self.config.lidar_range
        current_margin = current_forward_clearance - current_stopping_distance
        mask = np.zeros(self.num_actions, dtype=bool)
        predicted_margins = np.full(self.num_actions, -float("inf"), dtype=np.float64)
        non_collision = np.zeros(self.num_actions, dtype=bool)

        # 对每个动作复用真实动力学预测，排除立即碰撞或使制动裕度继续恶化的动作。
        for action in range(self.num_actions):
            _, next_velocity, effective_acceleration, candidate, _ = self._predict_action_dynamics(action)
            collides = self._dynamics_segment_in_collision(
                self.position,
                self.velocity,
                effective_acceleration,
                self.config.trajectory_dt,
            )
            if collides:
                continue

            non_collision[action] = True
            next_speed = float(np.linalg.norm(next_velocity))
            next_stopping_distance = next_speed * next_speed / (2.0 * self.config.max_deceleration)
            if next_speed > 1e-6:
                next_direction = next_velocity / next_speed
                forward_index = int(np.argmax(ACTION_DIRECTIONS[:-1] @ next_direction))
                forward_clearance = float(lidar_scan[forward_index] * self.config.lidar_range)
                travelled = max(
                    0.0,
                    float(np.dot(candidate - self.position, ACTION_DIRECTIONS[forward_index])),
                )
                forward_clearance = max(0.0, forward_clearance - travelled)
            else:
                forward_clearance = self.config.lidar_range
            predicted_margin = forward_clearance - next_stopping_distance
            predicted_margins[action] = predicted_margin
            mask[action] = bool(
                predicted_margin >= safety_buffer
                or predicted_margin >= current_margin - worsening_tolerance
            )

        # 极端情况下至少保留一个相对最安全的动作，避免策略无法选择动作。
        if not np.any(mask):
            candidates = np.flatnonzero(non_collision)
            if len(candidates):
                best_action = int(candidates[np.argmax(predicted_margins[candidates])])
            else:
                best_action = COAST_ACTION_INDEX
            mask[best_action] = True
        return mask

    # ==================== 26 方向雷达 ====================
    def _lidar_scan(self) -> np.ndarray:
        readings: list[float] = []
        for direction in ACTION_DIRECTIONS[:-1]:
            distance = self.config.lidar_range
            sample_count = max(2, int(math.ceil(distance / self.config.lidar_resolution)))
            probe_distances = np.minimum(
                distance,
                np.arange(1, sample_count + 1, dtype=np.float32)
                * self.config.lidar_resolution,
            )
            probes = self.position[None, :] + probe_distances[:, None] * direction[None, :]
            collisions = self._points_in_collision(probes)
            collision_indices = np.flatnonzero(collisions)
            if len(collision_indices):
                distance = float(probe_distances[int(collision_indices[0])])
            readings.append(distance / self.config.lidar_range)
        return np.asarray(readings, dtype=np.float32)

    # ==================== 起点与目标采样 ====================
    def _sample_goal_far_from(
        self,
        start: np.ndarray,
        max_start_distance: float | None = None,
        altitude_min: float | None = None,
        altitude_max: float | None = None,
    ) -> np.ndarray:
        for _ in range(10_000):
            point = self._sample_free_point(altitude_min, altitude_max)
            distance = float(np.linalg.norm(point - start))
            if distance < self.config.min_start_goal_distance:
                continue
            if max_start_distance is None or distance <= max_start_distance:
                return point
        raise RuntimeError("Could not sample a valid goal far from start.")

    def _sample_goal_near_obstacle(
        self,
        start: np.ndarray,
        min_clearance: float,
        max_clearance: float,
        max_start_distance: float | None = None,
        altitude_min: float | None = None,
        altitude_max: float | None = None,
        horizontal_only: bool = False,
    ) -> np.ndarray | None:
        """Sample a free goal in a requested shell around any building."""
        for _ in range(10_000):
            point = self._sample_free_point(altitude_min, altitude_max)
            distance = float(np.linalg.norm(point - start))
            if distance < self.config.min_start_goal_distance:
                continue
            if max_start_distance is not None and distance > max_start_distance:
                continue
            clearance = (
                self._nearest_obstacle_horizontal_clearance(point)
                if horizontal_only
                else self._nearest_obstacle_clearance(point)
            )
            if min_clearance <= clearance <= max_clearance:
                return point
        return None

    def _sample_free_point(
        self,
        altitude_min: float | None = None,
        altitude_max: float | None = None,
    ) -> np.ndarray:
        low = self.map_min + self.config.uav_radius
        high = self.map_max - self.config.uav_radius
        if altitude_min is not None:
            low[2] = max(low[2], float(altitude_min))
        if altitude_max is not None:
            high[2] = min(high[2], float(altitude_max))
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

    # ==================== 连续轨迹碰撞检测 ====================
    def _segment_in_collision(self, start: np.ndarray, end: np.ndarray) -> bool:
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / self.config.collision_resolution)) + 1)
        values = np.linspace(0.0, 1.0, count, dtype=np.float32)
        points = start[None, :] + values[:, None] * (end - start)[None, :]
        return bool(np.any(self._points_in_collision(points)))

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
        values = np.linspace(0.0, duration, count, dtype=np.float32)
        points = (
            start[None, :]
            + values[:, None] * initial_velocity[None, :]
            + 0.5 * values[:, None] * values[:, None] * acceleration[None, :]
        )
        return bool(np.any(self._points_in_collision(points)))

    def _point_in_collision(self, point: np.ndarray) -> bool:
        return bool(self._points_in_collision(np.asarray(point, dtype=np.float32)[None, :])[0])

    def _points_in_collision(self, points: np.ndarray) -> np.ndarray:
        """Vectorized boundary and axis-aligned obstacle checks for one or more points."""
        values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        radius = self.config.uav_radius
        collisions = np.any(values < self.map_min + radius, axis=1) | np.any(
            values > self.map_max - radius,
            axis=1,
        )
        if len(self._obstacle_mins):
            inside_obstacle = np.all(
                values[:, None, :] >= self._obstacle_mins[None, :, :] - radius,
                axis=2,
            ) & np.all(
                values[:, None, :] <= self._obstacle_maxs[None, :, :] + radius,
                axis=2,
            )
            collisions |= np.any(inside_obstacle, axis=1)
        return collisions

    # ==================== 障碍物和边界净空 ====================
    def _nearest_clearance(self, point: np.ndarray) -> float:
        boundary_clearance = float(min(np.min(point - self.map_min), np.min(self.map_max - point)))
        obstacle_clearance = self._nearest_obstacle_distance(point)
        return max(0.0, min(boundary_clearance, obstacle_clearance) - self.config.uav_radius)

    def _nearest_obstacle_clearance(self, point: np.ndarray) -> float:
        """Return UAV-surface clearance to the nearest building, excluding map boundaries."""
        obstacle_distance = self._nearest_obstacle_distance(point)
        return max(0.0, obstacle_distance - self.config.uav_radius)

    def _nearest_obstacle_horizontal_clearance(self, point: np.ndarray) -> float:
        """Return XY clearance so building-side targets exclude rooftop-only proximity."""
        if not len(self._obstacle_mins):
            return float("inf")
        value = np.asarray(point, dtype=np.float32)[:2]
        below = np.maximum(self._obstacle_mins[:, :2] - value[None, :], 0.0)
        above = np.maximum(value[None, :] - self._obstacle_maxs[:, :2], 0.0)
        distance = float(np.min(np.linalg.norm(below + above, axis=1)))
        return max(0.0, distance - self.config.uav_radius)

    def _nearest_obstacle_distance(self, point: np.ndarray) -> float:
        if not len(self._obstacle_mins):
            return float("inf")
        value = np.asarray(point, dtype=np.float32)
        below = np.maximum(self._obstacle_mins - value[None, :], 0.0)
        above = np.maximum(value[None, :] - self._obstacle_maxs, 0.0)
        deltas = below + above
        return float(np.min(np.linalg.norm(deltas, axis=1)))

    # ==================== 调试与评估信息汇总 ====================
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
