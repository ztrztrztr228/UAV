# -*- coding: utf-8 -*-
"""无人机三维路径规划环境。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .actions import ACTION_DIRECTIONS, ACTION_NAMES, HOVER_ACTION_INDEX
from .config import DEFAULT_SEED, UAVEnvConfig


class UAVPathPlanningEnv:
    """类似 gym 的三维无人机轨迹规划环境。

    核心接口：
        reset(): 初始化一个 episode，返回初始状态；
        step(action): 执行动作，返回 next_state, reward, done, info。

    三维状态空间共 40 维：
        1. 当前三维坐标 x/y/z，3 维；
        2. 目标相对向量 dx/dy/dz，3 维；
        3. 到目标点的三维距离，1 维；
        4. 朝向目标点的三维单位向量，3 维；
        5. 上一步三维移动方向，3 维；
        6. 26 个三维方向雷达距离，26 维；
        7. 当前 episode 进度，1 维。

    三维动作空间共 27 个离散动作：
        26 个三维邻接移动方向 + 1 个悬停 hover。
    """

    def __init__(self, config: UAVEnvConfig | None = None, seed: int = DEFAULT_SEED) -> None:
        self.config = config or UAVEnvConfig()#构造函数
        self.rng = np.random.default_rng(seed)

        self.num_actions = len(ACTION_NAMES)
        self.lidar_dim = self.num_actions - 1
        self.state_dim = 3 + 3 + 1 + 3 + 3 + self.lidar_dim + 1
        self.map_size = np.asarray(
            [self.config.map_width, self.config.map_height, self.config.map_altitude],
            dtype=np.float32,
        )
        self.max_distance = float(np.linalg.norm(self.map_size))

        self.position = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(3, dtype=np.float32)
        self.last_move = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory: list[np.ndarray] = []
        self.reset()
####新回合做初始化
    def reset(
        self,
        start: Sequence[float] | None = None,
        goal: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """重置三维环境并返回初始状态。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.start = self._sample_free_point() if start is None else self._validate_free_point(start, "start")
        self.goal = (
            self._sample_goal_far_from(self.start)
            if goal is None
            else self._validate_free_point(goal, "goal")
        )

        self.position = self.start.copy()
        self.last_move = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.path_length = 0.0
        self.done = False
        self.trajectory = [self.position.copy()]
        return self._get_state()
####执行动作
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """执行一个三维动作，返回下一状态、奖励、终止标志和调试信息。"""
        if self.done:
            return self._get_state(), 0.0, True, self._info("already_done")
#检查动作是否合理
        action = int(action)
        if not 0 <= action < self.num_actions:
            raise ValueError(f"Action must be in [0, {self.num_actions - 1}], got {action}.")
###做动作
        prev_position = self.position.copy()
        prev_distance = self.distance_to_goal()
        move = ACTION_DIRECTIONS[action] * self.config.step_length
        candidate = prev_position + move

        self.steps += 1
        collision = self._segment_in_collision(prev_position, candidate)
##检测是否碰撞
        if collision:
            self.done = True#碰撞就停
            reward = self.config.collision_penalty
            reward -= self.config.distance_penalty_scale * (prev_distance / self.max_distance)
            info = self._info("collision", collision=True, reward=reward)
            return self._get_state(), float(reward), True, info
##记录轨迹
        self.position = candidate.astype(np.float32)
        self.path_length += float(np.linalg.norm(move))
        self.trajectory.append(self.position.copy())

        new_distance = self.distance_to_goal()
        progress = prev_distance - new_distance
        reward = self._shaped_reward(action, move, progress, new_distance)
#到终点加分/超时
        reached_goal = new_distance <= self.config.goal_radius
        timeout = self.steps >= self.config.max_steps
        event = "running"

        if reached_goal:
            self.done = True
            event = "goal"
            speed_bonus = 1.0 - self.steps / max(1, self.config.max_steps)#步数少，奖励大
            reward += self.config.goal_reward + 25.0 * max(0.0, speed_bonus)
        elif timeout:
            self.done = True
            event = "timeout"
            reward += self.config.timeout_penalty

        self.last_move = move.astype(np.float32)
        info = self._info(
            event,
            success=reached_goal,
            collision=False,
            reward=reward,
            progress=progress,
        )
        return self._get_state(), float(reward), bool(self.done), info

    def distance_to_goal(self) -> float:
        """当前无人机到三维目标点的欧氏距离。"""
        return float(np.linalg.norm(self.goal - self.position))
####奖励函数
    def _shaped_reward(
        self,
        action: int,
        move: np.ndarray,
        progress: float,
        distance: float,
    ) -> float:
        """三维密集奖励函数。

        奖励仍以“接近目标”为核心，同时加入时间、悬停、转弯、近障碍物
        和轻微高度代价，避免无人机无意义地飞到过高位置。
        """
        reward = self.config.progress_reward_scale * progress
        reward -= self.config.distance_penalty_scale * (distance / self.max_distance)
        reward -= self.config.step_penalty

        if action == HOVER_ACTION_INDEX:
            reward -= self.config.hover_penalty

        prev_norm = float(np.linalg.norm(self.last_move))
        move_norm = float(np.linalg.norm(move))
        if prev_norm > 1e-6 and move_norm > 1e-6:
            prev_dir = self.last_move / prev_norm
            move_dir = move / move_norm
            turn_amount = 1.0 - float(np.clip(np.dot(prev_dir, move_dir), -1.0, 1.0))
            reward -= self.config.turn_penalty_scale * turn_amount

        clearance = self._nearest_clearance(self.position)
        if clearance < self.config.safety_radius:
            unsafe_ratio = (self.config.safety_radius - clearance) / max(1e-6, self.config.safety_radius)
            reward -= self.config.proximity_penalty_scale * unsafe_ratio

        # 轻微高度代价：鼓励无人机在满足避障的前提下不要无意义爬升太高。
        altitude_ratio = self.position[2] / max(1e-6, self.config.map_altitude)
        reward -= self.config.altitude_penalty_scale * float(altitude_ratio)

        return float(reward)

    def _get_state(self) -> np.ndarray:
        """构造 40 维三维状态向量，并进行归一化。"""
        delta = self.goal - self.position
        distance = float(np.linalg.norm(delta))
        heading = delta / distance if distance > 1e-9 else np.zeros(3, dtype=np.float32)

        prev_move = self.last_move / max(1e-6, self.config.step_length)
        lidar = self._lidar_scan()
        state = np.concatenate(
            [
                self.position / self.map_size,
                delta / self.map_size,
                np.asarray([distance / self.max_distance], dtype=np.float32),
                heading.astype(np.float32),
                prev_move.astype(np.float32),
                lidar.astype(np.float32),
                np.asarray([self.steps / max(1, self.config.max_steps)], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def _lidar_scan(self) -> np.ndarray:
        """模拟 26 个三维方向的雷达探测。"""
        readings: list[float] = []
        max_range = self.config.lidar_range
        resolution = self.config.lidar_resolution
        for direction in ACTION_DIRECTIONS[:-1]:
            distance = max_range
            n = max(2, int(math.ceil(max_range / resolution)))
            for i in range(1, n + 1):
                probe_distance = min(max_range, i * resolution)
                probe = self.position + direction * probe_distance
                if self._point_in_collision(probe):
                    distance = probe_distance
                    break
            readings.append(distance / max_range)
        return np.asarray(readings, dtype=np.float32)
####取点
    def _sample_goal_far_from(self, start: np.ndarray) -> np.ndarray:
        """随机采样一个离起点足够远的三维目标点。"""
        for _ in range(10_000):
            point = self._sample_free_point()
            if np.linalg.norm(point - start) >= self.config.min_start_goal_distance:
                return point
        raise RuntimeError("Could not sample a valid goal far from start.")

    def _sample_free_point(self) -> np.ndarray:
        """在三维地图空闲空间中随机采样一个合法点。"""
        low = np.asarray(
            [self.config.uav_radius, self.config.uav_radius, self.config.uav_radius],
            dtype=np.float32,
        )
        high = self.map_size - self.config.uav_radius
        for _ in range(10_000):
            point = self.rng.uniform(low, high).astype(np.float32)
            if not self._point_in_collision(point):
                return point
        raise RuntimeError("Could not sample a valid free point.")

    def _validate_free_point(self, point: Sequence[float], name: str) -> np.ndarray:
        """检查用户指定的三维起点/目标点是否合法。"""
        array = np.asarray(point, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"{name} must have three values x/y/z, got {point}.")
        if self._point_in_collision(array):
            raise ValueError(f"{name} point {array.tolist()} is outside the map or inside an obstacle.")
        return array
######碰撞检测
    def _segment_in_collision(self, start: np.ndarray, end: np.ndarray) -> bool:
        """检查一段三维飞行线段是否碰撞。"""
        length = float(np.linalg.norm(end - start))
        n = max(2, int(math.ceil(length / self.config.collision_resolution)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            point = start + t * (end - start)
            if self._point_in_collision(point):
                return True
        return False

    def _point_in_collision(self, point: np.ndarray) -> bool:
        """检查三维点是否越界或进入任意长方体障碍物。"""
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        radius = self.config.uav_radius
        if x < radius or x > self.config.map_width - radius:
            return True
        if y < radius or y > self.config.map_height - radius:
            return True
        if z < radius or z > self.config.map_altitude - radius:
            return True
        return any(obstacle.contains(point, margin=radius) for obstacle in self.config.obstacles)

    def _nearest_clearance(self, point: np.ndarray) -> float:
        """计算三维点到最近边界或障碍物的安全净空距离。"""
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        boundary_clearance = min(
            x,
            y,
            z,
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
    ) -> dict[str, object]:
        """返回调试和评估信息。"""
        return {
            "event": event,
            "success": bool(success),
            "collision": bool(collision),
            "distance_to_goal": self.distance_to_goal(),
            "clearance": self._nearest_clearance(self.position),
            "path_length": self.path_length,
            "steps": self.steps,
            "position": self.position.copy(),
            "goal": self.goal.copy(),
            "reward": float(reward),
            "progress": float(progress),
        }
