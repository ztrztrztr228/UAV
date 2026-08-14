# -*- coding: utf-8 -*-
"""Validation helpers for timed trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .environment import UAVPathPlanningEnv
from .trajectory import TimedTrajectory


@dataclass(frozen=True)
class TrajectoryValidationResult:
    """Numerical checks comparing timed trajectory samples to the planned path."""

    passed: bool
    deviation_tolerance: float
    max_deviation: float
    mean_deviation: float
    max_deviation_index: int
    max_deviation_time: float
    collision_free: bool
    within_bounds: bool
    max_speed: float
    max_horizontal_speed: float
    max_climb_speed: float
    max_descent_speed: float
    max_climb_angle_deg: float
    max_acceleration: float
    max_deceleration: float
    max_jerk: float
    speed_limit_satisfied: bool
    horizontal_speed_limit_satisfied: bool
    climb_speed_limit_satisfied: bool
    descent_speed_limit_satisfied: bool
    climb_angle_limit_satisfied: bool
    acceleration_limit_satisfied: bool
    deceleration_limit_satisfied: bool
    jerk_limit_satisfied: bool
    dynamics_consistent: bool
    max_position_integration_error: float
    max_velocity_integration_error: float
    goal_reached: bool
    final_goal_distance: float
    final_speed: float
    sample_count: int
    total_time: float
    total_length: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_points(points: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must be an array-like sequence of 3D coordinates.")
    if len(array) == 0:
        raise ValueError("points must contain at least one coordinate.")
    return array


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def distance_to_polyline(point: np.ndarray, polyline: Sequence[np.ndarray] | np.ndarray) -> float:
    """Return the shortest Euclidean distance from one point to a 3D polyline."""
    points = _as_points(polyline)
    if len(points) == 1:
        return float(np.linalg.norm(point - points[0]))
    return min(
        _point_to_segment_distance(point, start, end)
        for start, end in zip(points[:-1], points[1:])
    )


def trajectory_deviations(
    reference_path: Sequence[np.ndarray] | np.ndarray,
    trajectory: TimedTrajectory,
) -> np.ndarray:
    """Compute each timed sample's distance to the reference path polyline."""
    path_points = _as_points(reference_path)
    return np.asarray(
        [distance_to_polyline(point, path_points) for point in trajectory.position],
        dtype=np.float64,
    )


def _within_map_bounds(env: UAVPathPlanningEnv, point: np.ndarray) -> bool:
    radius = env.config.uav_radius
    return bool(np.all(point >= env.map_min + radius) and np.all(point <= env.map_max - radius))


def _trajectory_collision_free(env: UAVPathPlanningEnv, trajectory: TimedTrajectory) -> bool:
    for point in trajectory.position:
        if env._point_in_collision(point):
            return False
    for index in range(len(trajectory.position) - 1):
        duration = float(trajectory.time[index + 1] - trajectory.time[index])
        if duration <= 0.0:
            return False
        if env._dynamics_segment_in_collision(
            trajectory.position[index],
            trajectory.velocity[index],
            trajectory.acceleration[index + 1],
            duration,
        ):
            return False
    return True


def validate_timed_trajectory(
    env: UAVPathPlanningEnv,
    reference_path: Sequence[np.ndarray] | np.ndarray,
    trajectory: TimedTrajectory,
    deviation_tolerance: float,
    max_speed: float | None = None,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
    goal: Sequence[float] | np.ndarray | None = None,
    goal_radius: float | None = None,
    goal_speed_tolerance: float | None = None,
    integration_tolerance: float = 1e-4,
) -> TrajectoryValidationResult:
    """复算轨迹的碰撞、边界、动力学积分和任务终点约束。"""
    if deviation_tolerance < 0.0:
        raise ValueError("deviation_tolerance must be non-negative.")

    deviations = trajectory_deviations(reference_path, trajectory)
    max_index = int(np.argmax(deviations)) if len(deviations) else 0
    collision_free = _trajectory_collision_free(env, trajectory)
    within_bounds = all(_within_map_bounds(env, point) for point in trajectory.position)
    max_deviation = float(deviations[max_index]) if len(deviations) else 0.0
    mean_deviation = float(np.mean(deviations)) if len(deviations) else 0.0
    speed_limit = env.config.max_speed if max_speed is None else float(max_speed)
    horizontal_speed_limit = min(env.config.max_horizontal_speed, speed_limit)
    climb_speed_limit = min(env.config.max_climb_speed, speed_limit)
    descent_speed_limit = min(env.config.max_descent_speed, speed_limit)
    climb_angle_limit = env.config.max_climb_angle_deg
    acceleration_limit = env.config.max_acceleration if max_acceleration is None else float(max_acceleration)
    deceleration_limit = env.config.max_deceleration
    jerk_limit = env.config.max_jerk if max_jerk is None else float(max_jerk)
    if min(
        speed_limit,
        horizontal_speed_limit,
        climb_speed_limit,
        descent_speed_limit,
        acceleration_limit,
        deceleration_limit,
        jerk_limit,
    ) <= 0.0:
        raise ValueError("Dynamics limits must be positive.")

    if len(trajectory.time) > 1:
        dt = np.diff(trajectory.time)
        if np.any(dt <= 0.0):
            raise ValueError("Trajectory time values must be strictly increasing.")
        expected_position_delta = 0.5 * (trajectory.velocity[:-1] + trajectory.velocity[1:]) * dt[:, None]
        position_errors = np.linalg.norm(np.diff(trajectory.position, axis=0) - expected_position_delta, axis=1)
        expected_velocity_delta = trajectory.acceleration[1:] * dt[:, None]
        velocity_errors = np.linalg.norm(np.diff(trajectory.velocity, axis=0) - expected_velocity_delta, axis=1)
        jerk_values = np.linalg.norm(np.diff(trajectory.acceleration, axis=0) / dt[:, None], axis=1)
        deceleration_values = -np.diff(trajectory.speed) / dt
    else:
        position_errors = np.zeros(1, dtype=np.float64)
        velocity_errors = np.zeros(1, dtype=np.float64)
        jerk_values = np.zeros(1, dtype=np.float64)
        deceleration_values = np.zeros(1, dtype=np.float64)
    max_position_error = float(position_errors.max(initial=0.0))
    max_velocity_error = float(velocity_errors.max(initial=0.0))
    observed_max_jerk = float(jerk_values.max(initial=0.0))
    observed_max_deceleration = float(deceleration_values.max(initial=0.0))
    horizontal_speeds = np.linalg.norm(trajectory.velocity[:, :2], axis=1)
    climb_speeds = np.maximum(trajectory.velocity[:, 2], 0.0)
    descent_speeds = np.maximum(-trajectory.velocity[:, 2], 0.0)
    climb_angles = np.degrees(np.arctan2(climb_speeds, horizontal_speeds))
    observed_max_horizontal_speed = float(horizontal_speeds.max(initial=0.0))
    observed_max_climb_speed = float(climb_speeds.max(initial=0.0))
    observed_max_descent_speed = float(descent_speeds.max(initial=0.0))
    observed_max_climb_angle = float(climb_angles.max(initial=0.0))
    dynamics_consistent = max(max_position_error, max_velocity_error) <= integration_tolerance
    speed_ok = trajectory.max_speed <= speed_limit + integration_tolerance
    horizontal_speed_ok = observed_max_horizontal_speed <= horizontal_speed_limit + integration_tolerance
    climb_speed_ok = observed_max_climb_speed <= climb_speed_limit + integration_tolerance
    descent_speed_ok = observed_max_descent_speed <= descent_speed_limit + integration_tolerance
    climb_angle_ok = observed_max_climb_angle <= climb_angle_limit + integration_tolerance
    acceleration_ok = trajectory.max_acceleration <= acceleration_limit + integration_tolerance
    deceleration_ok = observed_max_deceleration <= deceleration_limit + integration_tolerance
    jerk_ok = observed_max_jerk <= jerk_limit + integration_tolerance

    goal_point = env.goal if goal is None else np.asarray(goal, dtype=np.float64)
    radius = env.config.goal_radius if goal_radius is None else float(goal_radius)
    speed_tolerance = (
        env.config.goal_speed_tolerance
        if goal_speed_tolerance is None
        else float(goal_speed_tolerance)
    )
    final_goal_distance = float(np.linalg.norm(trajectory.position[-1] - goal_point))
    final_speed = float(trajectory.speed[-1])
    goal_reached = final_goal_distance <= radius and final_speed <= speed_tolerance
    passed = all(
        (
            max_deviation <= deviation_tolerance,
            collision_free,
            within_bounds,
            speed_ok,
            horizontal_speed_ok,
            climb_speed_ok,
            descent_speed_ok,
            climb_angle_ok,
            acceleration_ok,
            deceleration_ok,
            jerk_ok,
            dynamics_consistent,
            goal_reached,
        )
    )

    return TrajectoryValidationResult(
        passed=bool(passed),
        deviation_tolerance=float(deviation_tolerance),
        max_deviation=max_deviation,
        mean_deviation=mean_deviation,
        max_deviation_index=max_index,
        max_deviation_time=float(trajectory.time[max_index]) if len(trajectory.time) else 0.0,
        collision_free=bool(collision_free),
        within_bounds=bool(within_bounds),
        max_speed=float(trajectory.max_speed),
        max_horizontal_speed=observed_max_horizontal_speed,
        max_climb_speed=observed_max_climb_speed,
        max_descent_speed=observed_max_descent_speed,
        max_climb_angle_deg=observed_max_climb_angle,
        max_acceleration=float(trajectory.max_acceleration),
        max_deceleration=observed_max_deceleration,
        max_jerk=observed_max_jerk,
        speed_limit_satisfied=bool(speed_ok),
        horizontal_speed_limit_satisfied=bool(horizontal_speed_ok),
        climb_speed_limit_satisfied=bool(climb_speed_ok),
        descent_speed_limit_satisfied=bool(descent_speed_ok),
        climb_angle_limit_satisfied=bool(climb_angle_ok),
        acceleration_limit_satisfied=bool(acceleration_ok),
        deceleration_limit_satisfied=bool(deceleration_ok),
        jerk_limit_satisfied=bool(jerk_ok),
        dynamics_consistent=bool(dynamics_consistent),
        max_position_integration_error=max_position_error,
        max_velocity_integration_error=max_velocity_error,
        goal_reached=bool(goal_reached),
        final_goal_distance=final_goal_distance,
        final_speed=final_speed,
        sample_count=int(len(trajectory.time)),
        total_time=float(trajectory.total_time),
        total_length=float(trajectory.total_length),
    )


def save_validation_json(result: TrajectoryValidationResult, output_path: Path) -> None:
    """Save trajectory validation metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved trajectory validation json to {output_path}")
