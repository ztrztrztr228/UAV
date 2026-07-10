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
    max_acceleration: float
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
    return bool(
        radius <= point[0] <= env.config.map_width - radius
        and radius <= point[1] <= env.config.map_height - radius
        and radius <= point[2] <= env.config.map_altitude - radius
    )


def _trajectory_collision_free(env: UAVPathPlanningEnv, positions: np.ndarray) -> bool:
    for point in positions:
        if env._point_in_collision(point):
            return False
    for start, end in zip(positions[:-1], positions[1:]):
        if env._segment_in_collision(start, end):
            return False
    return True


def validate_timed_trajectory(
    env: UAVPathPlanningEnv,
    reference_path: Sequence[np.ndarray] | np.ndarray,
    trajectory: TimedTrajectory,
    deviation_tolerance: float,
) -> TrajectoryValidationResult:
    """Validate that equal-time trajectory samples stay close to the planned path."""
    if deviation_tolerance < 0.0:
        raise ValueError("deviation_tolerance must be non-negative.")

    deviations = trajectory_deviations(reference_path, trajectory)
    max_index = int(np.argmax(deviations)) if len(deviations) else 0
    collision_free = _trajectory_collision_free(env, trajectory.position)
    within_bounds = all(_within_map_bounds(env, point) for point in trajectory.position)
    max_deviation = float(deviations[max_index]) if len(deviations) else 0.0
    mean_deviation = float(np.mean(deviations)) if len(deviations) else 0.0
    passed = max_deviation <= deviation_tolerance and collision_free and within_bounds

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
        max_acceleration=float(trajectory.max_acceleration),
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
