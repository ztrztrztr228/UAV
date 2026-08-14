# -*- coding: utf-8 -*-
"""Convert planned path points into an equal-time UAV trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimedTrajectory:
    """Trajectory samples with position, velocity, and acceleration."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    speed: np.ndarray
    acceleration_norm: np.ndarray
    path_s: np.ndarray
    total_time: float
    total_length: float
    max_speed: float
    max_acceleration: float


def dynamics_samples_to_trajectory(
    positions: list[np.ndarray] | np.ndarray,
    velocities: list[np.ndarray] | np.ndarray,
    accelerations: list[np.ndarray] | np.ndarray,
    dt: float,
) -> TimedTrajectory:
    """Build a trajectory directly from RL dynamics samples without smoothing."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    position = _as_points(positions)
    velocity = _as_points(velocities)
    acceleration = _as_points(accelerations)
    if not (len(position) == len(velocity) == len(acceleration)):
        raise ValueError("Position, velocity, and acceleration sample counts must match.")
    time = np.arange(len(position), dtype=np.float64) * float(dt)
    segment_lengths = (
        np.linalg.norm(np.diff(position, axis=0), axis=1)
        if len(position) > 1
        else np.asarray([], dtype=np.float64)
    )
    path_s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    return TimedTrajectory(
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        speed=speed,
        acceleration_norm=acceleration_norm,
        path_s=path_s,
        total_time=float(time[-1]) if len(time) else 0.0,
        total_length=float(path_s[-1]),
        max_speed=float(speed.max(initial=0.0)),
        max_acceleration=float(acceleration_norm.max(initial=0.0)),
    )


def _as_points(path: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray) -> np.ndarray:
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Path must be an array-like sequence of 3D points.")
    if len(points) == 0:
        raise ValueError("Path must contain at least one point.")
    return points


def smooth_path(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Smooth a polyline with Chaikin corner cutting while preserving endpoints."""
    smoothed = _as_points(points)
    iterations = max(0, int(iterations))
    for _ in range(iterations):
        if len(smoothed) < 3:
            break
        next_points = [smoothed[0]]
        for start, end in zip(smoothed[:-1], smoothed[1:]):
            next_points.append(0.75 * start + 0.25 * end)
            next_points.append(0.25 * start + 0.75 * end)
        next_points.append(smoothed[-1])
        smoothed = np.asarray(next_points, dtype=np.float64)
    return smoothed


def _remove_duplicate_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-9])
    return points[keep]


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.asarray([0.0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _minimum_duration(length: float, max_speed: float, max_acceleration: float) -> float:
    if length <= 1e-9:
        return 0.0
    time_to_max_speed = max_speed / max_acceleration
    accel_distance = 0.5 * max_acceleration * time_to_max_speed**2
    if 2.0 * accel_distance >= length:
        return 2.0 * np.sqrt(length / max_acceleration)
    cruise_distance = length - 2.0 * accel_distance
    return 2.0 * time_to_max_speed + cruise_distance / max_speed


def _trapezoid_s(
    time_values: np.ndarray,
    length: float,
    duration: float,
    max_speed: float,
    max_acceleration: float,
) -> np.ndarray:
    if length <= 1e-9:
        return np.zeros_like(time_values, dtype=np.float64)

    time_to_max_speed = max_speed / max_acceleration
    accel_distance = 0.5 * max_acceleration * time_to_max_speed**2

    if 2.0 * accel_distance >= length:
        peak_speed = np.sqrt(length * max_acceleration)
        accel_time = peak_speed / max_acceleration
        total_time = 2.0 * accel_time
        scale = duration / max(total_time, 1e-9)
        effective_acceleration = max_acceleration / (scale * scale)
        accel_time *= scale
        total_time = duration
        s_values = np.empty_like(time_values, dtype=np.float64)
        for i, t in enumerate(time_values):
            if t <= accel_time:
                s_values[i] = 0.5 * effective_acceleration * t**2
            else:
                remaining = max(0.0, total_time - t)
                s_values[i] = length - 0.5 * effective_acceleration * remaining**2
        return np.clip(s_values, 0.0, length)

    cruise_distance = length - 2.0 * accel_distance
    cruise_time = cruise_distance / max_speed
    total_time = 2.0 * time_to_max_speed + cruise_time
    scale = duration / max(total_time, 1e-9)
    effective_acceleration = max_acceleration / (scale * scale)
    effective_max_speed = max_speed / scale
    accel_time = time_to_max_speed * scale
    cruise_time *= scale
    decel_start = accel_time + cruise_time
    accel_distance = 0.5 * effective_acceleration * accel_time**2

    s_values = np.empty_like(time_values, dtype=np.float64)
    for i, t in enumerate(time_values):
        if t <= accel_time:
            s_values[i] = 0.5 * effective_acceleration * t**2
        elif t <= decel_start:
            s_values[i] = accel_distance + effective_max_speed * (t - accel_time)
        else:
            remaining = max(0.0, duration - t)
            s_values[i] = length - 0.5 * effective_acceleration * remaining**2
    return np.clip(s_values, 0.0, length)


def _sample_polyline(points: np.ndarray, cumulative_s: np.ndarray, sample_s: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points, len(sample_s), axis=0)
    x = np.interp(sample_s, cumulative_s, points[:, 0])
    y = np.interp(sample_s, cumulative_s, points[:, 1])
    z = np.interp(sample_s, cumulative_s, points[:, 2])
    return np.column_stack([x, y, z])


def _time_grid(duration: float, dt: float) -> np.ndarray:
    duration = max(float(dt), float(np.ceil(duration / dt) * dt))
    count = int(round(duration / dt)) + 1
    return np.arange(count, dtype=np.float64) * dt


def path_to_timed_trajectory(
    path: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
    dt: float = 1.0,
    max_speed: float = 23.18,
    max_acceleration: float = 3.0,
    smoothing_iterations: int = 1,
) -> TimedTrajectory:
    """Generate equal-time trajectory samples from a geometric path."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive.")
    if max_acceleration <= 0.0:
        raise ValueError("max_acceleration must be positive.")

    points = _remove_duplicate_points(_as_points(path))
    points = smooth_path(points, iterations=smoothing_iterations)
    points = _remove_duplicate_points(points)
    cumulative_s = _arc_lengths(points)
    total_length = float(cumulative_s[-1])
    minimum_duration = _minimum_duration(total_length, max_speed, max_acceleration)
    duration = max(float(dt), minimum_duration)

    # Increase duration until finite-difference derivatives respect requested limits.
    for _ in range(8):
        time_values = _time_grid(duration, dt)
        duration = float(time_values[-1])
        sample_count = len(time_values)
        sample_s = _trapezoid_s(time_values, total_length, duration, max_speed, max_acceleration)
        positions = _sample_polyline(points, cumulative_s, sample_s)
        if sample_count >= 3:
            velocities = np.gradient(positions, time_values, axis=0, edge_order=2)
            accelerations = np.gradient(velocities, time_values, axis=0, edge_order=2)
        else:
            velocities = np.zeros_like(positions)
            accelerations = np.zeros_like(positions)
        speeds = np.linalg.norm(velocities, axis=1)
        acceleration_norms = np.linalg.norm(accelerations, axis=1)
        if speeds.max(initial=0.0) <= max_speed * 1.02 and acceleration_norms.max(initial=0.0) <= max_acceleration * 1.02:
            break
        duration *= 1.2

    return TimedTrajectory(
        time=time_values,
        position=positions,
        velocity=velocities,
        acceleration=accelerations,
        speed=speeds,
        acceleration_norm=acceleration_norms,
        path_s=sample_s,
        total_time=float(time_values[-1]),
        total_length=total_length,
        max_speed=float(speeds.max(initial=0.0)),
        max_acceleration=float(acceleration_norms.max(initial=0.0)),
    )
