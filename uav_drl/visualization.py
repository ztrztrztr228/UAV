# -*- coding: utf-8 -*-
"""训练曲线、三维轨迹图和轨迹文件保存。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import BoxObstacle
from .environment import UAVPathPlanningEnv
from .training import TrainHistory
from .trajectory import TimedTrajectory
from .validation import TrajectoryValidationResult


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """计算滑动平均，让训练曲线更平滑。"""
    if len(values) == 0:
        return np.asarray([], dtype=np.float32)
    window = min(len(values), max(1, int(window)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(np.asarray(values, dtype=np.float32), kernel, mode="valid")


def plot_training(history: TrainHistory, output_path: Path) -> None:
    """绘制训练曲线，包括奖励、成功率、最终距离和 DQN loss。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip training plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(history.episode_rewards, alpha=0.35, label="episode reward")
    reward_window = min(30, max(1, len(history.episode_rewards)))
    smoothed = moving_average(history.episode_rewards, window=reward_window)
    if len(smoothed):
        x_values = np.arange(reward_window - 1, reward_window - 1 + len(smoothed))
        axes[0].plot(x_values, smoothed, label="ma30")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].legend()

    success_window = min(50, max(1, len(history.successes)))
    success_ma = moving_average([float(x) for x in history.successes], window=success_window)
    if len(success_ma):
        x_values = np.arange(success_window - 1, success_window - 1 + len(success_ma))
        axes[1].plot(x_values, success_ma)
    axes[1].set_title("Success Rate (MA50)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Episode")

    axes[2].plot(history.final_distances)
    axes[2].set_title("Final Distance To Goal")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("meters")

    axes[3].plot(history.losses)
    axes[3].set_title("DQN Loss")
    axes[3].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training plot to {output_path}")


def _box_faces(obstacle: BoxObstacle) -> list[list[tuple[float, float, float]]]:
    """返回长方体的 6 个面，用于 matplotlib 3D 绘图。"""
    x0, y0, z0 = obstacle.xmin, obstacle.ymin, obstacle.zmin
    x1, y1, z1 = obstacle.xmax, obstacle.ymax, obstacle.zmax
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]


def plot_trajectory(
    env: UAVPathPlanningEnv,
    trajectory: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    """绘制三维小区地图、长方体建筑物、起点、目标点和飞行轨迹。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib is not installed; skip trajectory plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D UAV DRL Path Planning")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(env.config.map_x_min, env.config.map_x_min + env.config.map_width)
    ax.set_ylim(env.config.map_y_min, env.config.map_y_min + env.config.map_height)
    ax.set_zlim(0, env.config.map_altitude)

    for obstacle in env.config.obstacles:
        poly = Poly3DCollection(
            _box_faces(obstacle),
            facecolor="#4a5568",
            edgecolor="#1a202c",
            linewidths=0.5,
            alpha=0.55,
        )
        ax.add_collection3d(poly)
        ax.text(
            (obstacle.xmin + obstacle.xmax) / 2,
            (obstacle.ymin + obstacle.ymax) / 2,
            obstacle.zmax + 0.4,
            obstacle.name.split("_", 1)[0],
            ha="center",
            va="bottom",
            fontsize=7,
        )

    if trajectory:
        points = np.asarray(trajectory, dtype=np.float32)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#2563eb", linewidth=2.0, label="trajectory")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#2563eb", s=8, alpha=0.45)
        start = points[0]
    else:
        start = env.start

    ax.scatter([start[0]], [start[1]], [start[2]], color="#16a34a", s=80, marker="o", label="start")
    ax.scatter([env.goal[0]], [env.goal[1]], [env.goal[2]], color="#dc2626", s=110, marker="*", label="goal")
    ax.legend(loc="upper right")
    ax.view_init(elev=24, azim=-55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {output_path}")


def save_trajectory_csv(trajectory: Sequence[np.ndarray], output_path: Path) -> None:
    """把三维轨迹保存为 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "z"])
        for step, point in enumerate(trajectory):
            writer.writerow([step, float(point[0]), float(point[1]), float(point[2])])
    print(f"Saved trajectory csv to {output_path}")


def save_timed_trajectory_csv(trajectory: TimedTrajectory, output_path: Path) -> None:
    """Save equal-time trajectory samples with velocity and acceleration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "time_s",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "speed",
                "ax",
                "ay",
                "az",
                "acceleration",
                "path_s",
            ]
        )
        for index, t in enumerate(trajectory.time):
            position = trajectory.position[index]
            velocity = trajectory.velocity[index]
            acceleration = trajectory.acceleration[index]
            writer.writerow(
                [
                    index,
                    float(t),
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(velocity[0]),
                    float(velocity[1]),
                    float(velocity[2]),
                    float(trajectory.speed[index]),
                    float(acceleration[0]),
                    float(acceleration[1]),
                    float(acceleration[2]),
                    float(trajectory.acceleration_norm[index]),
                    float(trajectory.path_s[index]),
                ]
            )
    print(f"Saved timed trajectory csv to {output_path}")


def plot_trajectory_profiles(trajectory: TimedTrajectory, output_path: Path) -> None:
    """Plot speed and acceleration over trajectory time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip trajectory profile plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(trajectory.time, trajectory.speed, color="#2563eb", label="speed")
    axes[0].axhline(trajectory.max_speed, color="#94a3b8", linestyle="--", linewidth=1.0, label="observed max")
    axes[0].set_ylabel("m/s")
    axes[0].set_title("Trajectory Speed")
    axes[0].legend()

    axes[1].plot(trajectory.time, trajectory.acceleration_norm, color="#dc2626", label="acceleration")
    axes[1].axhline(
        trajectory.max_acceleration,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.0,
        label="observed max",
    )
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("m/s^2")
    axes[1].set_title("Trajectory Acceleration")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory profile plot to {output_path}")


def plot_trajectory_validation(
    env: UAVPathPlanningEnv,
    reference_path: Sequence[np.ndarray],
    trajectory: TimedTrajectory,
    validation: TrajectoryValidationResult,
    output_path: Path,
) -> None:
    """Plot original path and timed trajectory together for visual validation."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("matplotlib is not installed; skip trajectory validation plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_points = np.asarray(reference_path, dtype=np.float64)
    timed_points = np.asarray(trajectory.position, dtype=np.float64)
    max_point = timed_points[validation.max_deviation_index]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        "Trajectory Validation "
        f"({'PASS' if validation.passed else 'FAIL'}, max error {validation.max_deviation:.2f} m)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(env.config.map_x_min, env.config.map_x_min + env.config.map_width)
    ax.set_ylim(env.config.map_y_min, env.config.map_y_min + env.config.map_height)
    ax.set_zlim(0, env.config.map_altitude)

    for obstacle in env.config.obstacles:
        poly = Poly3DCollection(
            _box_faces(obstacle),
            facecolor="#64748b",
            edgecolor="#334155",
            linewidths=0.5,
            alpha=0.38,
        )
        ax.add_collection3d(poly)

    ax.plot(
        reference_points[:, 0],
        reference_points[:, 1],
        reference_points[:, 2],
        color="#2563eb",
        linewidth=2.0,
        label="reference path",
    )
    ax.plot(
        timed_points[:, 0],
        timed_points[:, 1],
        timed_points[:, 2],
        color="#dc2626",
        linewidth=1.8,
        label="timed trajectory",
    )
    ax.scatter(
        timed_points[:, 0],
        timed_points[:, 1],
        timed_points[:, 2],
        color="#f97316",
        s=18,
        alpha=0.75,
        label="time samples",
    )
    ax.scatter(
        [max_point[0]],
        [max_point[1]],
        [max_point[2]],
        color="#111827",
        s=80,
        marker="x",
        label="max deviation",
    )
    ax.scatter(
        [reference_points[0, 0]],
        [reference_points[0, 1]],
        [reference_points[0, 2]],
        color="#16a34a",
        s=70,
        marker="o",
        label="start",
    )
    ax.scatter(
        [env.goal[0]],
        [env.goal[1]],
        [env.goal[2]],
        color="#7c3aed",
        s=95,
        marker="*",
        label="goal",
    )
    ax.legend(loc="upper right")
    ax.view_init(elev=24, azim=-55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory validation plot to {output_path}")
