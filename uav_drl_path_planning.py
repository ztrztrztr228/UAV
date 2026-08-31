# -*- coding: utf-8 -*-
"""无人机三维深度强化学习轨迹规划主入口。

具体实现位于 uav_drl/ 目录中。本文件负责命令行参数、创建环境/智能体、
训练、评估和保存结果。

快速测试：
    python uav_drl_path_planning.py --episodes 3 --eval-episodes 1 --no-plots

正式训练并绘制三维轨迹：
    python uav_drl_path_planning.py --episodes 800 --visualize

加载模型并输入指定三维目标点：
    python uav_drl_path_planning.py --skip-train --load-model outputs/uav_dqn.pt \
        --start-x 5 --start-y 5 --start-z 8 --target-x 92 --target-y 88 --target-z 12 --visualize
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from uav_drl.actions import ACTION_NAMES
from uav_drl.agent import DQNAgent, device_description
from uav_drl.config import DEFAULT_SEED, UAVEnvConfig, config_to_dict
from uav_drl.environment import UAVPathPlanningEnv
from uav_drl.qgc_plan import (
    FIRMWARE_TYPES,
    build_qgc_plan,
    prepare_qgc_waypoints,
    qgc_autoload_path,
    save_qgc_plan,
)
from uav_drl.scenes import available_scene_keys, get_training_scene
from uav_drl.trajectory import TimedTrajectory
from uav_drl.training import TrainHistory, evaluate_agent, train_dqn
from uav_drl.utils import fix_seed, optional_point_3d
from uav_drl.validation import (
    TrajectoryValidationResult,
    save_validation_json,
    validate_timed_trajectory,
)
from uav_drl.visualization import (
    plot_training,
    plot_trajectory,
    plot_trajectory_validation,
    plot_trajectory_profiles,
    save_timed_trajectory_csv,
    save_trajectory_csv,
)


# ==================== 命令行参数定义与校验 ====================
def parse_args() -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a dynamics-aware 3D DQN UAV trajectory planner.")

    # 运行模式、评估目标分布、随机种子和场景选择。
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--eval-goal-mode",
        choices=("match-training", "hard", "stress"),
        default="match-training",
        help=(
            "evaluation goal tier: standard training difficulty (default), the final "
            "hard curriculum tier, or the legacy unrestricted stress distribution"
        ),
    )
    parser.add_argument(
        "--eval-near-obstacle-probability",
        type=float,
        help="override the selected evaluation tier's near-building goal fraction",
    )
    parser.add_argument("--eval-near-obstacle-min-clearance", type=float)
    parser.add_argument("--eval-near-obstacle-max-clearance", type=float)
    parser.add_argument("--eval-goal-max-distance", type=float)
    parser.add_argument("--eval-goal-min-altitude", type=float)
    parser.add_argument("--eval-goal-max-altitude", type=float)
    parser.add_argument(
        "--eval-goal-side-clearance-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override whether near-building evaluation goals use horizontal side clearance",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scene", choices=available_scene_keys(), default="wujing_airfield")
    parser.add_argument("--list-scenes", action="store_true")

    # 地图范围、障碍物外扩以及无人机离散时间动力学约束。
    parser.add_argument("--map-x-min", type=float)
    parser.add_argument("--map-y-min", type=float)
    parser.add_argument("--map-width", type=float)
    parser.add_argument("--map-height", type=float)
    parser.add_argument("--map-altitude", type=float)
    parser.add_argument("--obstacle-inflation", type=float)
    parser.add_argument("--step-length", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--goal-radius", type=float, default=3.0)
    parser.add_argument("--default-altitude", type=float, default=8.0)
    parser.add_argument("--trajectory-dt", type=float, default=0.5)
    parser.add_argument("--max-horizontal-speed", type=float, default=23.0)
    parser.add_argument("--max-speed", type=float, default=23.18, help="maximum combined 3D speed")
    parser.add_argument("--max-climb-speed", type=float, default=2.85)
    parser.add_argument("--max-descent-speed", type=float, default=1.65)
    parser.add_argument("--max-climb-angle", type=float, default=90.0, help="degrees")
    parser.add_argument("--max-acceleration", type=float, default=15.5, help="measured peak acceleration")
    parser.add_argument(
        "--normal-acceleration",
        type=float,
        default=3.0,
        help="normal-flight acceleration command limit (conservative end of the measured 3-5 m/s^2 range)",
    )
    parser.add_argument("--max-deceleration", type=float, default=3.09)
    parser.add_argument("--max-jerk", type=float, default=78.0, help="smoothed jerk limit")
    parser.add_argument("--raw-max-jerk", type=float, default=142.0, help="raw-log peak for reference")
    parser.add_argument("--goal-speed-tolerance", type=float, default=1.0)
    parser.add_argument("--smoothing-iterations", type=int, default=1)

    # 轨迹形状奖励和导出轨迹与参考路径间的允许偏差。
    reward_group = parser.add_argument_group("trajectory reward shaping")
    reward_group.add_argument("--extra-altitude-penalty-scale", type=float, default=0.12)
    reward_group.add_argument("--extra-altitude-margin", type=float, default=3.0)
    reward_group.add_argument("--detour-penalty-scale", type=float, default=0.35)
    reward_group.add_argument("--turn-penalty-scale", type=float, default=0.20)
    reward_group.add_argument("--turn-speed-threshold", type=float, default=0.50)
    reward_group.add_argument("--goal-guidance-distance", type=float, default=60.0)
    reward_group.add_argument("--goal-altitude-penalty-scale", type=float, default=0.08)
    reward_group.add_argument("--vertical-speed-guidance-scale", type=float, default=0.30)
    reward_group.add_argument("--vertical-guidance-time", type=float, default=4.0)
    parser.add_argument(
        "--trajectory-deviation-tolerance",
        type=float,
        default=None,
        help="maximum allowed distance from timed trajectory samples to the original path",
    )

    # 用户可选的固定三维起点和目标点；留空时由环境采样。
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-z", type=float)
    parser.add_argument("--target-x", type=float)
    parser.add_argument("--target-y", type=float)
    parser.add_argument("--target-z", type=float)

    # DQN 超参数、目标课程学习、安全动作屏蔽和周期验证设置。
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--target-update-interval", type=int, default=300)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=500.0)
    parser.add_argument(
        "--train-near-obstacle-probability",
        type=float,
        default=None,
        help="final fraction of training goals near buildings; default depends on scene",
    )
    parser.add_argument("--train-near-obstacle-min-clearance", type=float)
    parser.add_argument("--train-near-obstacle-max-clearance", type=float)
    parser.add_argument("--train-near-obstacle-start-probability", type=float)
    parser.add_argument("--train-near-obstacle-start-min-clearance", type=float)
    parser.add_argument("--train-near-obstacle-start-max-clearance", type=float)
    parser.add_argument("--train-near-obstacle-curriculum-episodes", type=int)
    parser.add_argument("--train-near-obstacle-hard-probability", type=float)
    parser.add_argument("--train-near-obstacle-hard-min-clearance", type=float)
    parser.add_argument("--train-near-obstacle-hard-max-clearance", type=float)
    parser.add_argument("--train-near-obstacle-hardening-episodes", type=int)
    parser.add_argument("--train-goal-start-max-distance", type=float)
    parser.add_argument("--train-goal-final-max-distance", type=float)
    parser.add_argument("--train-goal-hard-max-distance", type=float)
    parser.add_argument("--train-goal-start-min-altitude", type=float)
    parser.add_argument("--train-goal-start-max-altitude", type=float)
    parser.add_argument("--train-goal-final-min-altitude", type=float)
    parser.add_argument("--train-goal-final-max-altitude", type=float)
    parser.add_argument("--train-goal-hard-min-altitude", type=float)
    parser.add_argument("--train-goal-hard-max-altitude", type=float)
    parser.add_argument(
        "--train-goal-side-clearance-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="sample building-near goals by horizontal side clearance, excluding rooftop-only goals",
    )
    parser.add_argument(
        "--disable-safe-action-mask",
        action="store_true",
        help="disable collision/braking-aware action masking",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1_000,
        help="run fixed epsilon=0 validation every N completed training episodes; 0 disables",
    )
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--migration-epsilon-start", type=float, default=0.05)
    parser.add_argument(
        "--keep-legacy-replay",
        action="store_true",
        help="keep old replay when appending state features (not recommended)",
    )

    # 输出目录、模型加载/保存、绘图和计算设备。
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, help="override this scene's run-results directory")
    parser.add_argument("--save-model", type=Path, help="override this scene's checkpoint path")
    parser.add_argument("--save-best-model", type=Path, help="override pure-policy best checkpoint path")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--fresh-start", action="store_true", help="do not auto-load the previous checkpoint")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        help="compute device: cpu (default), auto, cuda, or cuda:<index>",
    )

    # 只有显式启用时才使用的 QGroundControl 任务导出参数。
    qgc = parser.add_argument_group("QGroundControl Plan export")
    qgc.add_argument(
        "--export-qgc-plan",
        action="store_true",
        help="export a validated trajectory as qgc_mission.plan",
    )
    qgc.add_argument("--qgc-origin-lat-wgs84", type=float, help="WGS-84 latitude of local ENU (0,0,0)")
    qgc.add_argument("--qgc-origin-lon-wgs84", type=float, help="WGS-84 longitude of local ENU (0,0,0)")
    qgc.add_argument("--qgc-origin-alt-amsl", type=float, help="AMSL altitude in metres of local ENU z=0")
    qgc.add_argument("--qgc-firmware", choices=tuple(FIRMWARE_TYPES), default="px4")
    qgc.add_argument("--qgc-system-id", type=int, default=1, help="vehicle MAVLink system id for AutoLoad#.plan")
    qgc.add_argument(
        "--qgc-autoload-dir",
        type=Path,
        help="QGC Application Load/Save Path; writes AutoLoad<system-id>.plan",
    )
    qgc.add_argument("--qgc-takeoff-altitude", type=float, default=5.0)
    qgc.add_argument("--qgc-simplify-tolerance", type=float, default=1.0)
    qgc.add_argument("--qgc-max-segment-length", type=float, default=25.0)
    qgc.add_argument("--qgc-acceptance-radius", type=float, default=2.0)
    qgc.add_argument("--qgc-hover-speed", type=float, default=5.0)
    qgc.add_argument("--qgc-cruise-speed", type=float, default=15.0)
    qgc.add_argument("--qgc-end-action", choices=("none", "rtl", "land"), default="none")

    # 解析后统一检查跨参数约束，避免训练启动后才暴露配置错误。
    args = parser.parse_args()
    if (
        args.eval_near_obstacle_probability is not None
        and not 0.0 <= args.eval_near_obstacle_probability <= 1.0
    ):
        parser.error("--eval-near-obstacle-probability must be in [0, 1]")
    if (
        args.eval_near_obstacle_min_clearance is not None
        and args.eval_near_obstacle_min_clearance < 0.0
    ):
        parser.error("--eval-near-obstacle-min-clearance must be non-negative")
    if (
        args.eval_near_obstacle_max_clearance is not None
        and args.eval_near_obstacle_min_clearance is not None
        and args.eval_near_obstacle_max_clearance < args.eval_near_obstacle_min_clearance
    ):
        parser.error("--eval-near-obstacle-max-clearance must not be below the minimum")
    if args.train_near_obstacle_probability is not None and not 0.0 <= args.train_near_obstacle_probability <= 1.0:
        parser.error("--train-near-obstacle-probability must be in [0, 1]")
    if args.train_near_obstacle_min_clearance is not None and args.train_near_obstacle_min_clearance < 0.0:
        parser.error("--train-near-obstacle-min-clearance must be non-negative")
    if (
        args.train_near_obstacle_max_clearance is not None
        and args.train_near_obstacle_min_clearance is not None
        and args.train_near_obstacle_max_clearance < args.train_near_obstacle_min_clearance
    ):
        parser.error("--train-near-obstacle-max-clearance must not be below the minimum")
    if args.train_near_obstacle_start_probability is not None and not 0.0 <= args.train_near_obstacle_start_probability <= 1.0:
        parser.error("--train-near-obstacle-start-probability must be in [0, 1]")
    if args.train_near_obstacle_start_min_clearance is not None and args.train_near_obstacle_start_min_clearance < 0.0:
        parser.error("--train-near-obstacle-start-min-clearance must be non-negative")
    if (
        args.train_near_obstacle_start_max_clearance is not None
        and args.train_near_obstacle_start_min_clearance is not None
        and args.train_near_obstacle_start_max_clearance < args.train_near_obstacle_start_min_clearance
    ):
        parser.error("--train-near-obstacle-start-max-clearance must not be below the minimum")
    if args.train_near_obstacle_curriculum_episodes is not None and args.train_near_obstacle_curriculum_episodes < 0:
        parser.error("--train-near-obstacle-curriculum-episodes must be non-negative")
    if args.validation_interval < 0:
        parser.error("--validation-interval must be non-negative")
    if args.validation_episodes < 0:
        parser.error("--validation-episodes must be non-negative")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    if not 0.0 <= args.migration_epsilon_start <= 1.0:
        parser.error("--migration-epsilon-start must be in [0, 1]")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        parser.error("epsilon values must satisfy 0 <= epsilon-end <= epsilon-start <= 1")
    if args.epsilon_decay <= 0.0:
        parser.error("--epsilon-decay must be positive")
    if args.qgc_autoload_dir is not None and not args.export_qgc_plan:
        parser.error("--qgc-autoload-dir requires --export-qgc-plan")
    if args.export_qgc_plan:
        required = {
            "--qgc-origin-lat-wgs84": args.qgc_origin_lat_wgs84,
            "--qgc-origin-lon-wgs84": args.qgc_origin_lon_wgs84,
            "--qgc-origin-alt-amsl": args.qgc_origin_alt_amsl,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("QGC export requires " + ", ".join(missing))
        numeric_values = (
            args.qgc_origin_lat_wgs84,
            args.qgc_origin_lon_wgs84,
            args.qgc_origin_alt_amsl,
            args.qgc_takeoff_altitude,
            args.qgc_simplify_tolerance,
            args.qgc_max_segment_length,
            args.qgc_acceptance_radius,
            args.qgc_hover_speed,
            args.qgc_cruise_speed,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            parser.error("QGC coordinates and flight parameters must be finite")
        if not -90.0 <= args.qgc_origin_lat_wgs84 <= 90.0:
            parser.error("--qgc-origin-lat-wgs84 must be in [-90, 90]")
        if not -180.0 <= args.qgc_origin_lon_wgs84 <= 180.0:
            parser.error("--qgc-origin-lon-wgs84 must be in [-180, 180]")
        if not 1 <= args.qgc_system_id <= 255:
            parser.error("--qgc-system-id must be in [1, 255]")
        if args.qgc_takeoff_altitude <= 0.0:
            parser.error("--qgc-takeoff-altitude must be positive")
        if args.qgc_simplify_tolerance < 0.0:
            parser.error("--qgc-simplify-tolerance must be non-negative")
        if min(
            args.qgc_max_segment_length,
            args.qgc_acceptance_radius,
            args.qgc_hover_speed,
            args.qgc_cruise_speed,
        ) <= 0.0:
            parser.error("QGC segment length, acceptance radius, and speeds must be positive")
    return args


# ==================== 环境配置组装 ====================
def make_config(args: argparse.Namespace) -> UAVEnvConfig:
    """根据命令行参数生成三维环境配置。"""
    scene = get_training_scene(args.scene)
    return scene.make_config(
        obstacle_inflation=args.obstacle_inflation,
        map_x_min=args.map_x_min,
        map_y_min=args.map_y_min,
        map_width=args.map_width,
        map_height=args.map_height,
        map_altitude=args.map_altitude,
        step_length=args.step_length,
        max_steps=args.max_steps,
        goal_radius=args.goal_radius,
        trajectory_dt=args.trajectory_dt,
        max_horizontal_speed=args.max_horizontal_speed,
        max_speed=args.max_speed,
        max_climb_speed=args.max_climb_speed,
        max_descent_speed=args.max_descent_speed,
        max_climb_angle_deg=args.max_climb_angle,
        max_acceleration=args.max_acceleration,
        normal_acceleration=args.normal_acceleration,
        max_deceleration=args.max_deceleration,
        max_jerk=args.max_jerk,
        raw_max_jerk=args.raw_max_jerk,
        goal_speed_tolerance=args.goal_speed_tolerance,
        smoothing_iterations=args.smoothing_iterations,
        extra_altitude_penalty_scale=args.extra_altitude_penalty_scale,
        extra_altitude_margin=args.extra_altitude_margin,
        detour_penalty_scale=args.detour_penalty_scale,
        turn_penalty_scale=args.turn_penalty_scale,
        turn_speed_threshold=args.turn_speed_threshold,
        goal_guidance_distance=args.goal_guidance_distance,
        goal_altitude_penalty_scale=args.goal_altitude_penalty_scale,
        vertical_speed_guidance_scale=args.vertical_speed_guidance_scale,
        vertical_guidance_time=args.vertical_guidance_time,
    )


# ==================== 运行目录和随机状态管理 ====================
def make_run_output_dir(base_dir: Path) -> Path:
  #每次单独保存输出结果
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def capture_rng_state() -> dict[str, object]:
    """Capture random-generator state for exact checkpoint resume."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    """Restore random-generator state from a checkpoint when available."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [
            value.cpu() if isinstance(value, torch.Tensor) else value
            for value in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


# ==================== QGroundControl 导出编排 ====================
def _mission_polyline_is_collision_free(env: UAVPathPlanningEnv, points: np.ndarray) -> bool:
    """Check the straight segments that the autopilot will fly between QGC items."""
    return all(
        not env._segment_in_collision(start, end)
        for start, end in zip(points[:-1], points[1:])
    )


def export_qgc_outputs(
    args: argparse.Namespace,
    env: UAVPathPlanningEnv,
    trajectory: TimedTrajectory,
    validation: TrajectoryValidationResult,
    run_output_dir: Path,
) -> list[Path]:
    """Export a validated trajectory to the run directory and optional QGC AutoLoad path."""
    # 未通过完整轨迹验证时禁止生成可被地面站加载的任务文件。
    if not validation.passed:
        print("Skipped QGC Plan export because trajectory validation did not pass.")
        return []

    # 提取空中段并抽稀，减少飞控需要执行的航点数量。
    takeoff_point, waypoints = prepare_qgc_waypoints(
        trajectory.position,
        takeoff_altitude_m=args.qgc_takeoff_altitude,
        simplification_tolerance_m=args.qgc_simplify_tolerance,
        max_segment_length_m=args.qgc_max_segment_length,
    )

    def mission_polyline_points() -> np.ndarray:
        points = np.vstack([trajectory.position[0], takeoff_point, waypoints])
        if args.qgc_end_action == "land":
            landing_point = waypoints[-1].copy()
            landing_point[2] = env.config.uav_radius + 1e-3
            points = np.vstack([points, landing_point])
        return points

    # QGC 会在相邻任务点间直线飞行，因此必须对抽稀后的折线重新检查碰撞。
    mission_polyline = mission_polyline_points()
    if not _mission_polyline_is_collision_free(env, mission_polyline):
        print("Simplified QGC path was not collision-free; retrying without geometric tolerance.")
        takeoff_point, waypoints = prepare_qgc_waypoints(
            trajectory.position,
            takeoff_altitude_m=args.qgc_takeoff_altitude,
            simplification_tolerance_m=0.0,
            max_segment_length_m=args.qgc_max_segment_length,
        )
        mission_polyline = mission_polyline_points()
        if not _mission_polyline_is_collision_free(env, mission_polyline):
            raise ValueError(
                "QGC mission export refused: straight waypoint segments intersect an obstacle or map boundary."
            )

    # 将本地 ENU 航点转换为 WGS-84，并组装 QGC Plan JSON。
    plan = build_qgc_plan(
        home_point_enu=trajectory.position[0],
        takeoff_point_enu=takeoff_point,
        waypoint_points_enu=waypoints,
        origin_latitude_wgs84=args.qgc_origin_lat_wgs84,
        origin_longitude_wgs84=args.qgc_origin_lon_wgs84,
        origin_altitude_amsl_m=args.qgc_origin_alt_amsl,
        firmware=args.qgc_firmware,
        hover_speed_m_s=args.qgc_hover_speed,
        cruise_speed_m_s=args.qgc_cruise_speed,
        acceptance_radius_m=args.qgc_acceptance_radius,
        end_action=args.qgc_end_action,
    )

    # 始终保存运行目录副本；提供 AutoLoad 目录时再同步一份给 QGC。
    output_paths = [save_qgc_plan(plan, run_output_dir / "qgc_mission.plan")]
    print(
        f"Saved QGC Plan to {output_paths[0]} "
        f"({len(waypoints)} waypoints + takeoff, original samples={len(trajectory.position)})"
    )
    if args.qgc_autoload_dir is not None:
        autoload_path = qgc_autoload_path(args.qgc_autoload_dir, args.qgc_system_id)
        output_paths.append(save_qgc_plan(plan, autoload_path))
        print(f"Updated QGC AutoLoad Plan at {autoload_path}")
    return output_paths


# ==================== 分场景课程学习默认值 ====================
def apply_scene_training_defaults(
    args: argparse.Namespace,
    scene_key: str,
    config: UAVEnvConfig,
) -> None:
    """Use a gentler distance/altitude curriculum for fragmented residential maps."""
    residential = scene_key != "wujing_airfield"
    defaults = {
        "train_near_obstacle_probability": 0.50 if residential else 0.70,
        "train_near_obstacle_min_clearance": 5.0 if residential else 2.0,
        "train_near_obstacle_max_clearance": 15.0 if residential else 12.0,
        "train_near_obstacle_start_probability": 0.10 if residential else 0.30,
        "train_near_obstacle_start_min_clearance": 15.0,
        "train_near_obstacle_start_max_clearance": 30.0,
        "train_near_obstacle_curriculum_episodes": 2_000 if residential else 1_500,
        "train_near_obstacle_hard_probability": 0.70,
        "train_near_obstacle_hard_min_clearance": 2.0,
        "train_near_obstacle_hard_max_clearance": 12.0,
        "train_near_obstacle_hardening_episodes": 2_000 if residential else 0,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if not residential:
        if args.train_goal_side_clearance_only is None:
            args.train_goal_side_clearance_only = False
        return

    maximum_distance = float(
        np.linalg.norm([config.map_width, config.map_height, config.map_altitude])
    )
    start_max_distance = {
        "lanxianghu_villa": 250.0,
        "sanming_garden": 200.0,
        "spring_garden_phase2": 200.0,
    }[scene_key]
    altitude_limits = {
        "lanxianghu_villa": (5.0, 15.0, 5.0, 25.0, 5.0, 28.0),
        "sanming_garden": (5.0, 15.0, 5.0, 25.0, 5.0, 35.0),
        "spring_garden_phase2": (5.0, 20.0, 5.0, 40.0, 5.0, 60.0),
    }[scene_key]
    curriculum_defaults = {
        "train_goal_start_max_distance": start_max_distance,
        "train_goal_final_max_distance": maximum_distance,
        "train_goal_hard_max_distance": maximum_distance,
        "train_goal_start_min_altitude": altitude_limits[0],
        "train_goal_start_max_altitude": altitude_limits[1],
        "train_goal_final_min_altitude": altitude_limits[2],
        "train_goal_final_max_altitude": altitude_limits[3],
        "train_goal_hard_min_altitude": altitude_limits[4],
        "train_goal_hard_max_altitude": altitude_limits[5],
    }
    for name, value in curriculum_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.train_goal_side_clearance_only is None:
        args.train_goal_side_clearance_only = True


def validate_scene_training_curriculum(
    args: argparse.Namespace,
    config: UAVEnvConfig,
) -> None:
    """Fail early when a CLI override would make residential goal sampling impossible."""
    probability_fields = (
        "train_near_obstacle_start_probability",
        "train_near_obstacle_probability",
        "train_near_obstacle_hard_probability",
    )
    for name in probability_fields:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")

    for prefix in (
        "train_near_obstacle_start",
        "train_near_obstacle",
        "train_near_obstacle_hard",
    ):
        minimum = float(getattr(args, f"{prefix}_min_clearance"))
        maximum = float(getattr(args, f"{prefix}_max_clearance"))
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"Invalid {prefix.replace('_', '-')} clearance range.")

    if int(args.train_near_obstacle_curriculum_episodes) < 0:
        raise ValueError("--train-near-obstacle-curriculum-episodes must be non-negative.")
    if int(args.train_near_obstacle_hardening_episodes) < 0:
        raise ValueError("--train-near-obstacle-hardening-episodes must be non-negative.")

    for value in (
        args.train_goal_start_max_distance,
        args.train_goal_final_max_distance,
        args.train_goal_hard_max_distance,
    ):
        if value is not None and (
            not np.isfinite(value) or value < config.min_start_goal_distance
        ):
            raise ValueError(
                "Training goal maximum distance must be finite and no smaller than "
                "min_start_goal_distance."
            )

    flyable_min = config.uav_radius
    flyable_max = config.map_altitude - config.uav_radius
    for stage in ("start", "final", "hard"):
        minimum = getattr(args, f"train_goal_{stage}_min_altitude")
        maximum = getattr(args, f"train_goal_{stage}_max_altitude")
        if minimum is None and maximum is None:
            continue
        if minimum is None or maximum is None:
            raise ValueError(
                f"Both --train-goal-{stage}-min-altitude and the matching maximum "
                "must be provided together."
            )
        if not all(np.isfinite([minimum, maximum])):
            raise ValueError("Training goal altitude bounds must be finite.")
        if minimum < flyable_min or maximum > flyable_max or maximum < minimum:
            raise ValueError(
                f"Training goal {stage} altitude must stay in the flyable range "
                f"[{flyable_min}, {flyable_max}] m."
            )


# ==================== 评估目标分布解析 ====================
def resolve_evaluation_goal_sampling(
    args: argparse.Namespace,
    config: UAVEnvConfig,
) -> dict[str, object]:
    """Resolve one complete, reproducible evaluation goal distribution."""
    if args.eval_goal_mode == "match-training":
        resolved: dict[str, object] = {
            "mode": "match-training",
            "near_obstacle_probability": args.train_near_obstacle_probability,
            "min_clearance_m": args.train_near_obstacle_min_clearance,
            "max_clearance_m": args.train_near_obstacle_max_clearance,
            "max_distance_m": args.train_goal_final_max_distance,
            "min_altitude_m": args.train_goal_final_min_altitude,
            "max_altitude_m": args.train_goal_final_max_altitude,
            "side_clearance_only": args.train_goal_side_clearance_only,
        }
    elif args.eval_goal_mode == "hard":
        resolved = {
            "mode": "hard",
            "near_obstacle_probability": args.train_near_obstacle_hard_probability,
            "min_clearance_m": args.train_near_obstacle_hard_min_clearance,
            "max_clearance_m": args.train_near_obstacle_hard_max_clearance,
            "max_distance_m": args.train_goal_hard_max_distance,
            "min_altitude_m": args.train_goal_hard_min_altitude,
            "max_altitude_m": args.train_goal_hard_max_altitude,
            "side_clearance_only": args.train_goal_side_clearance_only,
        }
    else:
        resolved = {
            "mode": "stress",
            "near_obstacle_probability": 0.70,
            "min_clearance_m": 2.0,
            "max_clearance_m": 12.0,
            "max_distance_m": None,
            "min_altitude_m": None,
            "max_altitude_m": None,
            "side_clearance_only": False,
        }

    overrides = {
        "near_obstacle_probability": args.eval_near_obstacle_probability,
        "min_clearance_m": args.eval_near_obstacle_min_clearance,
        "max_clearance_m": args.eval_near_obstacle_max_clearance,
        "max_distance_m": args.eval_goal_max_distance,
        "min_altitude_m": args.eval_goal_min_altitude,
        "max_altitude_m": args.eval_goal_max_altitude,
        "side_clearance_only": args.eval_goal_side_clearance_only,
    }
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value

    probability = float(resolved["near_obstacle_probability"])
    minimum = float(resolved["min_clearance_m"])
    maximum = float(resolved["max_clearance_m"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Evaluation near-obstacle probability must be in [0, 1].")
    if minimum < 0.0 or maximum < minimum:
        raise ValueError("Invalid evaluation goal clearance range.")
    max_distance = resolved["max_distance_m"]
    if max_distance is not None and (
        not np.isfinite(max_distance) or max_distance < config.min_start_goal_distance
    ):
        raise ValueError("Evaluation goal maximum distance is invalid.")
    altitude_min = resolved["min_altitude_m"]
    altitude_max = resolved["max_altitude_m"]
    if (altitude_min is None) != (altitude_max is None):
        raise ValueError("Evaluation altitude minimum and maximum must be set together.")
    if altitude_min is not None:
        flyable_min = config.uav_radius
        flyable_max = config.map_altitude - config.uav_radius
        if (
            not all(np.isfinite([altitude_min, altitude_max]))
            or altitude_min < flyable_min
            or altitude_max > flyable_max
            or altitude_max < altitude_min
        ):
            raise ValueError(
                f"Evaluation altitude must stay in [{flyable_min}, {flyable_max}] m."
            )
    return resolved


# ==================== 最佳模型比较规则 ====================
def pure_policy_validation_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    """Rank checkpoints by rates so validation sets of different sizes remain comparable."""
    episodes = max(1, int(metrics.get("episodes", 1)))
    success_rate = float(
        metrics.get("success_rate", int(metrics.get("successes", 0)) / episodes)
    )
    collision_rate = float(
        metrics.get("collision_rate", int(metrics.get("collisions", 0)) / episodes)
    )
    return (
        success_rate,
        -collision_rate,
        -float(metrics.get("avg_success_steps", float("inf"))),
        float(metrics.get("avg_reward", -float("inf"))),
    )


# ==================== 完整训练、评估和导出主流程 ====================
def main() -> None:
    """组织完整三维训练和评估流程。"""
    # 1. 解析参数；--list-scenes 只打印场景信息，不启动训练。
    args = parse_args()
    if args.list_scenes:
        for key in available_scene_keys():
            scene = get_training_scene(key)
            print(f"{key}: {scene.display_name} ({scene.scene_name})")
        return

    # 2. 固定随机性，并按场景隔离模型目录和本次运行结果目录。
    started = time.time()
    fix_seed(args.seed)

    scene = get_training_scene(args.scene)
    scene_output_root = args.output_root / scene.key
    output_dir: Path = args.output_dir or scene_output_root / "runs"
    save_model: Path = args.save_model or scene_output_root / "models" / f"{scene.key}_dqn.pt"
    save_best_model: Path = args.save_best_model or save_model.with_name(
        f"{save_model.stem}_best{save_model.suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = make_run_output_dir(output_dir)

    # 3. 合并场景默认值和命令行覆盖项，得到最终训练/评估配置。
    config = make_config(args)
    apply_scene_training_defaults(args, scene.key, config)
    validate_scene_training_curriculum(args, config)
    evaluation_goal_sampling = resolve_evaluation_goal_sampling(args, config)
    default_z = min(max(args.default_altitude, 1.0), config.map_altitude - 1.0)
    start = optional_point_3d(args.start_x, args.start_y, args.start_z, "start", default_z)
    if start is None:
        start = scene.default_start(config)
    goal = optional_point_3d(args.target_x, args.target_y, args.target_z, "target", default_z)

    # 4. 运行开始前固化完整配置，便于日后复现实验和解释 checkpoint。
    run_metadata = {
        "scene_key": scene.key,
        "scene_display_name": scene.display_name,
        "checkpoint_path": str(save_model),
        "training_goal_sampling": {
            "start_probability": args.train_near_obstacle_start_probability,
            "final_probability": args.train_near_obstacle_probability,
            "start_min_clearance_m": args.train_near_obstacle_start_min_clearance,
            "start_max_clearance_m": args.train_near_obstacle_start_max_clearance,
            "final_min_clearance_m": args.train_near_obstacle_min_clearance,
            "final_max_clearance_m": args.train_near_obstacle_max_clearance,
            "curriculum_episodes": args.train_near_obstacle_curriculum_episodes,
            "start_max_distance_m": args.train_goal_start_max_distance,
            "final_max_distance_m": args.train_goal_final_max_distance,
            "start_altitude_m": [
                args.train_goal_start_min_altitude,
                args.train_goal_start_max_altitude,
            ],
            "final_altitude_m": [
                args.train_goal_final_min_altitude,
                args.train_goal_final_max_altitude,
            ],
            "hard_probability": args.train_near_obstacle_hard_probability,
            "hard_min_clearance_m": args.train_near_obstacle_hard_min_clearance,
            "hard_max_clearance_m": args.train_near_obstacle_hard_max_clearance,
            "hardening_episodes": args.train_near_obstacle_hardening_episodes,
            "hard_max_distance_m": args.train_goal_hard_max_distance,
            "hard_altitude_m": [
                args.train_goal_hard_min_altitude,
                args.train_goal_hard_max_altitude,
            ],
            "side_clearance_only": args.train_goal_side_clearance_only,
        },
        "safe_action_mask": not args.disable_safe_action_mask,
        "pure_policy_validation": {
            "interval": args.validation_interval,
            "episodes": args.validation_episodes,
            "best_checkpoint_path": str(save_best_model),
            "latest_checkpoint_interval": args.checkpoint_interval,
        },
        "evaluation_goal_sampling": evaluation_goal_sampling,
        "config": config_to_dict(config),
    }
    if args.export_qgc_plan:
        run_metadata["qgc_export"] = {
            "origin_latitude_wgs84": args.qgc_origin_lat_wgs84,
            "origin_longitude_wgs84": args.qgc_origin_lon_wgs84,
            "origin_altitude_amsl_m": args.qgc_origin_alt_amsl,
            "firmware": args.qgc_firmware,
            "system_id": args.qgc_system_id,
            "autoload_directory": str(args.qgc_autoload_dir) if args.qgc_autoload_dir else None,
            "takeoff_altitude_m": args.qgc_takeoff_altitude,
            "simplification_tolerance_m": args.qgc_simplify_tolerance,
            "max_segment_length_m": args.qgc_max_segment_length,
            "acceptance_radius_m": args.qgc_acceptance_radius,
            "hover_speed_m_s": args.qgc_hover_speed,
            "cruise_speed_m_s": args.qgc_cruise_speed,
            "end_action": args.qgc_end_action,
        }
    (run_output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. 环境决定状态/动作维度，智能体据此创建在线网络和目标网络。
    env = UAVPathPlanningEnv(config=config, seed=args.seed)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.num_actions,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=args.device,
    )

    # 输出本次运行的核心维度、场景、动力学、奖励和课程设置。
    print(
        f"state_dim={env.state_dim}, action_dim={env.num_actions}, "
        f"device={device_description(agent.device)}"
    )
    print(f"actions={dict(enumerate(ACTION_NAMES))}")
    print(
        f"scene={env.config.scene_name}, "
        f"origin_gcj02=({env.config.origin_lon_gcj02}, {env.config.origin_lat_gcj02}), "
        f"bounds=x[{env.map_min[0]}, {env.map_max[0]}], "
        f"y[{env.map_min[1]}, {env.map_max[1]}], z[{env.map_min[2]}, {env.map_max[2]}], "
        f"obstacles={len(env.config.obstacles)}, inflation={env.config.obstacle_inflation}m"
    )
    print(
        "dynamics="
        f"horizontal_speed<={config.max_horizontal_speed}m/s, "
        f"3d_speed<={config.max_speed}m/s, "
        f"climb<={config.max_climb_speed}m/s, "
        f"descent<={config.max_descent_speed}m/s, "
        f"climb_angle<={config.max_climb_angle_deg}deg, "
        f"climb_angle_at_max_horizontal_speed="
        f"{config.climb_angle_at_max_horizontal_speed_deg:.1f}deg, "
        f"normal_acceleration<={config.normal_acceleration}m/s^2, "
        f"peak_acceleration<={config.max_acceleration}m/s^2, "
        f"deceleration<={config.max_deceleration}m/s^2, "
        f"jerk<={config.max_jerk}m/s^3"
    )
    print(
        f"reward_v{config.reward_shaping_version}="
        f"extra_altitude={config.extra_altitude_penalty_scale}/m, "
        f"altitude_margin={config.extra_altitude_margin}m, "
        f"detour={config.detour_penalty_scale}/m, "
        f"turn={config.turn_penalty_scale}/pi_rad, "
        f"goal_altitude={config.goal_altitude_penalty_scale}/m, "
        f"vertical_guidance={config.vertical_speed_guidance_scale}, "
        f"guidance_distance={config.goal_guidance_distance}m"
    )
    print(
        "training="
        f"epsilon({args.epsilon_start}->{args.epsilon_end}, decay={args.epsilon_decay}), "
        f"near_obstacle_curriculum="
        f"{args.train_near_obstacle_start_probability:.0%}->"
        f"{args.train_near_obstacle_probability:.0%}/"
        f"{args.train_near_obstacle_curriculum_episodes}ep -> "
        f"{args.train_near_obstacle_hard_probability:.0%}/"
        f"{args.train_near_obstacle_hardening_episodes}ep, "
        f"clearance={args.train_near_obstacle_start_min_clearance}-"
        f"{args.train_near_obstacle_start_max_clearance}m -> "
        f"{args.train_near_obstacle_min_clearance}-"
        f"{args.train_near_obstacle_max_clearance}m -> "
        f"{args.train_near_obstacle_hard_min_clearance}-"
        f"{args.train_near_obstacle_hard_max_clearance}m/"
        f"{args.train_near_obstacle_hardening_episodes}ep, "
        f"safe_action_mask={not args.disable_safe_action_mask}"
    )
    print(
        "evaluation="
        f"mode={evaluation_goal_sampling['mode']}, "
        f"near_obstacle={float(evaluation_goal_sampling['near_obstacle_probability']):.0%}, "
        f"clearance={evaluation_goal_sampling['min_clearance_m']}-"
        f"{evaluation_goal_sampling['max_clearance_m']}m, "
        f"max_distance={evaluation_goal_sampling['max_distance_m']}, "
        f"altitude={evaluation_goal_sampling['min_altitude_m']}-"
        f"{evaluation_goal_sampling['max_altitude_m']}m, "
        f"side_clearance_only={evaluation_goal_sampling['side_clearance_only']}"
    )

    # 6. 优先加载用户指定模型；否则在非 fresh-start 模式下自动续接场景模型。
    resume_model = args.load_model
    if resume_model is None and not args.fresh_start and save_model.exists():
        resume_model = save_model

    trained_episodes = 0
    curriculum_trained_episodes = 0
    exploration_trained_episodes = 0
    effective_epsilon_start = args.epsilon_start
    if resume_model:
        # 住宅区旧经验缺少可靠动作掩码，状态迁移时默认只保留网络权重。
        discard_residential_legacy_replay = (
            scene.key != "wujing_airfield" and not args.keep_legacy_replay
        )
        checkpoint = agent.load(
            resume_model,
            discard_replay_on_state_migration=discard_residential_legacy_replay,
        )
        if (
            discard_residential_legacy_replay
            and int(checkpoint.get("replay_transition_version", 0)) < 2
            and len(agent.replay_buffer) > 0
        ):
            agent.replay_buffer.clear()
            checkpoint["replay_buffer_discarded"] = True
        # 防止把其他场景或其他奖励版本的模型错误接入当前训练。
        checkpoint_scene_key = checkpoint.get("scene_key")
        checkpoint_config = checkpoint.get("config")
        checkpoint_scene_name = (
            checkpoint_config.get("scene_name")
            if isinstance(checkpoint_config, dict)
            else None
        )
        if checkpoint_scene_key not in (None, scene.key) or checkpoint_scene_name not in (
            None,
            config.scene_name,
        ):
            raise ValueError(
                f"Checkpoint {resume_model} belongs to a different scene; "
                f"selected scene is {scene.key}."
            )
        checkpoint_reward_version = (
            checkpoint_config.get("reward_shaping_version", 1)
            if isinstance(checkpoint_config, dict)
            else 1
        )
        if checkpoint_reward_version != config.reward_shaping_version:
            raise ValueError(
                f"Checkpoint {resume_model} uses reward shaping version "
                f"{checkpoint_reward_version}, but the current environment uses "
                f"version {config.reward_shaping_version}. Start a new model with --fresh-start."
            )
        # 分别恢复总训练、课程学习和探索率进度，保证续训曲线连续。
        trained_episodes = int(checkpoint.get("trained_episodes", 0))
        curriculum_trained_episodes = int(
            checkpoint.get("curriculum_trained_episodes", 0)
        )
        exploration_trained_episodes = int(
            checkpoint.get("exploration_trained_episodes", trained_episodes)
        )
        effective_epsilon_start = float(
            checkpoint.get("effective_epsilon_start", args.epsilon_start)
        )
        if checkpoint.get("replay_buffer_discarded"):
            exploration_trained_episodes = 0
            effective_epsilon_start = min(
                args.epsilon_start,
                args.migration_epsilon_start,
            )
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"Loaded model from {resume_model}")
        if checkpoint.get("state_dim_migrated_from") is not None:
            print(
                "Migrated checkpoint state: "
                f"{checkpoint['state_dim_migrated_from']} -> {env.state_dim}; "
                "new braking-feature weights and replay columns were initialized to zero."
            )
        if checkpoint.get("replay_buffer_discarded"):
            print(
                "Discarded legacy replay transitions because they do not contain "
                "the new braking state/action masks; network weights were retained."
            )
        print(f"Resuming from episode {trained_episodes}, replay_buffer={len(agent.replay_buffer)}")
        print(f"Building-goal curriculum resumes from episode {curriculum_trained_episodes}")
        print(
            "Exploration adaptation resumes from episode "
            f"{exploration_trained_episodes} with epsilon_start={effective_epsilon_start}"
        )
    print(f"Run outputs will be saved to {run_output_dir}")

    # 记录续训起点，后续 callback 据此计算新增回合对应的绝对进度。
    resume_trained_episodes = trained_episodes
    resume_curriculum_episodes = curriculum_trained_episodes
    resume_exploration_episodes = exploration_trained_episodes

    def curriculum_progress_at(completed_episodes: int) -> int:
        newly_completed = max(0, completed_episodes - resume_trained_episodes)
        return resume_curriculum_episodes + newly_completed

    def exploration_progress_at(completed_episodes: int) -> int:
        newly_completed = max(0, completed_episodes - resume_trained_episodes)
        return resume_exploration_episodes + newly_completed

    # 7. 恢复历史最佳验证基线；评估分布变化时必须重新开始比较。
    best_validation_metrics = (
        checkpoint.get("best_validation_metrics")
        if resume_model and isinstance(checkpoint.get("best_validation_metrics"), dict)
        else None
    )
    if isinstance(best_validation_metrics, dict) and (
        best_validation_metrics.get("goal_sampling") != evaluation_goal_sampling
    ):
        print(
            "Reset the best-validation comparison baseline because the evaluation "
            "goal distribution changed; model weights and replay were kept."
        )
        best_validation_metrics = None
    if isinstance(best_validation_metrics, dict):
        best_validation_key = pure_policy_validation_key(best_validation_metrics)
    else:
        best_validation_key = (-1.0, -1.0, -float("inf"), -float("inf"))

    # 汇总 checkpoint 附加元数据，供最新模型和最佳模型共同使用。
    def checkpoint_extra_state(completed_episodes: int) -> dict[str, object]:
        return {
            "trained_episodes": completed_episodes,
            "curriculum_trained_episodes": curriculum_progress_at(completed_episodes),
            "exploration_trained_episodes": exploration_progress_at(completed_episodes),
            "epsilon_start": effective_epsilon_start,
            "effective_epsilon_start": effective_epsilon_start,
            "configured_epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "train_near_obstacle_probability": args.train_near_obstacle_probability,
            "train_near_obstacle_min_clearance": args.train_near_obstacle_min_clearance,
            "train_near_obstacle_max_clearance": args.train_near_obstacle_max_clearance,
            "train_near_obstacle_start_probability": args.train_near_obstacle_start_probability,
            "train_near_obstacle_start_min_clearance": args.train_near_obstacle_start_min_clearance,
            "train_near_obstacle_start_max_clearance": args.train_near_obstacle_start_max_clearance,
            "train_near_obstacle_curriculum_episodes": args.train_near_obstacle_curriculum_episodes,
            "train_near_obstacle_hard_probability": args.train_near_obstacle_hard_probability,
            "train_near_obstacle_hard_min_clearance": args.train_near_obstacle_hard_min_clearance,
            "train_near_obstacle_hard_max_clearance": args.train_near_obstacle_hard_max_clearance,
            "train_near_obstacle_hardening_episodes": args.train_near_obstacle_hardening_episodes,
            "train_goal_start_max_distance": args.train_goal_start_max_distance,
            "train_goal_final_max_distance": args.train_goal_final_max_distance,
            "train_goal_hard_max_distance": args.train_goal_hard_max_distance,
            "train_goal_start_min_altitude": args.train_goal_start_min_altitude,
            "train_goal_start_max_altitude": args.train_goal_start_max_altitude,
            "train_goal_final_min_altitude": args.train_goal_final_min_altitude,
            "train_goal_final_max_altitude": args.train_goal_final_max_altitude,
            "train_goal_hard_min_altitude": args.train_goal_hard_min_altitude,
            "train_goal_hard_max_altitude": args.train_goal_hard_max_altitude,
            "train_goal_side_clearance_only": args.train_goal_side_clearance_only,
            "evaluation_goal_sampling": evaluation_goal_sampling,
            "safe_action_mask": not args.disable_safe_action_mask,
            "best_validation_metrics": best_validation_metrics,
            "seed": args.seed,
            "scene_key": scene.key,
            "rng_state": capture_rng_state(),
        }

    # 独立验证环境避免训练环境状态影响固定种子纯策略验证。
    validation_env = UAVPathPlanningEnv(config=config, seed=args.seed + 100_000)

    def run_pure_policy_validation(completed_episodes: int, force: bool = False) -> None:
        nonlocal best_validation_key, best_validation_metrics
        if args.validation_interval == 0 or args.validation_episodes == 0:
            return
        if not force and completed_episodes % args.validation_interval != 0:
            return

        print(
            f"Starting pure-policy validation at episode {completed_episodes} "
            f"({args.validation_episodes} episodes, epsilon=0)..."
        )
        # 纯策略验证固定 epsilon=0，并使用与最终评估一致的目标分布。
        validation_results, _ = evaluate_agent(
            env=validation_env,
            agent=agent,
            episodes=args.validation_episodes,
            start=start,
            goal=goal,
            seed=args.seed + 200_000,
            near_obstacle_goal_probability=float(
                evaluation_goal_sampling["near_obstacle_probability"]
            ),
            near_obstacle_goal_min_clearance=float(
                evaluation_goal_sampling["min_clearance_m"]
            ),
            near_obstacle_goal_max_clearance=float(
                evaluation_goal_sampling["max_clearance_m"]
            ),
            goal_max_distance=evaluation_goal_sampling["max_distance_m"],
            goal_altitude_min=evaluation_goal_sampling["min_altitude_m"],
            goal_altitude_max=evaluation_goal_sampling["max_altitude_m"],
            goal_near_obstacle_horizontal_only=bool(
                evaluation_goal_sampling["side_clearance_only"]
            ),
            use_safe_action_mask=not args.disable_safe_action_mask,
            verbose=False,
            show_progress=True,
            progress_desc=f"validation@{completed_episodes}",
        )
        successes = sum(bool(item.get("success", False)) for item in validation_results)
        collisions = sum(bool(item.get("collision", False)) for item in validation_results)
        success_steps = [
            int(item["steps"])
            for item in validation_results
            if bool(item.get("success", False))
        ]
        avg_success_steps = (
            float(np.mean(success_steps)) if success_steps else float("inf")
        )
        avg_reward = float(
            np.mean([float(item["total_reward"]) for item in validation_results])
        )
        # 最佳模型按成功率、低碰撞、成功步数和奖励的优先级比较。
        validation_metrics = {
            "episodes": args.validation_episodes,
            "successes": successes,
            "success_rate": successes / args.validation_episodes,
            "collisions": collisions,
            "collision_rate": collisions / args.validation_episodes,
            "avg_success_steps": avg_success_steps,
            "avg_reward": avg_reward,
        }
        validation_key = pure_policy_validation_key(validation_metrics)
        print(
            "pure-policy validation: "
            f"episode={completed_episodes}, "
            f"success={successes}/{args.validation_episodes} "
            f"({successes / args.validation_episodes:.1%}), "
            f"collisions={collisions}, "
            f"avg_success_steps={avg_success_steps:.1f}, avg_reward={avg_reward:.2f}"
        )
        if validation_key > best_validation_key:
            best_validation_key = validation_key
            best_validation_metrics = {
                "completed_episodes": completed_episodes,
                "episodes": args.validation_episodes,
                "successes": successes,
                "success_rate": successes / args.validation_episodes,
                "collisions": collisions,
                "collision_rate": collisions / args.validation_episodes,
                "avg_success_steps": avg_success_steps,
                "avg_reward": avg_reward,
                "epsilon": 0.0,
                "goal_sampling": dict(evaluation_goal_sampling),
                "seed": args.seed + 200_000,
            }
            save_best_model.parent.mkdir(parents=True, exist_ok=True)
            saved_best_path = agent.save(
                save_best_model,
                config=config,
                extra_state=checkpoint_extra_state(completed_episodes),
            )
            print(f"Saved new pure-policy best model to {saved_best_path}")

    # 每回合结束后的统一钩子：按频率执行验证并保存可续训最新模型。
    def training_episode_end(completed_episodes: int) -> None:
        run_pure_policy_validation(completed_episodes)
        if args.checkpoint_interval and completed_episodes % args.checkpoint_interval == 0:
            save_model.parent.mkdir(parents=True, exist_ok=True)
            saved_checkpoint_path = agent.save(
                save_model,
                config=config,
                extra_state=checkpoint_extra_state(completed_episodes),
            )
            print(f"Saved periodic latest checkpoint to {saved_checkpoint_path}")

    # 8. 执行训练；skip-train 时直接使用已加载或新建的网络进入评估。
    history = TrainHistory()
    if not args.skip_train and args.episodes > 0:
        if resume_model:
            run_pure_policy_validation(trained_episodes, force=True)
        history = train_dqn(
            env=env,
            agent=agent,
            episodes=args.episodes,
            target_update_interval=args.target_update_interval,
            epsilon_start=effective_epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            start=start,
            goal=goal,
            near_obstacle_goal_probability=args.train_near_obstacle_probability,
            near_obstacle_goal_min_clearance=args.train_near_obstacle_min_clearance,
            near_obstacle_goal_max_clearance=args.train_near_obstacle_max_clearance,
            near_obstacle_goal_start_probability=args.train_near_obstacle_start_probability,
            near_obstacle_goal_start_min_clearance=args.train_near_obstacle_start_min_clearance,
            near_obstacle_goal_start_max_clearance=args.train_near_obstacle_start_max_clearance,
            near_obstacle_goal_curriculum_episodes=args.train_near_obstacle_curriculum_episodes,
            near_obstacle_goal_hard_probability=args.train_near_obstacle_hard_probability,
            near_obstacle_goal_hard_min_clearance=args.train_near_obstacle_hard_min_clearance,
            near_obstacle_goal_hard_max_clearance=args.train_near_obstacle_hard_max_clearance,
            near_obstacle_goal_hardening_episodes=args.train_near_obstacle_hardening_episodes,
            goal_max_distance_start=args.train_goal_start_max_distance,
            goal_max_distance_final=args.train_goal_final_max_distance,
            goal_max_distance_hard=args.train_goal_hard_max_distance,
            goal_altitude_start_min=args.train_goal_start_min_altitude,
            goal_altitude_start_max=args.train_goal_start_max_altitude,
            goal_altitude_final_min=args.train_goal_final_min_altitude,
            goal_altitude_final_max=args.train_goal_final_max_altitude,
            goal_altitude_hard_min=args.train_goal_hard_min_altitude,
            goal_altitude_hard_max=args.train_goal_hard_max_altitude,
            goal_near_obstacle_horizontal_only=args.train_goal_side_clearance_only,
            use_safe_action_mask=not args.disable_safe_action_mask,
            seed=args.seed,
            episode_offset=trained_episodes,
            epsilon_episode_offset=exploration_trained_episodes,
            curriculum_episode_offset=curriculum_trained_episodes,
            episode_end_callback=training_episode_end,
        )
        trained_episodes += args.episodes
        curriculum_trained_episodes += args.episodes
        exploration_trained_episodes += args.episodes
        if trained_episodes % max(1, args.validation_interval) != 0:
            run_pure_policy_validation(trained_episodes, force=True)
        # 训练结束后保存场景最新模型及本次运行目录快照。
        if save_model:
            save_model.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_state = checkpoint_extra_state(trained_episodes)
            saved_model_path = agent.save(
                save_model,
                config=config,
                extra_state=checkpoint_state,
            )
            print(f"Saved model to {saved_model_path}")
            run_model_path = run_output_dir / save_model.name
            saved_run_model_path = agent.save(
                run_model_path,
                config=config,
                extra_state=checkpoint_state,
            )
            print(f"Saved run model to {saved_run_model_path}")

        if not args.no_plots:
            plot_training(history, run_output_dir / "training_curve.png")

    # 9. 在选定目标分布上执行最终纯策略评估，并返回其中最佳轨迹。
    results, best_trajectory = evaluate_agent(
        env=env,
        agent=agent,
        episodes=args.eval_episodes,
        start=start,
        goal=goal,
        seed=args.seed,
        near_obstacle_goal_probability=float(
            evaluation_goal_sampling["near_obstacle_probability"]
        ),
        near_obstacle_goal_min_clearance=float(
            evaluation_goal_sampling["min_clearance_m"]
        ),
        near_obstacle_goal_max_clearance=float(
            evaluation_goal_sampling["max_clearance_m"]
        ),
        goal_max_distance=evaluation_goal_sampling["max_distance_m"],
        goal_altitude_min=evaluation_goal_sampling["min_altitude_m"],
        goal_altitude_max=evaluation_goal_sampling["max_altitude_m"],
        goal_near_obstacle_horizontal_only=bool(
            evaluation_goal_sampling["side_clearance_only"]
        ),
        use_safe_action_mask=not args.disable_safe_action_mask,
    )

    # 10. 保存最佳轨迹，并独立复算全部安全、动力学和终点条件。
    if best_trajectory:
        save_trajectory_csv(best_trajectory, run_output_dir / "best_trajectory.csv")
        timed_trajectory = env.timed_trajectory()
        save_timed_trajectory_csv(timed_trajectory, run_output_dir / "best_timed_trajectory.csv")
        deviation_tolerance = (
            1e-4
            if args.trajectory_deviation_tolerance is None
            else args.trajectory_deviation_tolerance
        )
        validation = validate_timed_trajectory(
            env=env,
            reference_path=best_trajectory,
            trajectory=timed_trajectory,
            deviation_tolerance=deviation_tolerance,
            max_speed=config.max_speed,
            max_acceleration=config.max_acceleration,
            max_jerk=config.max_jerk,
            goal=env.goal,
            goal_radius=config.goal_radius,
            goal_speed_tolerance=config.goal_speed_tolerance,
        )
        save_validation_json(validation, run_output_dir / "trajectory_validation.json")
        print(
            "timed trajectory: "
            f"samples={len(timed_trajectory.time)}, "
            f"duration={timed_trajectory.total_time:.2f}s, "
            f"length={timed_trajectory.total_length:.2f}m, "
            f"max_speed={timed_trajectory.max_speed:.2f}m/s, "
            f"max_horizontal_speed={validation.max_horizontal_speed:.2f}m/s, "
            f"max_climb_speed={validation.max_climb_speed:.2f}m/s, "
            f"max_descent_speed={validation.max_descent_speed:.2f}m/s, "
            f"max_acceleration={timed_trajectory.max_acceleration:.2f}m/s^2, "
            f"max_deceleration={validation.max_deceleration:.2f}m/s^2, "
            f"max_jerk={validation.max_jerk:.2f}m/s^3"
        )
        print(
            "trajectory validation: "
            f"passed={validation.passed}, "
            f"max_deviation={validation.max_deviation:.2f}m, "
            f"mean_deviation={validation.mean_deviation:.2f}m, "
            f"collision_free={validation.collision_free}, "
            f"within_bounds={validation.within_bounds}, "
            f"dynamics_consistent={validation.dynamics_consistent}, "
            f"goal_reached={validation.goal_reached}"
        )
        # 只有用户启用且验证通过时，export_qgc_outputs 才会生成飞控任务。
        if args.export_qgc_plan:
            export_qgc_outputs(args, env, timed_trajectory, validation, run_output_dir)

    # 11. 按绘图开关生成三维轨迹、动力学曲线和验证对比图。
    if (args.visualize or not args.no_plots) and best_trajectory:
        plot_trajectory(env, best_trajectory, run_output_dir / "best_trajectory.png")
        plot_trajectory_profiles(timed_trajectory, run_output_dir / "trajectory_profiles.png")
        plot_trajectory_validation(
            env,
            best_trajectory,
            timed_trajectory,
            validation,
            run_output_dir / "trajectory_validation.png",
        )

    if results:
        successes = [bool(result.get("success", False)) for result in results]
        avg_distance = np.mean([float(result["distance_to_goal"]) for result in results])
        print(f"final success rate: {np.mean(successes):.2%}, avg final distance: {avg_distance:.2f}")

    print(f"total time: {time.time() - started:.2f} sec")


if __name__ == "__main__":
    main()
