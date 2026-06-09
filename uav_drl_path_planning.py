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
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from uav_drl.actions import ACTION_NAMES
from uav_drl.agent import DQNAgent
from uav_drl.config import DEFAULT_SEED, UAVEnvConfig
from uav_drl.environment import UAVPathPlanningEnv
from uav_drl.training import TrainHistory, evaluate_agent, train_dqn
from uav_drl.utils import fix_seed, optional_point_3d
from uav_drl.visualization import plot_training, plot_trajectory, save_trajectory_csv


def parse_args() -> argparse.Namespace:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a 3D DQN UAV path planner.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--map-width", type=float, default=100.0)
    parser.add_argument("--map-height", type=float, default=100.0)
    parser.add_argument("--map-altitude", type=float, default=30.0)
    parser.add_argument("--step-length", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--goal-radius", type=float, default=3.0)
    parser.add_argument("--default-altitude", type=float, default=8.0)

    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--start-z", type=float)
    parser.add_argument("--target-x", type=float)
    parser.add_argument("--target-y", type=float)
    parser.add_argument("--target-z", type=float)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=80_000)
    parser.add_argument("--target-update-interval", type=int, default=300)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=350.0)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-model", type=Path, default=Path("outputs/uav_dqn.pt"))
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--fresh-start", action="store_true", help="do not auto-load the previous checkpoint")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--device", default=None, help="cpu, cuda, or auto when omitted")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> UAVEnvConfig:
    """根据命令行参数生成三维环境配置。"""
    return UAVEnvConfig(
        map_width=args.map_width,
        map_height=args.map_height,
        map_altitude=args.map_altitude,
        step_length=args.step_length,
        max_steps=args.max_steps,
        goal_radius=args.goal_radius,
    )


def ground_center_start(config: UAVEnvConfig) -> tuple[float, float, float]:
 #起点设置为地图中央地面
    return (
        config.map_width / 2.0,
        config.map_height / 2.0,
        config.uav_radius + 1e-3,
    )


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


def main() -> None:
    """组织完整三维训练和评估流程。"""
    args = parse_args()
    started = time.time()
    fix_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = make_run_output_dir(output_dir)

    config = make_config(args)
    default_z = min(max(args.default_altitude, 1.0), args.map_altitude - 1.0)
    start = optional_point_3d(args.start_x, args.start_y, args.start_z, "start", default_z)
    if start is None:
        start = ground_center_start(config)
    goal = optional_point_3d(args.target_x, args.target_y, args.target_z, "target", default_z)

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

    print(f"state_dim={env.state_dim}, action_dim={env.num_actions}, device={agent.device}")
    print(f"actions={dict(enumerate(ACTION_NAMES))}")
    print(
        "map="
        f"({env.config.map_width}, {env.config.map_height}, {env.config.map_altitude}), "
        f"obstacles={len(env.config.obstacles)}"
    )

    resume_model = args.load_model
    if resume_model is None and not args.fresh_start and args.save_model and args.save_model.exists():
        resume_model = args.save_model

    if resume_model:
        agent.load(resume_model)
        print(f"Loaded model from {resume_model}")
    print(f"Run outputs will be saved to {run_output_dir}")

    history = TrainHistory()
    if not args.skip_train and args.episodes > 0:
        history = train_dqn(
            env=env,
            agent=agent,
            episodes=args.episodes,
            target_update_interval=args.target_update_interval,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            start=start,
            goal=goal,
            seed=args.seed,
        )
        if args.save_model:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            agent.save(args.save_model, config=config)
            print(f"Saved model to {args.save_model}")
            run_model_path = run_output_dir / args.save_model.name
            agent.save(run_model_path, config=config)
            print(f"Saved run model to {run_model_path}")

        if not args.no_plots:
            plot_training(history, run_output_dir / "training_curve.png")

    results, best_trajectory = evaluate_agent(
        env=env,
        agent=agent,
        episodes=args.eval_episodes,
        start=start,
        goal=goal,
        seed=args.seed,
    )

    if best_trajectory:
        save_trajectory_csv(best_trajectory, run_output_dir / "best_trajectory.csv")

    if (args.visualize or not args.no_plots) and best_trajectory:
        plot_trajectory(env, best_trajectory, run_output_dir / "best_trajectory.png")

    if results:
        successes = [bool(result.get("success", False)) for result in results]
        avg_distance = np.mean([float(result["distance_to_goal"]) for result in results])
        print(f"final success rate: {np.mean(successes):.2%}, avg final distance: {avg_distance:.2f}")

    print(f"total time: {time.time() - started:.2f} sec")


if __name__ == "__main__":
    main()
