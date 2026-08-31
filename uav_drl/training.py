# -*- coding: utf-8 -*-
"""DQN 训练和评估流程。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .agent import DQNAgent
from .config import DEFAULT_SEED
from .environment import UAVPathPlanningEnv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 是可选依赖
    tqdm = None


# ==================== 训练指标数据结构 ====================
@dataclass
class TrainHistory:
    """训练过程记录，用于画图和分析收敛情况。"""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    collisions: list[bool] = field(default_factory=list)
    final_distances: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    goal_sampling_modes: list[str] = field(default_factory=list)
    goal_obstacle_clearances: list[float] = field(default_factory=list)
    curriculum_probabilities: list[float] = field(default_factory=list)
    safe_action_fractions: list[float] = field(default_factory=list)
    mean_speeds: list[float] = field(default_factory=list)


# ==================== 目标课程学习 ====================
@dataclass(frozen=True)
class GoalSamplingStage:
    """Goal-distribution parameters at one curriculum stage."""

    near_obstacle_probability: float
    near_obstacle_min_clearance: float
    near_obstacle_max_clearance: float
    max_start_distance: float | None = None
    altitude_min: float | None = None
    altitude_max: float | None = None


def _interpolate_optional(
    start_value: float | None,
    final_value: float | None,
    progress: float,
) -> float | None:
    if start_value is None or final_value is None:
        return final_value if progress >= 1.0 else start_value
    return float(start_value + progress * (final_value - start_value))


def _interpolate_goal_stage(
    start: GoalSamplingStage,
    final: GoalSamplingStage,
    progress: float,
) -> GoalSamplingStage:
    value = float(np.clip(progress, 0.0, 1.0))
    return GoalSamplingStage(
        near_obstacle_probability=float(
            start.near_obstacle_probability
            + value * (final.near_obstacle_probability - start.near_obstacle_probability)
        ),
        near_obstacle_min_clearance=float(
            start.near_obstacle_min_clearance
            + value * (final.near_obstacle_min_clearance - start.near_obstacle_min_clearance)
        ),
        near_obstacle_max_clearance=float(
            start.near_obstacle_max_clearance
            + value * (final.near_obstacle_max_clearance - start.near_obstacle_max_clearance)
        ),
        max_start_distance=_interpolate_optional(
            start.max_start_distance,
            final.max_start_distance,
            value,
        ),
        altitude_min=_interpolate_optional(start.altitude_min, final.altitude_min, value),
        altitude_max=_interpolate_optional(start.altitude_max, final.altitude_max, value),
    )


def goal_sampling_stage_by_episode(
    episode: int,
    curriculum_episodes: int,
    start: GoalSamplingStage,
    standard: GoalSamplingStage,
    hard: GoalSamplingStage | None = None,
    hardening_episodes: int = 0,
) -> GoalSamplingStage:
    """Interpolate easy -> standard, then optionally standard -> hard."""

    if curriculum_episodes <= 0:
        standard_progress = 1.0
    else:
        standard_progress = float(np.clip(episode / curriculum_episodes, 0.0, 1.0))
    standard_stage = _interpolate_goal_stage(start, standard, standard_progress)
    if episode <= curriculum_episodes or hard is None or hardening_episodes <= 0:
        return standard_stage

    hardening_progress = float(
        np.clip((episode - curriculum_episodes) / hardening_episodes, 0.0, 1.0)
    )
    return _interpolate_goal_stage(standard, hard, hardening_progress)


def epsilon_by_episode(
    episode: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
) -> float:
    """根据 episode 编号计算当前探索率 epsilon。"""
    # 指数衰减使训练早期探索较多、后期逐渐接近纯策略。
    return float(epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay))


def goal_curriculum_by_episode(
    episode: int,
    curriculum_episodes: int,
    start_probability: float,
    final_probability: float,
    start_min_clearance: float,
    start_max_clearance: float,
    final_min_clearance: float,
    final_max_clearance: float,
    hardening_episodes: int = 0,
    hard_probability: float | None = None,
    hard_min_clearance: float | None = None,
    hard_max_clearance: float | None = None,
) -> tuple[float, float, float]:
    """Move from easy to standard goals, then optionally to a harder shell."""
    start = GoalSamplingStage(
        start_probability,
        start_min_clearance,
        start_max_clearance,
    )
    standard = GoalSamplingStage(
        final_probability,
        final_min_clearance,
        final_max_clearance,
    )
    hard = None
    if hardening_episodes > 0:
        hard = GoalSamplingStage(
            final_probability if hard_probability is None else hard_probability,
            final_min_clearance if hard_min_clearance is None else hard_min_clearance,
            final_max_clearance if hard_max_clearance is None else hard_max_clearance,
        )
    stage = goal_sampling_stage_by_episode(
        episode=episode,
        curriculum_episodes=curriculum_episodes,
        start=start,
        standard=standard,
        hard=hard,
        hardening_episodes=hardening_episodes,
    )
    return (
        stage.near_obstacle_probability,
        stage.near_obstacle_min_clearance,
        stage.near_obstacle_max_clearance,
    )

# ==================== DQN 训练主循环 ====================
def train_dqn(
    env: UAVPathPlanningEnv,
    agent: DQNAgent,
    episodes: int,
    target_update_interval: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    start: Sequence[float] | None = None,
    goal: Sequence[float] | None = None,
    near_obstacle_goal_probability: float = 0.70,
    near_obstacle_goal_min_clearance: float = 2.0,
    near_obstacle_goal_max_clearance: float = 12.0,
    near_obstacle_goal_start_probability: float = 0.30,
    near_obstacle_goal_start_min_clearance: float = 15.0,
    near_obstacle_goal_start_max_clearance: float = 30.0,
    near_obstacle_goal_curriculum_episodes: int = 1_500,
    near_obstacle_goal_hard_probability: float | None = None,
    near_obstacle_goal_hard_min_clearance: float | None = None,
    near_obstacle_goal_hard_max_clearance: float | None = None,
    near_obstacle_goal_hardening_episodes: int = 0,
    goal_max_distance_start: float | None = None,
    goal_max_distance_final: float | None = None,
    goal_max_distance_hard: float | None = None,
    goal_altitude_start_min: float | None = None,
    goal_altitude_start_max: float | None = None,
    goal_altitude_final_min: float | None = None,
    goal_altitude_final_max: float | None = None,
    goal_altitude_hard_min: float | None = None,
    goal_altitude_hard_max: float | None = None,
    goal_near_obstacle_horizontal_only: bool = False,
    use_safe_action_mask: bool = True,
    seed: int = DEFAULT_SEED,
    episode_offset: int = 0,
    epsilon_episode_offset: int | None = None,
    curriculum_episode_offset: int | None = None,
    episode_end_callback: Callable[[int], None] | None = None,
) -> TrainHistory:
    """训练 DQN 智能体。

    每个 episode 中：重置环境、选择动作、执行动作、存储经验、更新网络、
    定期同步目标网络，并记录奖励、成功率、碰撞率等指标。
    """
    # 初始化指标记录器和可选进度条。
    history = TrainHistory()
    iterator: Iterable[int] = range(episodes)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="training")

    global_step = 0
    # 课程进度和探索率进度可分别续接，支持迁移旧模型后重启探索衰减。
    curriculum_offset = (
        episode_offset
        if curriculum_episode_offset is None
        else curriculum_episode_offset
    )
    epsilon_offset = episode_offset if epsilon_episode_offset is None else epsilon_episode_offset
    # 把命令行中的简单、标准、困难三层参数整理为统一的数据结构。
    start_stage = GoalSamplingStage(
        near_obstacle_goal_start_probability,
        near_obstacle_goal_start_min_clearance,
        near_obstacle_goal_start_max_clearance,
        max_start_distance=goal_max_distance_start,
        altitude_min=goal_altitude_start_min,
        altitude_max=goal_altitude_start_max,
    )
    standard_stage = GoalSamplingStage(
        near_obstacle_goal_probability,
        near_obstacle_goal_min_clearance,
        near_obstacle_goal_max_clearance,
        max_start_distance=goal_max_distance_final,
        altitude_min=goal_altitude_final_min,
        altitude_max=goal_altitude_final_max,
    )
    hard_stage = None
    if near_obstacle_goal_hardening_episodes > 0:
        hard_stage = GoalSamplingStage(
            near_obstacle_goal_probability
            if near_obstacle_goal_hard_probability is None
            else near_obstacle_goal_hard_probability,
            near_obstacle_goal_min_clearance
            if near_obstacle_goal_hard_min_clearance is None
            else near_obstacle_goal_hard_min_clearance,
            near_obstacle_goal_max_clearance
            if near_obstacle_goal_hard_max_clearance is None
            else near_obstacle_goal_hard_max_clearance,
            max_start_distance=(
                goal_max_distance_final
                if goal_max_distance_hard is None
                else goal_max_distance_hard
            ),
            altitude_min=(
                goal_altitude_final_min
                if goal_altitude_hard_min is None
                else goal_altitude_hard_min
            ),
            altitude_max=(
                goal_altitude_final_max
                if goal_altitude_hard_max is None
                else goal_altitude_hard_max
            ),
        )
    # 每个 episode 根据当前绝对进度插值目标分布，再独立固定随机种子。
    for episode in iterator:
        absolute_episode = episode_offset + episode
        curriculum_episode = curriculum_offset + episode
        epsilon_episode = epsilon_offset + episode
        curriculum_stage = goal_sampling_stage_by_episode(
            episode=curriculum_episode,
            curriculum_episodes=near_obstacle_goal_curriculum_episodes,
            start=start_stage,
            standard=standard_stage,
            hard=hard_stage,
            hardening_episodes=near_obstacle_goal_hardening_episodes,
        )
        curriculum_probability = curriculum_stage.near_obstacle_probability
        curriculum_min_clearance = curriculum_stage.near_obstacle_min_clearance
        curriculum_max_clearance = curriculum_stage.near_obstacle_max_clearance
        state = env.reset(
            start=start,
            goal=goal,
            seed=seed + absolute_episode,
            goal_near_obstacle_probability=curriculum_probability,
            goal_near_obstacle_min_clearance=curriculum_min_clearance,
            goal_near_obstacle_max_clearance=curriculum_max_clearance,
            goal_max_start_distance=curriculum_stage.max_start_distance,
            goal_altitude_min=curriculum_stage.altitude_min,
            goal_altitude_max=curriculum_stage.altitude_max,
            goal_near_obstacle_horizontal_only=goal_near_obstacle_horizontal_only,
        )
        epsilon = epsilon_by_episode(epsilon_episode, epsilon_start, epsilon_end, epsilon_decay)
        total_reward = 0.0
        episode_losses: list[float] = []
        episode_safe_action_fractions: list[float] = []
        episode_speeds: list[float] = []
        final_info: dict[str, object] = {}

        # 当前状态的动作掩码会被下一步计算结果直接复用，避免重复扫描。
        action_mask = (
            env.safe_action_mask()
            if use_safe_action_mask
            else np.ones(env.num_actions, dtype=bool)
        )
        while True:
            episode_safe_action_fractions.append(float(np.mean(action_mask)))
            action = agent.select_action(
                state,
                epsilon=epsilon,
                action_mask=action_mask,
            )
            next_state, reward, done, info = env.step(action)

            # 终止状态无需预测下一动作；非终止状态计算一次安全掩码并写入经验。
            next_action_mask = (
                np.ones(env.num_actions, dtype=bool)
                if done or not use_safe_action_mask
                else env.safe_action_mask()
            )
            agent.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_action_mask=next_action_mask,
            )

            # 每个环境步都尝试从经验池学习；样本不足时 learn() 会直接返回。
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            action_mask = next_action_mask
            total_reward += reward
            episode_speeds.append(float(info.get("speed", np.linalg.norm(env.velocity))))
            global_step += 1

            if global_step % target_update_interval == 0:
                agent.update_target_network()

            if done:
                final_info = info
                break

        # 回合结束后汇总奖励、终局、安全动作比例和速度等训练指标。
        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        history.episode_rewards.append(float(total_reward))
        history.episode_lengths.append(int(final_info.get("steps", env.steps)))
        history.successes.append(bool(final_info.get("success", False)))
        history.collisions.append(bool(final_info.get("collision", False)))
        history.final_distances.append(float(final_info.get("distance_to_goal", env.distance_to_goal())))
        history.losses.append(mean_loss)
        history.epsilons.append(epsilon)
        history.goal_sampling_modes.append(env.goal_sampling_mode)
        history.goal_obstacle_clearances.append(float(env.goal_obstacle_clearance))
        history.curriculum_probabilities.append(curriculum_probability)
        history.safe_action_fractions.append(
            float(np.mean(episode_safe_action_fractions))
            if episode_safe_action_fractions
            else 1.0
        )
        history.mean_speeds.append(
            float(np.mean(episode_speeds)) if episode_speeds else 0.0
        )

        # 优先更新 tqdm 后缀；无 tqdm 时按固定间隔输出文本日志。
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            recent_success = np.mean(history.successes[-50:]) if history.successes else 0.0
            recent_near_goal_rate = np.mean(
                [mode == "near_obstacle" for mode in history.goal_sampling_modes[-50:]]
            )
            iterator.set_postfix(
                reward=f"{np.mean(history.episode_rewards[-20:]):.1f}",
                success=f"{recent_success:.2f}",
                eps=f"{epsilon:.2f}",
                near_goal=f"{recent_near_goal_rate:.2f}",
                safe=f"{history.safe_action_fractions[-1]:.2f}",
                speed=f"{history.mean_speeds[-1]:.1f}",
            )
        elif (episode + 1) % 10 == 0 or episode == 0:
            recent_success = np.mean(history.successes[-50:]) if history.successes else 0.0
            recent_near_goal_rate = np.mean(
                [mode == "near_obstacle" for mode in history.goal_sampling_modes[-50:]]
            )
            print(
                f"episode {episode + 1:4d}/{episodes}: "
                f"reward={total_reward:8.2f}, success_50={recent_success:.2f}, "
                f"epsilon={epsilon:.3f}, near_goal_50={recent_near_goal_rate:.2f}, "
                f"curriculum={curriculum_probability:.2f}, "
                f"safe_actions={history.safe_action_fractions[-1]:.2f}, "
                f"speed={history.mean_speeds[-1]:.2f}m/s"
            )

        # 回调由主入口用于周期验证和 checkpoint 保存。
        if episode_end_callback is not None:
            episode_end_callback(absolute_episode + 1)

    agent.update_target_network()
    return history


# ==================== 纯策略评估 ====================
def evaluate_agent(
    env: UAVPathPlanningEnv,
    agent: DQNAgent,
    episodes: int,
    start: Sequence[float] | None = None,
    goal: Sequence[float] | None = None,
    seed: int = DEFAULT_SEED,
    near_obstacle_goal_probability: float = 0.0,
    near_obstacle_goal_min_clearance: float = 2.0,
    near_obstacle_goal_max_clearance: float = 12.0,
    goal_max_distance: float | None = None,
    goal_altitude_min: float | None = None,
    goal_altitude_max: float | None = None,
    goal_near_obstacle_horizontal_only: bool = False,
    use_safe_action_mask: bool = True,
    verbose: bool = True,
    show_progress: bool = False,
    progress_desc: str = "evaluating",
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """评估训练好的策略。

    评估时 epsilon=0，不再随机探索，直接选择 Q 值最大的动作。
    返回每轮评估结果，以及表现最好的一条轨迹。
    """
    agent.policy_net.eval()
    results: list[dict[str, object]] = []
    best_trajectory: list[np.ndarray] = []
    best_velocities: list[np.ndarray] = []
    best_accelerations: list[np.ndarray] = []
    best_start: np.ndarray | None = None
    best_goal: np.ndarray | None = None
    best_score = -float("inf")

    iterator: Iterable[int] = range(episodes)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=progress_desc, leave=False)

    # 评估固定 epsilon=0，但仍使用与训练一致的安全动作屏蔽。
    for episode in iterator:
        state = env.reset(
            start=start,
            goal=goal,
            seed=seed + 10_000 + episode,
            goal_near_obstacle_probability=near_obstacle_goal_probability,
            goal_near_obstacle_min_clearance=near_obstacle_goal_min_clearance,
            goal_near_obstacle_max_clearance=near_obstacle_goal_max_clearance,
            goal_max_start_distance=goal_max_distance,
            goal_altitude_min=goal_altitude_min,
            goal_altitude_max=goal_altitude_max,
            goal_near_obstacle_horizontal_only=goal_near_obstacle_horizontal_only,
        )
        total_reward = 0.0
        final_info: dict[str, object] = {}

        while True:
            action_mask = (
                env.safe_action_mask()
                if use_safe_action_mask
                else np.ones(env.num_actions, dtype=bool)
            )
            action = agent.select_action(state, epsilon=0.0, action_mask=action_mask)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                final_info = dict(info)
                break

        final_info["total_reward"] = float(total_reward)
        final_info["start"] = env.start.copy()
        final_info["goal"] = env.goal.copy()
        results.append(final_info)

        # 优先选择成功轨迹；如果都失败，则选择奖励最高的轨迹。
        # 成功轨迹优先于失败轨迹；同类轨迹再按累计奖励排序。
        score = total_reward + (1000.0 if final_info.get("success") else 0.0)
        if score > best_score:
            best_score = score
            best_trajectory = [point.copy() for point in env.trajectory]
            best_velocities = [value.copy() for value in env.velocity_trajectory]
            best_accelerations = [value.copy() for value in env.acceleration_trajectory]
            best_start = env.start.copy()
            best_goal = env.goal.copy()

        goal_clearance = final_info.get("goal_obstacle_clearance")
        goal_clearance_text = "n/a" if goal_clearance is None else f"{float(goal_clearance):.2f}m"
        if verbose:
            print(
                f"eval {episode + 1}/{episodes}: "
                f"event={final_info['event']}, reward={total_reward:.2f}, "
                f"steps={final_info['steps']}, distance={final_info['distance_to_goal']:.2f}, "
                f"path={final_info['path_length']:.2f}, "
                f"goal_sample={final_info.get('goal_sampling_mode', 'unknown')}, "
                f"goal_building_clearance={goal_clearance_text}"
            )

    if results:
        successes = [bool(result.get("success", False)) for result in results]
        collisions = [bool(result.get("collision", False)) for result in results]
        if verbose:
            print(
                "evaluation summary: "
                f"success_rate={np.mean(successes):.2%}, "
                f"collision_rate={np.mean(collisions):.2%}, "
                f"avg_reward={np.mean([r['total_reward'] for r in results]):.2f}, "
                f"near_obstacle_goal_rate="
                f"{np.mean([r.get('goal_sampling_mode') == 'near_obstacle' for r in results]):.2%}"
            )
    elif verbose:
        print("evaluation skipped: eval_episodes=0")

    # 把最佳回合恢复到环境中，供主入口继续导出、验证和绘图。
    if best_start is not None and best_goal is not None:
        env.start = best_start
        env.goal = best_goal
        env.trajectory = best_trajectory
        env.velocity_trajectory = best_velocities
        env.acceleration_trajectory = best_accelerations
        env.position = best_trajectory[-1].copy()
        env.velocity = best_velocities[-1].copy()
        env.acceleration = best_accelerations[-1].copy()

    return results, best_trajectory
