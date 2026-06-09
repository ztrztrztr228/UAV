# -*- coding: utf-8 -*-
"""DQN 训练和评估流程。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .agent import DQNAgent
from .config import DEFAULT_SEED
from .environment import UAVPathPlanningEnv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 是可选依赖
    tqdm = None


@dataclass
class TrainHistory:
    """训练过程记录，用于画图和分析收敛情况。"""

    episode_rewards: list[float] = field(default_factory=list)#创建每步的奖励列表
    episode_lengths: list[int] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    collisions: list[bool] = field(default_factory=list)
    final_distances: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def epsilon_by_episode(
    episode: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
) -> float:
    """根据 episode 编号计算当前探索率 epsilon。"""
    return float(epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay))
#随机探索随回合减小

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
    seed: int = DEFAULT_SEED,
    episode_offset: int = 0,
) -> TrainHistory:
    """训练 DQN 智能体。

    每个 episode 中：重置环境、选择动作、执行动作、存储经验、更新网络、
    定期同步目标网络，并记录奖励、成功率、碰撞率等指标。
    """
    history = TrainHistory()
    iterator: Iterable[int] = range(episodes)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="training")

    global_step = 0
    for episode in iterator:
        absolute_episode = episode_offset + episode
        state = env.reset(start=start, goal=goal, seed=seed + absolute_episode)
        epsilon = epsilon_by_episode(absolute_episode, epsilon_start, epsilon_end, epsilon_decay)
        total_reward = 0.0
        episode_losses: list[float] = []
        final_info: dict[str, object] = {}#每回合重置

        while True:#一个回合
            action = agent.select_action(state, epsilon=epsilon)#agent选动作
            next_state, reward, done, info = env.step(action)#动作导致下一步状态
            agent.replay_buffer.push(state, action, reward, next_state, done)#存入经验池

            loss = agent.learn()#更新网络
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            global_step += 1

            if global_step % target_update_interval == 0:
                agent.update_target_network()#隔一段时间更新目标网络

            if done:
                final_info = info
                break

        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        history.episode_rewards.append(float(total_reward))
        history.episode_lengths.append(int(final_info.get("steps", env.steps)))
        history.successes.append(bool(final_info.get("success", False)))
        history.collisions.append(bool(final_info.get("collision", False)))
        history.final_distances.append(float(final_info.get("distance_to_goal", env.distance_to_goal())))
        history.losses.append(mean_loss)
        history.epsilons.append(epsilon)#参数存入列表

        if tqdm is not None and hasattr(iterator, "set_postfix"):#显示进度条
            recent_success = np.mean(history.successes[-50:]) if history.successes else 0.0
            iterator.set_postfix(
                reward=f"{np.mean(history.episode_rewards[-20:]):.1f}",
                success=f"{recent_success:.2f}",
                eps=f"{epsilon:.2f}",
            )
        elif (episode + 1) % 10 == 0 or episode == 0:
            recent_success = np.mean(history.successes[-50:]) if history.successes else 0.0
            print(
                f"episode {episode + 1:4d}/{episodes}: "
                f"reward={total_reward:8.2f}, success_50={recent_success:.2f}, epsilon={epsilon:.3f}"
            )

    agent.update_target_network()
    return history


def evaluate_agent(
    env: UAVPathPlanningEnv,
    agent: DQNAgent,
    episodes: int,
    start: Sequence[float] | None = None,
    goal: Sequence[float] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    """评估训练好的策略。

    评估时 epsilon=0，不再随机探索，直接选择 Q 值最大的动作。
    返回每轮评估结果，以及表现最好的一条轨迹。
    """
    agent.policy_net.eval()
    results: list[dict[str, object]] = []
    best_trajectory: list[np.ndarray] = []
    best_start: np.ndarray | None = None
    best_goal: np.ndarray | None = None
    best_score = -float("inf")

    for episode in range(episodes):
        state = env.reset(start=start, goal=goal, seed=seed + 10_000 + episode)
        total_reward = 0.0
        final_info: dict[str, object] = {}

        while True:
            action = agent.select_action(state, epsilon=0.0)
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
        score = total_reward + (1000.0 if final_info.get("success") else 0.0)
        if score > best_score:
            best_score = score
            best_trajectory = [point.copy() for point in env.trajectory]
            best_start = env.start.copy()
            best_goal = env.goal.copy()

        print(
            f"eval {episode + 1}/{episodes}: "
            f"event={final_info['event']}, reward={total_reward:.2f}, "
            f"steps={final_info['steps']}, distance={final_info['distance_to_goal']:.2f}, "
            f"path={final_info['path_length']:.2f}"
        )

    successes = [bool(result.get("success", False)) for result in results]
    collisions = [bool(result.get("collision", False)) for result in results]
    print(
        "evaluation summary: "
        f"success_rate={np.mean(successes):.2%}, "
        f"collision_rate={np.mean(collisions):.2%}, "
        f"avg_reward={np.mean([r['total_reward'] for r in results]):.2f}"
    )

    if best_start is not None and best_goal is not None:
        env.start = best_start
        env.goal = best_goal

    return results, best_trajectory
