# -*- coding: utf-8 -*-
"""DQN 智能体和经验回放池。"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from .config import UAVEnvConfig, config_to_dict
from .model import QNetwork


class ReplayBuffer:
    """经验回放池。

    存储 (state, action, reward, next_state, done)，训练时随机采样 batch，
    用来打破轨迹数据的时间相关性，提高 DQN 训练稳定性。
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """保存一条交互经验。"""
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机采样一个 batch，并转换为 tensor。"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_state_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor


class DQNAgent:
    """DQN 智能体。

    主要功能：
        1. 根据状态选择动作；
        2. 从经验回放池采样训练；
        3. 维护在线网络 policy_net 和目标网络 target_net；
        4. 保存和加载模型。

    当前实现采用 Double DQN：
        policy_net 选择下一状态动作，target_net 估计该动作价值。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_size: int = 80_000,
        grad_clip: float = 10.0,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.policy_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_target_network()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """epsilon-greedy 动作选择。

        训练时以 epsilon 概率随机探索；评估时 epsilon=0，选择 Q 值最大动作。
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def learn(self) -> float | None:
        """从经验回放池采样并更新一次 Q 网络。"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size,
            self.device,
        )

        # 当前动作价值 Q(s,a)。
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN 目标：在线网络选动作，目标网络估值。
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def update_target_network(self) -> None:
        """同步目标网络参数。"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path, config: UAVEnvConfig | None = None) -> None:
        """保存模型和环境配置。"""
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "config": config_to_dict(config) if config is not None else None,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        """加载已训练模型。"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["policy_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
