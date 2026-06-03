# -*- coding: utf-8 -*-
"""DQN 神经网络模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """DQN 的 Q 值网络。

    输入是环境状态，输出是每个离散动作对应的 Q 值。
    Q 值越大，表示网络认为该动作带来的长期累计奖励越高。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回所有动作的 Q 值。"""
        return self.net(state)
