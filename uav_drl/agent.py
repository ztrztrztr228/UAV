# -*- coding: utf-8 -*-
"""DQN 智能体和经验回放池。"""

from __future__ import annotations

import errno
import os
import random
import time
import uuid
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from .config import UAVEnvConfig, config_to_dict
from .model import QNetwork


# ==================== Checkpoint 原子保存辅助配置 ====================
_CHECKPOINT_REPLACE_RETRY_DELAYS = (0.0, 0.25, 0.5, 1.0, 2.0)
_TRANSIENT_CHECKPOINT_ERROR_CODES = {
    errno.EACCES,
    errno.EBUSY,
    5,     # Windows ERROR_ACCESS_DENIED
    32,    # Windows ERROR_SHARING_VIOLATION
    1224,  # Windows ERROR_USER_MAPPED_FILE
}


def _is_transient_checkpoint_error(exc: OSError) -> bool:
    codes = {exc.errno, getattr(exc, "winerror", None)}
    return bool(codes & _TRANSIENT_CHECKPOINT_ERROR_CODES)


def _atomic_replace_checkpoint(source: Path, target: Path) -> None:
    """Replace a checkpoint, retrying transient Windows scanner/file-map locks."""
    last_error: OSError | None = None
    for delay in _CHECKPOINT_REPLACE_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if not _is_transient_checkpoint_error(exc):
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


# ==================== PyTorch 计算设备选择 ====================
def resolve_device(device: str | None = None) -> torch.device:
    """Resolve an explicit or automatic compute device with a clear CUDA error."""
    requested = (device or "cpu").strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot use it. "
            f"Installed torch={torch.__version__}, CUDA build={torch.version.cuda}. "
            "Install the CUDA PyTorch build or run with --device cpu."
        )
    try:
        resolved = torch.device(requested)
    except RuntimeError as exc:
        raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>.") from exc
    if resolved.type == "cuda" and resolved.index is not None:
        if resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {resolved.index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def device_description(device: torch.device) -> str:
    """Return a concise human-readable description of the selected device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return (
            f"{device} ({torch.cuda.get_device_name(index)}, "
            f"PyTorch {torch.__version__}, CUDA {torch.version.cuda})"
        )
    return f"{device} (PyTorch {torch.__version__})"


# ==================== 经验回放池 ====================
class ReplayBuffer:
    """经验回放池。

    存储 (state, action, reward, next_state, done)，训练时随机采样 batch，
    用来打破轨迹数据的时间相关性，提高 DQN 训练稳定性。
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.buffer: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
        ] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def state_dict(self) -> dict[str, object]:
        """Return serializable replay-buffer state for checkpoint resume."""
        return {
            "capacity": self.buffer.maxlen,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "items": list(self.buffer),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        # 从 checkpoint 恢复经验；同时兼容旧版五字段 transition。
        capacity = int(state.get("capacity") or self.buffer.maxlen or 0)
        items = state.get("items", [])
        migrated_items = []
        for item in items:
            if len(item) == 5:
                old_state, action, reward, old_next_state, done = item
                next_action_mask = np.ones(self.action_dim, dtype=bool)
            elif len(item) == 6:
                old_state, action, reward, old_next_state, done, next_action_mask = item
            else:
                raise ValueError("Checkpoint replay transition has an unsupported format.")
            migrated_items.append(
                (
                    self._migrate_state(old_state),
                    int(action),
                    float(reward),
                    self._migrate_state(old_next_state),
                    bool(done),
                    self._migrate_action_mask(next_action_mask),
                )
            )
        self.buffer = deque(migrated_items, maxlen=capacity)

    def _migrate_state(self, state: np.ndarray) -> np.ndarray:
        # 旧状态缺少末尾新增特征时补零，确保能继续训练 50 维网络。
        array = np.asarray(state, dtype=np.float32).reshape(-1)
        if len(array) > self.state_dim:
            raise ValueError(
                f"Replay state dimension {len(array)} exceeds current dimension {self.state_dim}."
            )
        if len(array) < self.state_dim:
            array = np.pad(array, (0, self.state_dim - len(array)))
        return array.astype(np.float32, copy=False)

    def _migrate_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        # 旧经验没有可靠掩码时退化为“所有动作可选”。
        array = np.asarray(action_mask, dtype=bool).reshape(-1)
        if len(array) != self.action_dim:
            return np.ones(self.action_dim, dtype=bool)
        if not np.any(array):
            return np.ones(self.action_dim, dtype=bool)
        return array

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        """保存一条交互经验。"""
        # 经验除标准五元组外，还保存下一状态动作掩码供 Double DQN 计算目标值。
        mask = (
            np.ones(self.action_dim, dtype=bool)
            if next_action_mask is None
            else self._migrate_action_mask(next_action_mask)
        )
        self.buffer.append(
            (
                self._migrate_state(state),
                int(action),
                float(reward),
                self._migrate_state(next_state),
                bool(done),
                mask.copy(),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """随机采样一个 batch，并转换为 tensor。"""
        # Python 随机采样后一次堆叠，统一搬运到训练设备。
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)
        state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_state_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
        next_action_mask_tensor = torch.as_tensor(
            np.stack(next_action_masks), dtype=torch.bool, device=device
        )
        return (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            done_tensor,
            next_action_mask_tensor,
        )


# ==================== Double DQN 智能体 ====================
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
        self.device = resolve_device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        # 在线网络负责选动作和学习，目标网络负责生成较稳定的监督目标。
        self.policy_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim=state_dim, action_dim=action_dim)
        self.update_target_network()

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """epsilon-greedy 动作选择。

        训练时以 epsilon 概率随机探索；评估时 epsilon=0，选择 Q 值最大动作。
        """
        # 先把安全掩码转成合法动作集合，随机探索和贪心策略共用这一集合。
        valid_mask = (
            np.ones(self.action_dim, dtype=bool)
            if action_mask is None
            else np.asarray(action_mask, dtype=bool).reshape(-1)
        )
        if len(valid_mask) != self.action_dim:
            raise ValueError(
                f"Action mask must have {self.action_dim} entries, got {len(valid_mask)}."
            )
        valid_actions = np.flatnonzero(valid_mask)
        if not len(valid_actions):
            raise ValueError("Action mask must allow at least one action.")
        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))
        self.policy_net.eval()
        with torch.no_grad():
            # 网络输出全部动作价值，无效动作设为负无穷后再取最大值。
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            return int(q_values.argmax(dim=1).item())

    def learn(self) -> float | None:
        """从经验回放池采batch size个样本并用他们更新一次 Q 网络。"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.policy_net.train()
        # 从回放池抽取无时间相关性的随机 batch。
        states, actions, rewards, next_states, dones, next_action_masks = self.replay_buffer.sample(
            self.batch_size,
            self.device,
        )

        # 当前动作价值 Q(s,a)。
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN 目标：在线网络选动作，目标网络估值。
            next_policy_values = self.policy_net(next_states).masked_fill(
                ~next_action_masks,
                -torch.inf,
            )
            next_actions = next_policy_values.argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        # Huber loss 抑制异常 TD 误差，再通过梯度裁剪限制一次更新幅度。
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def update_target_network(self) -> None:
        """同步目标网络参数。"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(
        self,
        path: str | Path,
        config: UAVEnvConfig | None = None,
        extra_state: dict[str, object] | None = None,
    ) -> Path:
        """Safely save a checkpoint without truncating the current good file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # checkpoint 同时保存网络、优化器、经验池、环境配置和续训进度。
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "replay_transition_version": 2,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "config": config_to_dict(config) if config is not None else None,
        }
        if extra_state:
            payload.update(extra_state)
        temporary_path = target_path.with_name(
            f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            # Serialize to a new file first. The existing checkpoint remains intact
            # even if PyTorch, Defender, an indexer, or Windows interrupts this step.
            torch.save(payload, temporary_path)
            try:
                _atomic_replace_checkpoint(temporary_path, target_path)
                return target_path
            except OSError as exc:
                if not _is_transient_checkpoint_error(exc):
                    raise
                recovery_path = target_path.with_name(
                    f"{target_path.stem}_recovery_{time.strftime('%Y%m%d_%H%M%S')}_"
                    f"{uuid.uuid4().hex[:8]}{target_path.suffix}"
                )
                os.replace(temporary_path, recovery_path)
                warnings.warn(
                    f"Checkpoint {target_path} stayed locked after retries; the same "
                    f"training state was preserved at {recovery_path}. Training will continue.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return recovery_path
        finally:
            # A failed serialization must not leave large temporary files behind.
            temporary_path.unlink(missing_ok=True)

    def _load_network_with_appended_state_features(
        self,
        network: QNetwork,
        loaded_state: dict[str, torch.Tensor],
    ) -> bool:
        """Load a network, zero-initializing columns for newly appended state features."""
        # 仅允许第一层输入列变多；新增列权重置零，其余参数必须完全匹配。
        current_state = network.state_dict()
        migrated = False
        compatible_state: dict[str, torch.Tensor] = {}
        for name, current_value in current_state.items():
            if name not in loaded_state:
                raise ValueError(f"Checkpoint network is missing parameter {name!r}.")
            loaded_value = loaded_state[name]
            if loaded_value.shape == current_value.shape:
                compatible_state[name] = loaded_value
                continue
            if (
                name == "net.0.weight"
                and loaded_value.ndim == 2
                and current_value.ndim == 2
                and loaded_value.shape[0] == current_value.shape[0]
                and loaded_value.shape[1] < current_value.shape[1]
            ):
                expanded = torch.zeros_like(current_value)
                expanded[:, : loaded_value.shape[1]] = loaded_value.to(expanded.device)
                compatible_state[name] = expanded
                migrated = True
                continue
            raise ValueError(
                f"Checkpoint parameter {name!r} has shape {tuple(loaded_value.shape)}, "
                f"expected {tuple(current_value.shape)}."
            )
        network.load_state_dict(compatible_state)
        return migrated

    def load(
        self,
        path: str | Path,
        discard_replay_on_state_migration: bool = False,
    ) -> dict[str, object]:
        """加载已训练模型。"""
        # mmap=False 可避免 Windows 对 checkpoint 建立长期文件映射而阻塞后续覆盖。
        try:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
                mmap=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        checkpoint_state_dim = int(checkpoint.get("state_dim", self.state_dim))
        if checkpoint_state_dim > self.state_dim:
            raise ValueError(
                "Checkpoint state dimension exceeds the current dynamics environment."
            )
        if int(checkpoint.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the environment.")
        # 分别迁移在线网络和目标网络，必要时按设置丢弃不可靠的旧经验池。
        try:
            migrated_policy = self._load_network_with_appended_state_features(
                self.policy_net,
                checkpoint["policy_net"],
            )
            migrated_target = self._load_network_with_appended_state_features(
                self.target_net,
                checkpoint.get("target_net", checkpoint["policy_net"]),
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Checkpoint network shape is incompatible with the phase-2 dynamics model."
            ) from exc
        migrated = bool(migrated_policy or migrated_target or checkpoint_state_dim != self.state_dim)
        if "optimizer" in checkpoint and not migrated:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        replay_discarded = bool(migrated and discard_replay_on_state_migration)
        if "replay_buffer" in checkpoint and not replay_discarded:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        elif replay_discarded:
            self.replay_buffer.clear()
        checkpoint["state_dim_migrated_from"] = checkpoint_state_dim if migrated else None
        checkpoint["replay_buffer_discarded"] = replay_discarded
        return checkpoint
