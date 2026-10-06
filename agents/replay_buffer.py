"""Replay buffer for DQN transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
import random

import numpy as np
from numpy.typing import NDArray


STATE_DIM = 79
ACTION_DIM = 19


@dataclass(frozen=True)
class Transition:
    state: NDArray[np.float32]
    action: int
    reward: float
    next_state: NDArray[np.float32]
    terminated: bool
    truncated: bool
    next_action_mask: NDArray[np.bool_]


class ReplayBuffer:
    """Fixed-capacity storage for observable DQN transitions."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, Integral):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._buffer: deque[Transition] = deque(maxlen=int(capacity))

    def push(
        self,
        state: NDArray[np.float32],
        action: int,
        reward: float,
        next_state: NDArray[np.float32],
        terminated: bool,
        truncated: bool,
        next_action_mask: NDArray[np.bool_],
    ) -> None:
        """Store one transition, copying array inputs to prevent aliasing."""
        state_array = self._state_array(state, "state")
        next_state_array = self._state_array(next_state, "next_state")
        mask_array = np.asarray(next_action_mask, dtype=np.bool_)
        if mask_array.shape != (ACTION_DIM,):
            raise ValueError(
                f"next_action_mask must have shape ({ACTION_DIM},), got {mask_array.shape}"
            )
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= action < ACTION_DIM:
            raise ValueError(f"action must be between 0 and {ACTION_DIM - 1}")
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError("reward must be a real number")
        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError("terminated must be a bool")
        if not isinstance(truncated, (bool, np.bool_)):
            raise TypeError("truncated must be a bool")

        self._buffer.append(
            Transition(
                state=state_array,
                action=int(action),
                reward=float(reward),
                next_state=next_state_array,
                terminated=bool(terminated),
                truncated=bool(truncated),
                next_action_mask=mask_array.copy(),
            )
        )

    def sample(
        self, batch_size: int
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Return a uniformly sampled batch as seven NumPy arrays."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer of size "
                f"{len(self._buffer)}"
            )

        transitions = random.sample(tuple(self._buffer), int(batch_size))
        return (
            np.stack([item.state for item in transitions]).astype(np.float32, copy=False),
            np.asarray([item.action for item in transitions], dtype=np.int64),
            np.asarray([item.reward for item in transitions], dtype=np.float32),
            np.stack([item.next_state for item in transitions]).astype(np.float32, copy=False),
            np.asarray([item.terminated for item in transitions], dtype=np.bool_),
            np.asarray([item.truncated for item in transitions], dtype=np.bool_),
            np.stack([item.next_action_mask for item in transitions]).astype(
                np.bool_, copy=False
            ),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    @staticmethod
    def _state_array(state: NDArray[np.float32], name: str) -> NDArray[np.float32]:
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (STATE_DIM,):
            raise ValueError(f"{name} must have shape ({STATE_DIM},), got {array.shape}")
        return array.copy()
