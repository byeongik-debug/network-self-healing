"""Replay storage for variable-size graph observations and action sets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np

from env.expanded.graph_observation import GraphObservation


def copy_graph(observation: GraphObservation) -> GraphObservation:
    return GraphObservation(
        observation.node_names,
        observation.node_features.copy(),
        observation.edge_index.copy(),
        observation.edge_features.copy(),
        observation.action_features.copy(),
        observation.action_targets.copy(),
    )


@dataclass(frozen=True)
class GraphTransition:
    state: GraphObservation
    action: int
    reward: float
    next_state: GraphObservation
    terminated: bool
    truncated: bool
    next_action_mask: np.ndarray


class GraphReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: deque[GraphTransition] = deque(maxlen=capacity)

    def push(
        self,
        state: GraphObservation,
        action: int,
        reward: float,
        next_state: GraphObservation,
        terminated: bool,
        truncated: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        mask = np.asarray(next_action_mask, dtype=np.bool_)
        if mask.shape != (next_state.action_features.shape[0],):
            raise ValueError("next action mask has the wrong variable action dimension")
        if not 0 <= action < state.action_features.shape[0]:
            raise ValueError("action is outside the state's action catalog")
        self._buffer.append(GraphTransition(
            copy_graph(state), int(action), float(reward), copy_graph(next_state),
            bool(terminated), bool(truncated), mask.copy(),
        ))

    def sample(self, batch_size: int) -> tuple[GraphTransition, ...]:
        if batch_size <= 0 or batch_size > len(self._buffer):
            raise ValueError("batch size must be positive and no larger than the buffer")
        return tuple(random.sample(tuple(self._buffer), batch_size))

    def __len__(self) -> int:
        return len(self._buffer)
