"""Vanilla DQN network and agent with action masking."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from agents.replay_buffer import ReplayBuffer


STATE_DIM = 79
ACTION_DIM = 19


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """Minimal vanilla DQN agent.

    Time-limit truncation ends collection but does not stop Bellman
    bootstrapping. Only a true MDP termination stops bootstrapping.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.online_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
    ) -> int:
        """Choose a valid action using masked epsilon-greedy selection."""
        state_array = np.asarray(state, dtype=np.float32)
        mask_array = np.asarray(action_mask, dtype=np.bool_)
        if state_array.shape != (self.state_dim,):
            raise ValueError(
                f"state must have shape ({self.state_dim},), got {state_array.shape}"
            )
        if mask_array.shape != (self.action_dim,):
            raise ValueError(
                f"action_mask must have shape ({self.action_dim},), got {mask_array.shape}"
            )
        valid_actions = np.flatnonzero(mask_array)
        if valid_actions.size == 0:
            raise ValueError("action_mask must contain at least one valid action")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist()))

        state_tensor = torch.as_tensor(state_array, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask_array, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor)
            action = q_values.masked_fill(~mask_tensor, -torch.inf).argmax(dim=1)
        return int(action.item())

    def update(self, replay_buffer: "ReplayBuffer", batch_size: int) -> float:
        """Sample one batch, update the online network, and return its loss."""
        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            _truncated,
            next_action_masks,
        ) = replay_buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        terminated_t = torch.as_tensor(
            terminated, dtype=torch.bool, device=self.device
        )
        next_masks_t = torch.as_tensor(
            next_action_masks, dtype=torch.bool, device=self.device
        )

        bootstrapped = ~terminated_t
        if torch.any(bootstrapped & ~next_masks_t.any(dim=1)):
            raise ValueError(
                "non-terminal transition has no valid action in next_action_mask"
            )

        current_q = self.online_network(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Compute next-state values only where they are used. This keeps an
        # all-False terminal mask from producing an infinite value in the loss.
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if torch.any(bootstrapped):
                next_q = self.target_network(next_states_t[bootstrapped])
                valid_next_q = next_q.masked_fill(
                    ~next_masks_t[bootstrapped], -torch.inf
                )
                next_values[bootstrapped] = valid_next_q.max(dim=1).values
            targets = rewards_t + self.gamma * next_values

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        """Hard-copy online-network parameters to the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())
