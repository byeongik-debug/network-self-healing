"""Topology-size-independent action-value network for expanded experiments."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from agents.graph_replay_buffer import GraphReplayBuffer
from env.expanded.graph_observation import GraphObservation


class GraphActionQNetwork(nn.Module):
    """Scores a variable number of actions over a variable-size network graph."""

    def __init__(self, node_dim: int, edge_dim: int, action_dim: int, hidden: int = 96):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU())
        self.message_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
            for _ in range(2)
        ])
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.ReLU())
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, observation: GraphObservation) -> torch.Tensor:
        device = next(self.parameters()).device
        nodes = torch.as_tensor(observation.node_features, dtype=torch.float32, device=device)
        edges = torch.as_tensor(observation.edge_features, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(observation.edge_index, dtype=torch.long, device=device)
        actions = torch.as_tensor(observation.action_features, dtype=torch.float32, device=device)
        targets = torch.as_tensor(observation.action_targets, dtype=torch.long, device=device)

        node_hidden = self.node_encoder(nodes)
        edge_hidden = self.edge_encoder(edges)
        source, destination = edge_index
        for update in self.message_layers:
            aggregate = torch.zeros_like(node_hidden)
            degree = torch.zeros(node_hidden.shape[0], 1, device=device)
            aggregate.index_add_(0, destination, node_hidden[source] + edge_hidden)
            degree.index_add_(0, destination, torch.ones(destination.shape[0], 1, device=device))
            aggregate = aggregate / degree.clamp_min(1.0)
            node_hidden = update(torch.cat((node_hidden, aggregate), dim=-1))

        count = actions.shape[0]
        global_hidden = node_hidden.mean(dim=0, keepdim=True).expand(count, -1)
        zero = torch.zeros(1, node_hidden.shape[1], device=device)
        padded = torch.cat((node_hidden, zero), dim=0)
        safe_targets = targets.where(targets >= 0, node_hidden.shape[0])
        primary = padded[safe_targets[:, 0]]
        secondary = padded[safe_targets[:, 1]]
        action_hidden = self.action_encoder(actions)
        return self.scorer(torch.cat(
            (global_hidden, primary, secondary, action_hidden), dim=-1
        )).squeeze(-1)

    def select_action(
        self, observation: GraphObservation, action_mask: np.ndarray, epsilon: float = 0.0
    ) -> int:
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (observation.action_features.shape[0],):
            raise ValueError("action mask must match the observation action count")
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("at least one action must be valid")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if random.random() < epsilon:
            return int(random.choice(valid.tolist()))
        with torch.no_grad():
            values = self(observation)
            torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
            return int(values.masked_fill(~torch_mask, -torch.inf).argmax().item())


class GraphDQNAgent:
    """DQN update logic that permits different graph/action sizes per sample."""

    def __init__(
        self, node_dim: int, edge_dim: int, action_dim: int, *, hidden: int = 96,
        gamma: float = 0.99, learning_rate: float = 1e-3, device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.online_network = GraphActionQNetwork(
            node_dim, edge_dim, action_dim, hidden
        ).to(self.device)
        self.target_network = GraphActionQNetwork(
            node_dim, edge_dim, action_dim, hidden
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_target_network()
        self.target_network.eval()

    def select_action(
        self, observation: GraphObservation, action_mask: np.ndarray, epsilon: float
    ) -> int:
        return self.online_network.select_action(observation, action_mask, epsilon)

    def update(self, replay: GraphReplayBuffer, batch_size: int) -> float:
        transitions = replay.sample(batch_size)
        current_values: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for transition in transitions:
            current_values.append(self.online_network(transition.state)[transition.action])
            with torch.no_grad():
                next_value = torch.zeros((), device=self.device)
                if not transition.terminated:
                    mask = torch.as_tensor(
                        transition.next_action_mask, dtype=torch.bool, device=self.device
                    )
                    if not mask.any():
                        raise ValueError("non-terminal graph transition has no valid next action")
                    next_q = self.target_network(transition.next_state)
                    next_value = next_q.masked_fill(~mask, -torch.inf).max()
                targets.append(torch.as_tensor(transition.reward, device=self.device)
                               + self.gamma * next_value)
        loss = self.loss_fn(torch.stack(current_values), torch.stack(targets))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())
