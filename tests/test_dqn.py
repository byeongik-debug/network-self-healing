from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.dqn import DQNAgent, QNetwork
from agents.replay_buffer import ReplayBuffer


def add_transition(
    buffer: ReplayBuffer,
    *,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones(19, dtype=bool)
    buffer.push(
        np.ones(79, dtype=np.float32),
        0,
        reward,
        np.full(79, 2.0, dtype=np.float32),
        terminated,
        truncated,
        mask,
    )


def test_networks_start_equal_and_q_network_supports_batches() -> None:
    agent = DQNAgent()
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        )
    )
    assert QNetwork()(torch.zeros(7, 79)).shape == (7, 19)


def test_exploration_selects_only_valid_actions() -> None:
    agent = DQNAgent()
    mask = np.zeros(19, dtype=bool)
    mask[[2, 8]] = True
    selected = {agent.select_action(np.zeros(79), mask, 1.0) for _ in range(100)}
    assert selected <= {2, 8}


def test_exploitation_masks_invalid_highest_q_value() -> None:
    agent = DQNAgent()
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        agent.online_network.network[-1].bias.copy_(torch.arange(19.0))
    mask = np.ones(19, dtype=bool)
    mask[18] = False
    assert agent.select_action(np.zeros(79), mask, 0.0) == 17


def test_select_action_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="at least one valid action"):
        DQNAgent().select_action(np.zeros(79), np.zeros(19, dtype=bool), 0.0)


def test_update_changes_only_online_until_target_sync_and_returns_finite_loss() -> None:
    agent = DQNAgent()
    buffer = ReplayBuffer(1)
    add_transition(buffer)
    target_before = [parameter.detach().clone() for parameter in agent.target_network.parameters()]

    loss = agent.update(buffer, 1)

    assert np.isfinite(loss)
    assert any(
        not torch.equal(online, target)
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, agent.target_network.parameters())
    )
    agent.update_target_network()
    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.online_network.parameters(), agent.target_network.parameters()
        )
    )


def test_terminal_empty_mask_is_safe_but_nonterminal_empty_mask_is_rejected() -> None:
    empty_mask = np.zeros(19, dtype=bool)
    terminal_buffer = ReplayBuffer(1)
    add_transition(terminal_buffer, terminated=True, mask=empty_mask)
    assert np.isfinite(DQNAgent().update(terminal_buffer, 1))

    nonterminal_buffer = ReplayBuffer(1)
    add_transition(nonterminal_buffer, mask=empty_mask)
    with pytest.raises(ValueError, match="non-terminal transition"):
        DQNAgent().update(nonterminal_buffer, 1)


def test_terminated_stops_bootstrap_but_truncated_does_not() -> None:
    agent = DQNAgent(gamma=0.5)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        for parameter in agent.target_network.parameters():
            parameter.zero_()
        agent.target_network.network[-1].bias.fill_(4.0)

    terminal = ReplayBuffer(1)
    add_transition(terminal, reward=1.0, terminated=True)
    truncated = ReplayBuffer(1)
    add_transition(truncated, reward=1.0, truncated=True)

    terminal_loss = agent.update(terminal, 1)
    # Restore online Q(s,a)=0 so the losses directly expose each target.
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
    truncated_loss = agent.update(truncated, 1)

    assert terminal_loss == pytest.approx(0.5)  # Huber(0, 1)
    assert truncated_loss == pytest.approx(2.5)  # Huber(0, 1 + .5 * 4)


def test_bellman_max_uses_next_action_mask() -> None:
    agent = DQNAgent(gamma=1.0)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        for parameter in agent.target_network.parameters():
            parameter.zero_()
        agent.target_network.network[-1].bias.copy_(torch.arange(19.0))
    mask = np.zeros(19, dtype=bool)
    mask[3] = True
    buffer = ReplayBuffer(1)
    add_transition(buffer, reward=0.0, mask=mask)
    assert agent.update(buffer, 1) == pytest.approx(2.5)  # Huber(0, 3)
