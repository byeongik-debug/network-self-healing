from __future__ import annotations

import numpy as np
import torch

from agents.dqn import DQNAgent
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.train_dqn import TrainingConfig
from experiments.train_safe_dqn import SafeDQNTrainer, train_safe_dqn
from safety.viability_shield import MinimumDamageViabilityShield


def smoke_config(tmp_path, **overrides) -> TrainingConfig:
    values = {
        "episodes": 12,
        "batch_size": 2,
        "replay_capacity": 100,
        "replay_warmup": 2,
        "epsilon_decay_steps": 20,
        "target_sync_interval": 5,
        "max_episode_steps": 4,
        "seed": 2026,
        "output_dir": str(tmp_path),
        "save_csv": False,
        "save_checkpoint": False,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_exploration_and_exploitation_use_only_combined_mask() -> None:
    env = NetworkEnv(fault_scenario=FaultDataset().stratified_split("eval", 2026)[3])
    observation, _ = env.reset(seed=2026)
    combined = MinimumDamageViabilityShield().get_combined_mask(env)
    blocked = np.flatnonzero(np.asarray(env.get_action_mask()) & ~combined)
    assert blocked.size
    agent = DQNAgent()
    state = ObservationEncoder().encode(observation)
    assert all(agent.select_action(state, combined, 1.0) in np.flatnonzero(combined) for _ in range(100))
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        agent.online_network.network[-1].bias[blocked[0]] = 100.0
    assert agent.select_action(state, combined, 0.0) in np.flatnonzero(combined)


def test_safe_smoke_training_updates_and_stores_combined_next_masks(tmp_path) -> None:
    trainer = SafeDQNTrainer(smoke_config(tmp_path))
    result = trainer.train()
    assert len(result.logs) == 12
    assert result.optimizer_updates == result.global_step - 1
    assert result.target_sync_steps == tuple(range(5, result.global_step + 1, 5))
    assert sum(log.viability_violation_count for log in result.logs) == 0
    assert all(log.empty_combined_mask_count == 0 for log in result.logs)
    transitions = list(trainer.replay_buffer._buffer)
    assert transitions
    assert all(
        (not transition.next_action_mask.any()) if transition.terminated
        else transition.next_action_mask.any()
        for transition in transitions
    )


def test_safe_training_uses_train_split_only(tmp_path) -> None:
    result = train_safe_dqn(smoke_config(tmp_path, replay_warmup=100))
    dataset = FaultDataset()
    train_ids = {item.scenario_id for item in dataset.stratified_split("train", 2026)}
    eval_ids = {item.scenario_id for item in dataset.stratified_split("eval", 2026)}
    assert set(result.sampled_scenario_ids) <= train_ids
    assert set(result.sampled_scenario_ids).isdisjoint(eval_ids)


def test_safe_smoke_training_is_seed_reproducible(tmp_path) -> None:
    first = train_safe_dqn(smoke_config(tmp_path / "a", replay_warmup=100))
    second = train_safe_dqn(smoke_config(tmp_path / "b", replay_warmup=100))
    assert first.sampled_scenario_ids == second.sampled_scenario_ids
    assert first.logs == second.logs
