from __future__ import annotations

import csv

import pytest
import torch

from env.faults import FaultDataset
from experiments.train_dqn import (
    DQNTrainer,
    TrainingConfig,
    epsilon_at_step,
    train_dqn,
)


def smoke_config(tmp_path, **overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "episodes": 3,
        "batch_size": 2,
        "replay_capacity": 32,
        "replay_warmup": 2,
        "epsilon_decay_steps": 4,
        "target_sync_interval": 2,
        "max_episode_steps": 2,
        "seed": 17,
        "output_dir": str(tmp_path),
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_epsilon_uses_global_step_and_respects_floor(tmp_path) -> None:
    config = smoke_config(tmp_path)
    assert epsilon_at_step(config, 0) == 1.0
    assert epsilon_at_step(config, 2) == pytest.approx(0.525)
    assert epsilon_at_step(config, 4) == 0.05
    assert epsilon_at_step(config, 100) == 0.05


def test_smoke_training_stores_transitions_updates_and_syncs(tmp_path) -> None:
    result = train_dqn(smoke_config(tmp_path, save_csv=False, save_checkpoint=False))
    assert len(result.logs) == 3
    assert result.replay_size == result.global_step
    assert result.optimizer_updates == result.global_step - 1
    assert result.target_sync_steps == tuple(
        range(2, result.global_step + 1, 2)
    )
    assert all(log.terminated or log.truncated for log in result.logs)
    assert all(log.episode_steps <= 2 for log in result.logs)


def test_warmup_prevents_optimizer_updates(tmp_path) -> None:
    result = train_dqn(
        smoke_config(
            tmp_path,
            episodes=2,
            replay_warmup=100,
            save_csv=False,
            save_checkpoint=False,
        )
    )
    assert result.global_step > 0
    assert result.optimizer_updates == 0
    assert all(log.mean_loss is None for log in result.logs)


def test_only_stratified_training_scenarios_are_sampled(tmp_path) -> None:
    config = smoke_config(tmp_path, save_csv=False, save_checkpoint=False)
    trainer = DQNTrainer(config)
    dataset = FaultDataset()
    train_ids = {item.scenario_id for item in dataset.stratified_split("train", 2026)}
    eval_ids = {item.scenario_id for item in dataset.stratified_split("eval", 2026)}
    assert len(train_ids) == 204
    assert len(eval_ids) == 51
    assert train_ids.isdisjoint(eval_ids)

    result = trainer.train()
    assert set(result.sampled_scenario_ids) <= train_ids
    assert set(result.sampled_scenario_ids).isdisjoint(eval_ids)


def test_csv_and_checkpoint_contain_required_fields(tmp_path) -> None:
    config = smoke_config(tmp_path, episodes=1, replay_warmup=100)
    result = train_dqn(config)
    assert result.csv_path is not None and result.csv_path.exists()
    assert result.checkpoint_path is not None and result.checkpoint_path.exists()

    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert {
        "episode", "seed", "global_step", "epsilon", "episode_return",
        "episode_steps", "terminated", "truncated", "recovery_success",
        "termination_reason", "mean_loss", "optimizer_updates",
        "actual_configuration_changes", "diagnoses", "invalid_actions",
        "new_policy_damage_events", "unique_newly_damaged_policies",
        "service_damage_actions", "post_recovery_damage_events",
        "final_damaged_policies",
    } <= rows[0].keys()

    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    assert {
        "online_network_state_dict", "target_network_state_dict",
        "optimizer_state_dict", "training_config", "seed", "global_step",
    } <= checkpoint.keys()


def test_same_seed_reproduces_sampling_and_episode_logs(tmp_path) -> None:
    first = train_dqn(
        smoke_config(
            tmp_path / "first", replay_warmup=100,
            save_csv=False, save_checkpoint=False,
        )
    )
    second = train_dqn(
        smoke_config(
            tmp_path / "second", replay_warmup=100,
            save_csv=False, save_checkpoint=False,
        )
    )
    assert first.sampled_scenario_ids == second.sampled_scenario_ids
    assert first.logs == second.logs
