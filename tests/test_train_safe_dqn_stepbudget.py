from __future__ import annotations

import numpy as np

from env.faults import FaultDataset
from experiments.train_dqn import TrainingConfig
from experiments.train_safe_dqn_stepbudget import StepBudgetSafeDQNTrainer


def config(tmp_path, **overrides):
    values = dict(episodes=1, batch_size=2, replay_capacity=100,
                  replay_warmup=2, epsilon_decay_steps=20,
                  target_sync_interval=5, max_episode_steps=4, seed=2026,
                  output_dir=str(tmp_path), save_csv=False, save_checkpoint=False)
    values.update(overrides)
    return TrainingConfig(**values)


def test_exact_transition_budget_can_cut_last_episode(tmp_path):
    trainer = StepBudgetSafeDQNTrainer(config(tmp_path), target_steps=17)
    result = trainer.train()
    assert result.global_step == 17
    assert sum(log.episode_steps for log in result.logs) == 17
    assert result.logs[-1].budget_cut
    assert not result.logs[-1].completed
    assert result.logs[-1].termination_reason == "budget_cut"


def test_stepbudget_split_masks_and_truncation_semantics(tmp_path):
    trainer = StepBudgetSafeDQNTrainer(config(tmp_path), target_steps=17)
    result = trainer.train()
    train_ids = {s.scenario_id for s in FaultDataset().stratified_split("train", 2026)}
    eval_ids = {s.scenario_id for s in FaultDataset().stratified_split("eval", 2026)}
    assert set(result.sampled_scenario_ids) <= train_ids
    assert set(result.sampled_scenario_ids).isdisjoint(eval_ids)
    assert sum(log.viability_violation_count for log in result.logs) == 0
    assert sum(log.empty_combined_mask_count for log in result.logs) == 0
    for transition in trainer.replay_buffer._buffer:
        if transition.terminated:
            assert not np.any(transition.next_action_mask)
        else:
            assert np.any(transition.next_action_mask)


def test_stepbudget_fresh_seed_is_deterministic(tmp_path):
    first = StepBudgetSafeDQNTrainer(config(tmp_path / "a"), 31).train()
    second = StepBudgetSafeDQNTrainer(config(tmp_path / "b"), 31).train()
    assert first.sampled_scenario_ids == second.sampled_scenario_ids
    assert first.logs == second.logs

