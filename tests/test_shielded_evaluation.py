from __future__ import annotations

from env.faults import FaultDataset
from experiments.evaluate_dqn import EvaluationConfig
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator


def test_shielded_evaluation_is_deterministic_and_does_not_update_model() -> None:
    scenarios = FaultDataset().stratified_split("eval", 2026)[:3]
    config = EvaluationConfig(max_episode_steps=10)
    evaluator = ShieldedDQNEvaluator(config, scenarios=scenarios)
    first = evaluator.evaluate(save=False)
    second = evaluator.evaluate(save=False)
    assert first["rows"] == second["rows"]
    assert first["trajectories"] == second["trajectories"]
    assert first["summary"]["evaluation"]["parameters_unchanged"]


def test_shielded_actions_realize_no_more_than_initial_minimum_cost() -> None:
    scenarios = FaultDataset().stratified_split("eval", 2026)[:5]
    evaluator = ShieldedDQNEvaluator(scenarios=scenarios)
    result = evaluator.evaluate(save=False)
    for row in result["rows"]:
        assert row.new_policy_damage_events <= row.theoretical_minimum_damage
        assert row.invalid_actions == 0
