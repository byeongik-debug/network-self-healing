from __future__ import annotations

import torch

from agents.dqn import DQNAgent
from env.faults import FaultDataset
from experiments.evaluate_dqn import DQNEvaluator, EvaluationConfig
from experiments.train_dqn import TrainingConfig


def checkpoint(tmp_path):
    agent = DQNAgent()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "online_network_state_dict": agent.online_network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "training_config": TrainingConfig(episodes=1).__dict__,
            "seed": 2026,
            "global_step": 0,
        },
        path,
    )
    return path


def config(tmp_path, model_path) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint_path=str(model_path),
        output_csv=str(tmp_path / "eval.csv"),
        summary_json=str(tmp_path / "summary.json"),
        trajectories_json=str(tmp_path / "trajectories.json"),
        max_episode_steps=2,
    )


def test_evaluation_uses_only_eval_scenarios_and_keeps_model_unchanged(tmp_path) -> None:
    model_path = checkpoint(tmp_path)
    evaluator = DQNEvaluator(config(tmp_path, model_path))
    result = evaluator.evaluate()
    train_ids = {item.scenario_id for item in evaluator.train_scenarios}
    evaluated_ids = {row.scenario_id for row in result["rows"]}
    assert len(evaluated_ids) == 51
    assert train_ids.isdisjoint(evaluated_ids)
    assert result["summary"]["evaluation"]["epsilon"] == 0.0
    assert result["summary"]["evaluation"]["parameters_unchanged"]
    assert (tmp_path / "eval.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "trajectories.json").exists()


def test_repeated_evaluation_is_deterministic(tmp_path) -> None:
    model_path = checkpoint(tmp_path)
    scenarios = FaultDataset().stratified_split("eval", 2026)[:3]
    evaluator = DQNEvaluator(
        config(tmp_path, model_path), scenarios=scenarios
    )
    first = evaluator.evaluate(save=False)
    second = evaluator.evaluate(save=False)
    assert first["rows"] == second["rows"]
    assert first["trajectories"] == second["trajectories"]
