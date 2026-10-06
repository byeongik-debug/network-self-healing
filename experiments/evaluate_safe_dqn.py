"""Evaluate the freshly trained shield-aware Safe DQN checkpoint."""

from __future__ import annotations

import json

from experiments.evaluate_dqn import EvaluationConfig
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator


def evaluate_safe_dqn():
    config = EvaluationConfig(
        checkpoint_path="results/safe_dqn_seed_2026_ep3000.pt",
        output_csv="results/safe_dqn_eval_seed_2026_ep3000.csv",
        summary_json="results/safe_dqn_eval_summary_seed_2026_ep3000.json",
        trajectories_json="results/safe_dqn_eval_trajectories_seed_2026_ep3000.json",
    )
    evaluator = ShieldedDQNEvaluator(
        config,
        output_csv=config.output_csv,
        output_summary=config.summary_json,
        output_trajectories=config.trajectories_json,
    )
    return evaluator.evaluate()


if __name__ == "__main__":
    result = evaluate_safe_dqn()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
