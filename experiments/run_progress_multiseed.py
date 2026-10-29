"""Train and evaluate one seed of the lexicographic progress-shield ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.evaluate_dqn import EvaluationConfig, summarize_rows
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from experiments.train_dqn import TrainingConfig
from experiments.train_safe_dqn_stepbudget import (
    TARGET_STEPS,
    StepBudgetSafeDQNTrainer,
)
from safety.progress_viability_shield import ProgressViabilityShield


def run(seed: int, target_steps: int = TARGET_STEPS) -> dict[str, object]:
    # The network and batches are small; a single BLAS thread avoids dominant
    # thread-pool overhead and is frozen as part of the experimental protocol.
    torch.set_num_threads(1)
    root = Path("results/multiseed_progress")
    root.mkdir(parents=True, exist_ok=True)
    shield = ProgressViabilityShield()
    config = TrainingConfig(
        episodes=1, seed=seed, split_seed=2026, output_dir=str(root)
    )
    started = perf_counter()
    trained = StepBudgetSafeDQNTrainer(
        config, target_steps, shield=shield
    ).train()
    training_seconds = perf_counter() - started
    if trained.checkpoint_path is None:
        raise RuntimeError("training did not save a checkpoint")
    checkpoint = root / f"progress_safe_seed_{seed}.pt"
    trained.checkpoint_path.replace(checkpoint)

    evaluation = ShieldedDQNEvaluator(
        EvaluationConfig(checkpoint_path=str(checkpoint)),
        shield=ProgressViabilityShield(),
        output_csv=str(root / f"progress_safe_seed_{seed}_eval.csv"),
        output_summary=str(root / f"progress_safe_seed_{seed}_eval_summary.json"),
        output_trajectories=str(root / f"progress_safe_seed_{seed}_trajectories.json"),
    ).evaluate(save=True)
    rows = evaluation["rows"]
    overall = summarize_rows(rows)
    summary: dict[str, object] = {
        "algorithm": "progress_safe",
        "seed": seed,
        "transition_budget": target_steps,
        "torch_num_threads": torch.get_num_threads(),
        "training_seconds": training_seconds,
        "global_steps": trained.global_step,
        "optimizer_updates": trained.optimizer_updates,
        "completed_episodes": trained.completed_episodes,
        "evaluation": overall,
        "recovered_scenarios": sum(row.recovery_success for row in rows),
        "total_damage": sum(row.new_policy_damage_events for row in rows),
        "theoretical_minimum_damage": sum(
            row.theoretical_minimum_damage for row in rows
        ),
        "timeouts": sum(row.timeout for row in rows),
        "viability_violations": sum(
            log.viability_violation_count for log in trained.logs
        ),
        "training_termination_reasons": {
            reason: sum(log.termination_reason == reason for log in trained.logs)
            for reason in sorted({log.termination_reason for log in trained.logs})
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "training_csv": str(trained.csv_path),
            "evaluation_csv": str(root / f"progress_safe_seed_{seed}_eval.csv"),
            "trajectories": str(root / f"progress_safe_seed_{seed}_trajectories.json"),
        },
    }
    (root / f"progress_safe_seed_{seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("--target-steps", type=int, default=TARGET_STEPS)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.target_steps), indent=2))


if __name__ == "__main__":
    main()
