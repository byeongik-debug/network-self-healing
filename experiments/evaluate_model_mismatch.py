"""Evaluate Progress-Safe DQN under stochastic command no-op failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from env.action_failure_env import ActionFailureNetworkEnv
from env.faults import FaultDataset
from experiments.evaluate_dqn import EvaluationConfig
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from safety.progress_viability_shield import ProgressViabilityShield


SEEDS = tuple(range(2026, 2036))
PROBABILITIES = (0.0, 0.1, 0.2, 0.3)
TRIALS = 10


def run() -> tuple[list[dict[str, object]], dict[str, object]]:
    scenarios = FaultDataset().stratified_split("eval", 2026)
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        checkpoint = f"results/multiseed_progress/progress_safe_seed_{seed}.pt"
        evaluator = ShieldedDQNEvaluator(
            EvaluationConfig(checkpoint_path=checkpoint),
            shield=ProgressViabilityShield(),
        )
        for probability in PROBABILITIES:
            for trial in range(TRIALS):
                for scenario_index, scenario in enumerate(scenarios):
                    reset_seed = seed * 100_000 + int(probability * 100) * 1_000 \
                        + trial * len(scenarios) + scenario_index
                    env = ActionFailureNetworkEnv(
                        split="eval",
                        fault_scenario=scenario,
                        max_steps=30,
                        action_failure_probability=probability,
                    )
                    observation, _ = env.reset(seed=reset_seed)
                    state = evaluator.encoder.encode(observation)
                    terminated = truncated = False
                    for _ in range(30):
                        decision = evaluator.shield.get_decision(
                            env, np.asarray(env.get_action_mask(), dtype=np.bool_)
                        )
                        action_id = evaluator.agent.select_action(
                            state, decision.combined_mask, 0.0
                        )
                        observation, _, terminated, truncated, _ = env.step(action_id)
                        state = evaluator.encoder.encode(observation)
                        if terminated or truncated:
                            break
                    metrics = env.get_metrics()
                    rows.append({
                        "checkpoint_seed": seed,
                        "failure_probability": probability,
                        "trial": trial,
                        "scenario_id": scenario.scenario_id,
                        "recovery_success": env.simulator.is_policy_healthy(),
                        "declared_success": bool(terminated and metrics["policy_healthy"]),
                        "timeout": bool(truncated),
                        "steps": int(metrics["steps"]),
                        "action_failures": int(metrics["action_failures"]),
                        "new_policy_damage_events": int(metrics["new_policy_damage_events"]),
                    })

    aggregate: dict[str, object] = {}
    for probability in PROBABILITIES:
        selected = [row for row in rows if row["failure_probability"] == probability]
        aggregate[str(probability)] = {
            "evaluations": len(selected),
            "functional_recovery_rate": fmean(bool(row["recovery_success"]) for row in selected),
            "declared_success_rate": fmean(bool(row["declared_success"]) for row in selected),
            "timeout_rate": fmean(bool(row["timeout"]) for row in selected),
            "mean_steps": fmean(int(row["steps"]) for row in selected),
            "p95_steps": float(np.percentile([int(row["steps"]) for row in selected], 95)),
            "mean_action_failures": fmean(int(row["action_failures"]) for row in selected),
            "total_damage": sum(int(row["new_policy_damage_events"]) for row in selected),
            "mean_damage": fmean(int(row["new_policy_damage_events"]) for row in selected),
        }
    report = {
        "design": {
            "checkpoint_seeds": list(SEEDS),
            "failure_probabilities": list(PROBABILITIES),
            "trials_per_seed_scenario": TRIALS,
            "scenarios": len(scenarios),
            "failure_mode": "configuration command consumes a step but is a no-op",
            "max_steps": 30,
        },
        "aggregate": aggregate,
    }
    return rows, report


def main() -> None:
    torch.set_num_threads(1)
    rows, report = run()
    csv_path = Path("results/model_mismatch_scenario_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path("results/model_mismatch_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
