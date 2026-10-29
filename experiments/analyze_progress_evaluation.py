"""Aggregate the 10-seed frozen-checkpoint progress-shield evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np


SEEDS = tuple(range(2026, 2036))


def bootstrap_mean_ci(values: list[float], seed: int = 2026) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def main() -> None:
    original_rows = list(csv.DictReader(
        Path("results/multiseed/multiseed_eval_seed_summary.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    original = {
        int(row["seed"]): row for row in original_rows if row["algorithm"] == "safe"
    }
    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        summary = json.loads(
            Path(f"results/progress_eval_seed_{seed}/summary.json").read_text(
                encoding="utf-8"
            )
        )
        prior = original[seed]
        rows.append({
            "seed": seed,
            "original_recovery": float(prior["functional_recovery"]),
            "progress_recovery": float(summary["overall"]["functional_recovery_rate"]),
            "original_timeout": float(prior["timeout"]),
            "progress_timeout": float(summary["overall"]["timeout_rate"]),
            "original_total_damage": int(prior["total_damage"]),
            "progress_total_damage": int(summary["optimality"]["realized_total_damage"]),
            "progress_safety_gap": int(summary["optimality"]["realized_minus_theoretical"]),
            "progress_mean_steps": float(summary["overall"]["mean_steps"]),
            "progress_configuration_recovery": float(
                summary["overall"]["configuration_recovery_rate"]
            ),
            "progress_interventions": int(summary["shield"]["total_interventions"]),
        })

    output_csv = Path("results/progress_multiseed_summary.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "progress_recovery", "progress_timeout", "progress_total_damage",
        "progress_mean_steps", "progress_configuration_recovery",
        "progress_interventions",
    )
    aggregate = {
        metric: {
            "mean": fmean(float(row[metric]) for row in rows),
            "bootstrap_95_ci": bootstrap_mean_ci(
                [float(row[metric]) for row in rows]
            ),
            "min": min(float(row[metric]) for row in rows),
            "max": max(float(row[metric]) for row in rows),
        }
        for metric in metrics
    }
    report = {
        "design": {
            "seeds": list(SEEDS),
            "checkpoint_policy": "frozen Safe DQN checkpoints; no retraining",
            "evaluation_scenarios_per_seed": 51,
            "bootstrap_resamples": 20_000,
        },
        "gates": {
            "perfect_recovery_seeds": sum(row["progress_recovery"] == 1 for row in rows),
            "zero_timeout_seeds": sum(row["progress_timeout"] == 0 for row in rows),
            "zero_safety_gap_seeds": sum(row["progress_safety_gap"] == 0 for row in rows),
        },
        "aggregate": aggregate,
        "seed_2033": next(row for row in rows if row["seed"] == 2033),
        "interpretation": (
            "This isolates the inference-time shield effect. It does not establish "
            "that training with the progress shield is equivalent; fresh paired "
            "training remains required."
        ),
    }
    Path("results/progress_multiseed_statistical_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
