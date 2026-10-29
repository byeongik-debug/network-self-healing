"""Paper-facing comparison of Vanilla, Safe, and fresh Progress-Safe runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.stats import rankdata, wilcoxon


SEEDS = tuple(range(2026, 2036))
ROOT = Path("results")


def bootstrap_mean(values: list[float], seed: int = 2026) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "bootstrap_95_ci": [float(x) for x in np.quantile(means, (0.025, 0.975))],
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_wilcoxon(left: list[float], right: list[float]) -> dict[str, object]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    test = wilcoxon(differences, zero_method="wilcox", method="auto")
    nonzero = differences[differences != 0]
    ranks = rankdata(abs(nonzero))
    effect = (
        float(ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
        / float(ranks.sum())
    )
    return {
        "difference_definition": "left - right",
        "differences": differences.tolist(),
        "statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
        "rank_biserial": effect,
    }


def main() -> None:
    original = list(csv.DictReader(
        (ROOT / "multiseed/multiseed_eval_seed_summary.csv").open(
            newline="", encoding="utf-8"
        )
    ))
    lookup = {(row["algorithm"], int(row["seed"])): row for row in original}
    rows: list[dict[str, object]] = []
    training_integrity: list[dict[str, object]] = []
    for seed in SEEDS:
        for algorithm in ("vanilla", "safe"):
            source = lookup[(algorithm, seed)]
            rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "functional_recovery": float(source["functional_recovery"]),
                "timeout": float(source["timeout"]),
                "total_damage": int(source["total_damage"]),
                "mean_steps": float(source["mean_steps"]),
                "configuration_recovery": float(source["configuration_recovery"]),
                "safety_gap": (
                    float(source["safety_optimality_gap"])
                    if source["safety_optimality_gap"] else None
                ),
            })
        progress = json.loads(
            (ROOT / f"multiseed_progress/progress_safe_seed_{seed}_summary.json")
            .read_text(encoding="utf-8")
        )
        evaluation = progress["evaluation"]
        rows.append({
            "algorithm": "progress_safe",
            "seed": seed,
            "functional_recovery": float(evaluation["functional_recovery_rate"]),
            "timeout": float(evaluation["timeout_rate"]),
            "total_damage": int(progress["total_damage"]),
            "mean_steps": float(evaluation["mean_steps"]),
            "configuration_recovery": float(evaluation["configuration_recovery_rate"]),
            "safety_gap": int(progress["total_damage"])
            - int(progress["theoretical_minimum_damage"]),
        })
        training_integrity.append({
            "seed": seed,
            "global_steps": int(progress["global_steps"]),
            "optimizer_updates": int(progress["optimizer_updates"]),
            "viability_violations": int(progress["viability_violations"]),
            "completed_episodes": int(progress["completed_episodes"]),
            "training_seconds": float(progress["training_seconds"]),
        })

    output_csv = ROOT / "fresh_progress_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_algorithm = {
        algorithm: [row for row in rows if row["algorithm"] == algorithm]
        for algorithm in ("vanilla", "safe", "progress_safe")
    }
    metrics = (
        "functional_recovery", "timeout", "total_damage", "mean_steps",
        "configuration_recovery",
    )
    aggregate = {
        algorithm: {
            **{
                metric: bootstrap_mean([float(row[metric]) for row in selected])
                for metric in metrics
            },
            "perfect_recovery_seeds": sum(
                row["functional_recovery"] == 1 for row in selected
            ),
        }
        for algorithm, selected in by_algorithm.items()
    }
    vanilla_damage = [float(row["total_damage"]) for row in by_algorithm["vanilla"]]
    progress_damage = [float(row["total_damage"]) for row in by_algorithm["progress_safe"]]
    report = {
        "design": {
            "seeds": list(SEEDS),
            "transition_budget": 35_747,
            "eval_scenarios_per_seed": 51,
            "split_seed": 2026,
            "bootstrap_resamples": 20_000,
        },
        "aggregate": aggregate,
        "paired_vanilla_minus_progress_damage": paired_wilcoxon(
            vanilla_damage, progress_damage
        ),
        "progress_training_integrity": {
            "all_exact_budget": all(x["global_steps"] == 35_747 for x in training_integrity),
            "all_optimizer_updates_34748": all(
                x["optimizer_updates"] == 34_748 for x in training_integrity
            ),
            "total_viability_violations": sum(
                x["viability_violations"] for x in training_integrity
            ),
            "training_runtime_seconds": bootstrap_mean(
                [x["training_seconds"] for x in training_integrity]
            ),
            "per_seed": training_integrity,
        },
        "reporting_warning": (
            "The original Safe seed 2033 damage is low because recovery failed. "
            "Do not interpret unconditional Safe-vs-Progress damage without the "
            "joint recovery outcome."
        ),
    }
    (ROOT / "fresh_progress_statistical_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
