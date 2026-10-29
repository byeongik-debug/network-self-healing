"""Aggregate matched-budget expanded Graph DQN runs at the seed level."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = tuple(range(2026, 2031))


def _bootstrap(values: list[float], *, seed: int = 2026, draws: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def _stats(values: list[float]) -> dict[str, object]:
    return {
        "mean": fmean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "seed_bootstrap_95ci": _bootstrap(values),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for seed in seeds:
        t4_dir = (root / "results/expanded/graph_dqn/eval_300ep_full" if seed == 2026
                  else root / f"results/expanded/graph_dqn/eval_seed_{seed}_ep300_full")
        t6_dir = (root / "results/expanded/t6_zero_shot" if seed == 2026
                  else root / f"results/expanded/t6_zero_shot_seed_{seed}")
        training_path = root / f"results/expanded/graph_dqn/training_seed_{seed}_ep300.csv"
        with training_path.open(newline="", encoding="utf-8") as handle:
            training = list(csv.DictReader(handle))
        training_success = sum(row["success"].lower() == "true" for row in training) / len(training)
        training_damage = sum(int(row["new_policy_damage_events"]) for row in training)
        for topology, path in (
            ("T4-Ring", t4_dir / "heldout_evaluation_summary.json"),
            ("T6-Mesh", t6_dir / "summary.json"),
        ):
            summary = _load_json(path)
            for condition in ("nominal", "combined"):
                item = summary[condition]
                rows.append({
                    "seed": seed,
                    "training_success_rate": training_success,
                    "training_damage": training_damage,
                    "topology": topology,
                    "condition": condition,
                    "episodes": item["episodes"],
                    "success_rate": item["success_rate"],
                    "mean_steps": item["mean_steps"],
                    "damage": item["damage"],
                })
    groups: dict[str, dict] = {}
    for topology in ("T4-Ring", "T6-Mesh"):
        for condition in ("nominal", "combined"):
            group = [row for row in rows
                     if row["topology"] == topology and row["condition"] == condition]
            groups[f"{topology}:{condition}"] = {
                "seeds": len(group),
                "episodes_per_seed": group[0]["episodes"],
                "total_episodes": sum(row["episodes"] for row in group),
                "success_rate": _stats([row["success_rate"] for row in group]),
                "mean_steps": _stats([row["mean_steps"] for row in group]),
                "total_damage": sum(row["damage"] for row in group),
            }
    summary = {
        "scope": {
            "seeds": list(seeds), "training_episodes_per_seed": 300,
            "evaluation_guidance": False, "statistical_unit": "seed",
            "bootstrap_draws": 20_000,
        },
        "training": {
            "success_rate": _stats(sorted({row["seed"]: row["training_success_rate"] for row in rows}.values())),
            "total_damage": sum({row["seed"]: row["training_damage"] for row in rows}.values()),
        },
        "evaluation": groups,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/expanded/multiseed")
    args = parser.parse_args()
    rows, summary = collect(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
