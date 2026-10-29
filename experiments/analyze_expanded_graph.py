"""Create manuscript-ready summaries for expanded Graph DQN experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import fmean


def _boolean(value: object) -> bool:
    return str(value).lower() == "true"


def _bootstrap_ci(values: list[float], rng: random.Random, draws: int = 20_000) -> list[float]:
    estimates = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def evaluation_summary(rows: list[dict], seed: int = 2026) -> dict:
    result: dict[str, dict] = {}
    rng = random.Random(seed)
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        for impact, group in (
            ("all", condition_rows),
            ("impacting", [row for row in condition_rows if not _boolean(row["initially_healthy"])]),
            ("no_impact_control", [row for row in condition_rows if _boolean(row["initially_healthy"])]),
        ):
            if not group:
                continue
            successes = [float(_boolean(row["success"])) for row in group]
            steps = [float(row["steps"]) for row in group]
            result[f"{condition}:{impact}"] = {
                "episodes": len(group),
                "success_rate": fmean(successes),
                "success_rate_bootstrap_95ci": _bootstrap_ci(successes, rng),
                "mean_steps": fmean(steps),
                "mean_steps_bootstrap_95ci": _bootstrap_ci(steps, rng),
                "damage": sum(int(row["new_policy_damage_events"]) for row in group),
            }
    return result


def training_summary(rows: list[dict], bin_size: int = 50) -> list[dict]:
    bins = []
    for start in range(0, len(rows), bin_size):
        group = rows[start:start + bin_size]
        steps = sum(int(row["steps"]) for row in group)
        bins.append({
            "episode_start": start,
            "episode_end": start + len(group) - 1,
            "episodes": len(group),
            "success_rate": sum(_boolean(row["success"]) for row in group) / len(group),
            "mean_steps": fmean(float(row["steps"]) for row in group),
            "guided_action_fraction": sum(int(row["guided_actions"]) for row in group) / steps,
            "mean_loss": fmean(float(row["mean_loss"]) for row in group if row["mean_loss"]),
            "damage": sum(int(row["new_policy_damage_events"]) for row in group),
        })
    return bins


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {
        "scope": {
            "checkpoint_seed": 2026,
            "training_episodes": 300,
            "held_out_topology": "T4-Ring",
            "guidance_used_during_evaluation": False,
            "status": "preliminary_single_seed",
        },
        "evaluation": evaluation_summary(_read(args.evaluation)),
        "training_bins": training_summary(_read(args.training)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
