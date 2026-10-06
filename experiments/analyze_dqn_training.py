"""Post-training summaries for a saved DQN episode log and checkpoint."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median

import numpy as np
import torch

from env.faults import FaultDataset


ROLLING_FIELDS = (
    "episode_return",
    "recovery_success",
    "episode_steps",
    "new_policy_damage_events",
    "service_damage_actions",
    "unique_newly_damaged_policies",
    "post_recovery_damage_events",
    "mean_loss",
)


def _number(row: dict[str, str], key: str) -> float | None:
    value = row[key]
    if value == "":
        return None
    if value in ("True", "False"):
        return float(value == "True")
    return float(value)


def _mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_number(row, key) for row in rows]
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _segment(rows: list[dict[str, str]], start: int, end: int) -> dict[str, object]:
    selected = rows[start - 1:end]
    reasons = Counter(row["termination_reason"] for row in selected)
    return {
        "episodes": f"{start}-{end}",
        "mean_return": _mean(selected, "episode_return"),
        "recovery_success_rate": _mean(selected, "recovery_success"),
        "mean_episode_steps": _mean(selected, "episode_steps"),
        "incorrect_declare_rate": reasons["incorrect_declare_done"] / len(selected),
        "timeout_rate": reasons["step_limit"] / len(selected),
        "mean_new_policy_damage_events": _mean(selected, "new_policy_damage_events"),
        "mean_service_damage_actions": _mean(selected, "service_damage_actions"),
        "mean_unique_newly_damaged_policies": _mean(
            selected, "unique_newly_damaged_policies"
        ),
        "mean_loss": _mean(selected, "mean_loss"),
    }


def analyze(
    training_csv: str | Path = "results/dqn_training_seed_2026.csv",
    checkpoint: str | Path = "results/dqn_vanilla_seed_2026.pt",
    curve_csv: str | Path = "results/dqn_learning_curve_seed_2026.csv",
) -> dict[str, object]:
    with Path(training_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    curve_path = Path(curve_csv)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_fields = ["episode", "window_start", "window_size"] + [
        f"{field}_rolling_100" for field in ROLLING_FIELDS
    ]
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            window = rows[max(0, index - 99):index + 1]
            output: dict[str, object] = {
                "episode": int(row["episode"]) + 1,
                "window_start": int(window[0]["episode"]) + 1,
                "window_size": len(window),
            }
            output.update(
                {f"{field}_rolling_100": _mean(window, field) for field in ROLLING_FIELDS}
            )
            writer.writerow(output)

    intervals = [
        _segment(rows, 1, 100),
        _segment(rows, 101, 200),
        _segment(rows, 201, 400),
        _segment(rows, 401, 600),
        _segment(rows, 601, 800),
        _segment(rows, 801, 1000),
    ]
    last = rows[-100:]
    damage = np.asarray(
        [float(row["new_policy_damage_events"]) for row in last], dtype=np.float64
    )
    reasons = Counter(row["termination_reason"] for row in last)
    last_summary = {
        "mean_return": _mean(last, "episode_return"),
        "median_return": median(float(row["episode_return"]) for row in last),
        "recovery_success_rate": _mean(last, "recovery_success"),
        "mean_steps": _mean(last, "episode_steps"),
        "median_steps": median(float(row["episode_steps"]) for row in last),
        "timeout_rate": reasons["step_limit"] / len(last),
        "incorrect_declare_rate": reasons["incorrect_declare_done"] / len(last),
        "mean_new_policy_damage_events": float(damage.mean()),
        "median_new_policy_damage_events": float(np.median(damage)),
        "p95_new_policy_damage_events": float(np.percentile(damage, 95)),
        "max_new_policy_damage_events": float(damage.max()),
        "mean_unique_newly_damaged_policies": _mean(
            last, "unique_newly_damaged_policies"
        ),
        "mean_service_damage_actions": _mean(last, "service_damage_actions"),
        "mean_post_recovery_damage_events": _mean(
            last, "post_recovery_damage_events"
        ),
        "mean_loss": _mean(last, "mean_loss"),
    }

    sampled = Counter(row["scenario_id"] for row in rows)
    dataset = FaultDataset()
    train_ids = {item.scenario_id for item in dataset.stratified_split("train", 2026)}
    eval_ids = {item.scenario_id for item in dataset.stratified_split("eval", 2026)}
    train_sample_counts = [sampled[scenario_id] for scenario_id in train_ids]
    losses = [
        float(row["mean_loss"]) for row in rows if row["mean_loss"] != ""
    ]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    q_tensors = list(saved["online_network_state_dict"].values()) + list(
        saved["target_network_state_dict"].values()
    )
    summary: dict[str, object] = {
        "episodes": len(rows),
        "final_global_step": int(rows[-1]["global_step"]),
        "final_epsilon": float(rows[-1]["epsilon"]),
        "intervals": intervals,
        "last_100": last_summary,
        "loss": {
            "count": len(losses),
            "all_finite": bool(np.isfinite(losses).all()),
            "min": min(losses),
            "max": max(losses),
            "first_100_available_mean": fmean(losses[:100]),
            "last_100_mean": fmean(losses[-100:]),
        },
        "network_parameters": {
            "all_finite": all(torch.isfinite(tensor).all().item() for tensor in q_tensors),
            "max_abs": max(tensor.abs().max().item() for tensor in q_tensors),
        },
        "coverage": {
            "unique_train_scenarios": len(sampled),
            "train_total": len(train_ids),
            "coverage_percent": 100.0 * len(sampled) / len(train_ids),
            "minimum_samples": min(train_sample_counts),
            "maximum_samples": max(train_sample_counts),
            "mean_samples": fmean(train_sample_counts),
            "unknown_or_eval_samples": sorted(set(sampled) - train_ids),
            "sampled_eval_scenarios": sorted(set(sampled) & eval_ids),
        },
        "curve_csv": str(curve_path),
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
