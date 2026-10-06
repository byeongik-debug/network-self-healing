"""Convergence diagnostics for the seed-2026 3000-episode DQN run."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median

import numpy as np
import torch

from agents.dqn import QNetwork
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.analyze_dqn_training import _mean


METRICS = (
    "episode_return",
    "recovery_success",
    "episode_steps",
    "new_policy_damage_events",
    "unique_newly_damaged_policies",
    "service_damage_actions",
    "post_recovery_damage_events",
    "mean_loss",
)


def _rates(rows: list[dict[str, str]]) -> tuple[float, float]:
    reasons = Counter(row["termination_reason"] for row in rows)
    return (
        reasons["incorrect_declare_done"] / len(rows),
        reasons["step_limit"] / len(rows),
    )


def _summary(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    incorrect, timeout = _rates(rows)
    return {
        "episodes": len(rows),
        "mean_return": _mean(rows, "episode_return"),
        "recovery_success_rate": _mean(rows, "recovery_success"),
        "mean_steps": _mean(rows, "episode_steps"),
        "incorrect_declare_rate": incorrect,
        "timeout_rate": timeout,
        "mean_new_policy_damage_events": _mean(rows, "new_policy_damage_events"),
        "mean_unique_newly_damaged_policies": _mean(
            rows, "unique_newly_damaged_policies"
        ),
        "mean_service_damage_actions": _mean(rows, "service_damage_actions"),
        "mean_post_recovery_damage_events": _mean(
            rows, "post_recovery_damage_events"
        ),
        "mean_loss": _mean(rows, "mean_loss"),
    }


def analyze() -> dict[str, object]:
    training_path = Path("results/dqn_training_seed_2026_ep3000.csv")
    old_path = Path("results/dqn_training_seed_2026.csv")
    checkpoint_path = Path("results/dqn_vanilla_seed_2026_ep3000.pt")
    curve_path = Path("results/dqn_learning_curve_seed_2026_ep3000.csv")
    with training_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with old_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))

    curve_fields = ["episode", "global_step", "epsilon", "window_start", "window_size"]
    curve_fields += [f"{field}_rolling_100" for field in METRICS]
    curve_fields += ["incorrect_declare_rolling_100", "timeout_rolling_100"]
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            window = rows[max(0, index - 99):index + 1]
            incorrect, timeout = _rates(window)
            output: dict[str, object] = {
                "episode": index + 1,
                "global_step": row["global_step"],
                "epsilon": row["epsilon"],
                "window_start": max(1, index - 98),
                "window_size": len(window),
                "incorrect_declare_rolling_100": incorrect,
                "timeout_rolling_100": timeout,
            }
            output.update(
                {f"{field}_rolling_100": _mean(window, field) for field in METRICS}
            )
            writer.writerow(output)

    ranges = ((1, 500), (501, 1000), (1001, 1500), (1501, 2000),
              (2001, 2500), (2501, 3000))
    intervals = []
    for start, end in ranges:
        result = _summary(rows[start - 1:end])
        result["range"] = f"{start}-{end}"
        intervals.append(result)

    final_500_rows = rows[-500:]
    final_500 = _summary(final_500_rows)
    final_500["median_return"] = median(
        float(row["episode_return"]) for row in final_500_rows
    )
    final_500["median_steps"] = median(
        float(row["episode_steps"]) for row in final_500_rows
    )
    damage = np.asarray(
        [float(row["new_policy_damage_events"]) for row in final_500_rows]
    )
    final_500.update(
        median_new_policy_damage_events=float(np.median(damage)),
        p95_new_policy_damage_events=float(np.percentile(damage, 95)),
        max_new_policy_damage_events=float(damage.max()),
        max_loss=max(float(row["mean_loss"]) for row in final_500_rows),
    )

    floor_step = 20_000
    floor_index = next(
        index for index, row in enumerate(rows) if int(row["global_step"]) > floor_step
    )
    previous_end = int(rows[floor_index - 1]["global_step"]) if floor_index else 0
    complete_post_floor = [
        row
        for index, row in enumerate(rows)
        if (int(rows[index - 1]["global_step"]) if index else 0) >= floor_step
    ]

    sampled = Counter(row["scenario_id"] for row in rows)
    dataset = FaultDataset()
    train = dataset.stratified_split("train", 2026)
    evaluation = dataset.stratified_split("eval", 2026)
    train_ids = {scenario.scenario_id for scenario in train}
    eval_ids = {scenario.scenario_id for scenario in evaluation}
    counts = [sampled[scenario_id] for scenario_id in train_ids]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    online_tensors = list(checkpoint["online_network_state_dict"].values())
    target_tensors = list(checkpoint["target_network_state_dict"].values())
    all_losses = [float(row["mean_loss"]) for row in rows if row["mean_loss"]]
    network = QNetwork()
    network.load_state_dict(checkpoint["online_network_state_dict"])
    encoder = ObservationEncoder()
    states = np.stack([
        encoder.encode(
            NetworkEnv(split="train", fault_scenario=scenario).reset(seed=2026)[0]
        )
        for scenario in train
    ])
    with torch.no_grad():
        q_values = network(torch.as_tensor(states, dtype=torch.float32))

    old_last = _summary(old_rows[-100:])
    new_last = _summary(rows[-100:])
    comparable_fields = [field for field in old_rows[0] if field != "mean_loss"]
    first_1000_reproduced = all(
        all(old[field] == new[field] for field in comparable_fields)
        and (
            (old["mean_loss"] == new["mean_loss"])
            or np.isclose(float(old["mean_loss"]), float(new["mean_loss"]), rtol=0, atol=1e-12)
        )
        for old, new in zip(old_rows, rows[:1000])
    )

    return {
        "episodes": len(rows),
        "global_steps": int(rows[-1]["global_step"]),
        "optimizer_updates": sum(int(row["optimizer_updates"]) for row in rows),
        "target_sync_count": int(rows[-1]["global_step"]) // 500,
        "final_epsilon": float(rows[-1]["epsilon"]),
        "epsilon_floor": {
            "global_step": floor_step,
            "episode_containing_first_floor_action": floor_index + 1,
            "episode_start_global_step": previous_end,
            "episodes_with_any_floor_actions": len(rows) - floor_index,
            "complete_episodes_starting_at_floor": len(complete_post_floor),
            "post_floor_environment_steps": int(rows[-1]["global_step"]) - floor_step,
            "complete_post_floor_performance": _summary(complete_post_floor),
        },
        "intervals": intervals,
        "final_500": final_500,
        "coverage": {
            "unique": len(sampled),
            "total": len(train_ids),
            "percent": 100 * len(sampled) / len(train_ids),
            "min": min(counts),
            "max": max(counts),
            "mean": fmean(counts),
            "eval_samples": sum(sampled[scenario_id] for scenario_id in eval_ids),
            "unknown_samples": sum(
                count for scenario_id, count in sampled.items() if scenario_id not in train_ids
            ),
        },
        "stability": {
            "loss_count": len(all_losses),
            "all_losses_finite": bool(np.isfinite(all_losses).all()),
            "max_loss": max(all_losses),
            "online_parameters_finite": all(
                torch.isfinite(value).all().item() for value in online_tensors
            ),
            "target_parameters_finite": all(
                torch.isfinite(value).all().item() for value in target_tensors
            ),
            "max_abs_parameter": max(
                value.abs().max().item() for value in online_tensors + target_tensors
            ),
            "q_values_finite": bool(torch.isfinite(q_values).all().item()),
            "max_abs_q_value": q_values.abs().max().item(),
            "q_min": q_values.min().item(),
            "q_max": q_values.max().item(),
        },
        "old_901_1000": old_last,
        "new_2901_3000": new_last,
        "first_1000_reproduced": first_1000_reproduced,
        "curve_csv": str(curve_path),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
