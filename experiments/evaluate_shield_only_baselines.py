"""Learning-free Eval-51 baselines for isolating shield contribution."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable

import numpy as np

from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from experiments.evaluation import configuration_recovery, functional_recovery
from safety.progress_viability_shield import ProgressViabilityShield
from safety.viability_shield import MinimumDamageViabilityShield


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "progress_random", "progress_first", "minimum_damage_random", "valid_random",
)
RANDOM_SEEDS = tuple(range(2026, 2036))


@dataclass(frozen=True)
class BaselineRow:
    method: str
    selector_seed: int
    scenario_id: str
    fault_type: str
    severity: str
    fault_count: int
    functional_recovery: bool
    configuration_recovery: bool
    new_policy_damage_events: int
    steps: int
    timeout: bool
    terminated: bool


def _mask(method: str, env: NetworkEnv, shield: object | None) -> np.ndarray:
    valid = np.asarray(env.get_action_mask(), dtype=np.bool_)
    if method == "valid_random":
        return valid
    assert isinstance(shield, (ProgressViabilityShield, MinimumDamageViabilityShield))
    return shield.get_decision(env, valid).combined_mask


def evaluate(
    method: str,
    scenarios: Iterable[FaultScenario],
    selector_seed: int,
    *,
    max_steps: int = 30,
) -> list[BaselineRow]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    shield = (ProgressViabilityShield() if method.startswith("progress_")
              else MinimumDamageViabilityShield() if method == "minimum_damage_random"
              else None)
    rng = np.random.default_rng(selector_seed)
    rows: list[BaselineRow] = []
    for scenario in scenarios:
        env = NetworkEnv(split="eval", fault_scenario=scenario, max_steps=max_steps)
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(max_steps):
            admissible = np.flatnonzero(_mask(method, env, shield))
            if not len(admissible):
                break
            action = int(admissible[0] if method == "progress_first"
                         else rng.choice(admissible))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        metrics = env.get_metrics()
        rows.append(BaselineRow(
            method, selector_seed, scenario.scenario_id, scenario.fault_type,
            scenario.severity, scenario.fault_count,
            functional_recovery(env.simulator),
            configuration_recovery(env.simulator.snapshot()),
            int(metrics["new_policy_damage_events"]), int(metrics["steps"]),
            bool(truncated), bool(terminated),
        ))
    return rows


def summarize(rows: list[BaselineRow]) -> dict[str, object]:
    successes = [row for row in rows if row.functional_recovery]
    return {
        "episodes": len(rows),
        "functional_recoveries": len(successes),
        "functional_recovery_rate": len(successes) / len(rows),
        "total_new_policy_damage_events": sum(row.new_policy_damage_events for row in rows),
        "damage_per_successful_episode": (
            sum(row.new_policy_damage_events for row in successes) / len(successes)
            if successes else None
        ),
        "mean_steps": fmean(row.steps for row in rows),
        "timeouts": sum(row.timeout for row in rows),
        "exact_configuration_recoveries": sum(row.configuration_recovery for row in rows),
        "exact_configuration_recovery_rate": fmean(row.configuration_recovery for row in rows),
    }


def run(output_dir: Path, *, save: bool = True) -> dict[str, object]:
    scenarios = FaultDataset().stratified_split("eval", 2026)
    all_rows: dict[str, list[BaselineRow]] = {method: [] for method in METHODS}
    per_seed: dict[str, dict[str, dict[str, object]]] = {method: {} for method in METHODS}
    for method in METHODS:
        seeds = (2026,) if method == "progress_first" else RANDOM_SEEDS
        for seed in seeds:
            rows = evaluate(method, scenarios, seed)
            all_rows[method].extend(rows)
            per_seed[method][str(seed)] = summarize(rows)
            if save:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"{method}_seed_{seed}.csv"
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite existing result: {path}")
                with path.open("x", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
                    writer.writeheader()
                    writer.writerows(asdict(row) for row in rows)
    summary = {
        "protocol": {
            "dataset": "FaultDataset().stratified_split('eval', 2026)",
            "scenario_count": len(scenarios), "environment_reset_seed": 0,
            "random_selector_seeds": list(RANDOM_SEEDS), "max_steps": 30,
            "learning": False,
        },
        "methods": {
            method: {"aggregate": summarize(rows), "per_seed": per_seed[method]}
            for method, rows in all_rows.items()
        },
    }
    if save:
        path = output_dir / "summary.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing result: {path}")
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/shield_only_baselines")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
