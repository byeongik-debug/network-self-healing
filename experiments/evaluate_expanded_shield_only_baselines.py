"""Learning-free T4/T6 baselines for the expanded study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from baselines.expanded import run_shielded_reference_baseline
from env.expanded.actions import build_action_catalog
from env.expanded.dataset import build_scenarios
from env.expanded.environment import ExpandedAction, ExpandedNetworkEnv
from env.expanded.shield import SafetyMaskCache
from env.expanded.topologies import build_topology
from experiments.evaluate_graph_dqn import EVALUATION_CONDITIONS
from experiments.evaluate_t6_zero_shot import representative_scenarios


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("shielded_config_diff", "shield_random", "progress_shield_random")


def _scenario_suite() -> list[tuple[str, str, tuple, bool]]:
    suite = [
        (scenario.scenario_id, "T4-Ring", scenario.faults, scenario.initially_healthy)
        for scenario in build_scenarios() if scenario.split == "test"
    ]
    topology = build_topology("T6-Mesh")
    for scenario in representative_scenarios():
        env = ExpandedNetworkEnv(topology)
        env.reset(scenario.faults)
        suite.append((scenario.scenario_id, "T6-Mesh", scenario.faults,
                      env.simulator.is_policy_healthy()))
    return suite


def _run_random(
    env: ExpandedNetworkEnv, catalog: tuple, rng: np.random.Generator,
    *, require_progress: bool,
) -> dict:
    shield = SafetyMaskCache(catalog)
    terminated = truncated = False
    success = False
    while env.metrics()["steps"] < env.max_steps:
        mask = shield.mask(env, require_progress=require_progress)
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            break
        action_id = int(rng.choice(allowed))
        _, _, terminated, truncated, info = env.step(catalog[action_id].action)
        if terminated or truncated:
            success = bool(info.get("recovery_success", False))
            break
    return {**env.metrics(), "success": success, "terminated": terminated,
            "truncated": truncated, "safety_deadlock": not terminated and not truncated}


def evaluate(selector_seed: int = 2026, max_steps: int = 30) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(selector_seed)
    for index, (scenario_id, topology_name, faults, initially_healthy) in enumerate(
        _scenario_suite()
    ):
        topology = build_topology(topology_name)
        catalog = build_action_catalog(
            topology, acl_scope="policy" if topology_name == "T6-Mesh" else "all"
        )
        for condition, parameters in EVALUATION_CONDITIONS.items():
            for algorithm in ALGORITHMS:
                env = ExpandedNetworkEnv(topology, max_steps=max_steps, **parameters)
                env.reset(faults, seed=selector_seed + index)
                if algorithm == "shielded_config_diff":
                    result = run_shielded_reference_baseline(env)
                else:
                    result = _run_random(
                        env, catalog, rng,
                        require_progress=algorithm == "progress_shield_random",
                    )
                rows.append({
                    "scenario_id": scenario_id, "topology": topology_name,
                    "condition": condition, "algorithm": algorithm,
                    "selector_seed": selector_seed,
                    "initially_healthy": initially_healthy, **result,
                })
    return rows


def summarize(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for topology in ("T4-Ring", "T6-Mesh"):
        for condition in EVALUATION_CONDITIONS:
            for algorithm in ALGORITHMS:
                group = [row for row in rows if row["topology"] == topology
                         and row["condition"] == condition
                         and row["algorithm"] == algorithm]
                result[f"{topology}:{condition}:{algorithm}"] = {
                    "episodes": len(group),
                    "success_rate": fmean(bool(row["success"]) for row in group),
                    "mean_steps": fmean(int(row["steps"]) for row in group),
                    "timeouts": sum(bool(row["truncated"]) for row in group),
                    "damage": sum(int(row["new_policy_damage_events"]) for row in group),
                    "safety_deadlocks": sum(bool(row["safety_deadlock"]) for row in group),
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/expanded/shield_only_baselines")
    parser.add_argument("--selector-seed", type=int, default=2026)
    args = parser.parse_args()
    rows = evaluate(args.selector_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = (args.output_dir / "scenario_results.csv",
                           args.output_dir / "summary.json")
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite existing expanded baseline results")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "protocol": {"selector_seed": args.selector_seed, "max_steps": 30,
                     "shield_view": "true", "random_seed_count": 1,
                     "conditions": list(EVALUATION_CONDITIONS)},
        "results": summarize(rows),
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
