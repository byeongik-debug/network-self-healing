"""Evaluate expanded oracle baseline and write paper-ready scenario records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from baselines.expanded import run_reference_baseline, run_shielded_reference_baseline
from env.expanded.dataset import build_scenarios
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.topologies import build_topology


CONDITIONS = {
    "nominal": {},
    "failure_10": {"command_failure_probability": 0.10},
    "failure_30": {"command_failure_probability": 0.30},
    "delay_stale": {"actuation_delay": 2, "telemetry_delay": 2,
                    "stale_probability": 0.50},
    "combined": {"command_failure_probability": 0.20, "actuation_delay": 2,
                 "telemetry_delay": 2, "stale_probability": 0.50},
}


def evaluate(seed: int = 2026) -> list[dict]:
    rows: list[dict] = []
    for scenario_index, scenario in enumerate(build_scenarios()):
        for condition, parameters in CONDITIONS.items():
            for algorithm, runner in (("oracle", run_reference_baseline),
                                      ("shielded_oracle", run_shielded_reference_baseline)):
                env = ExpandedNetworkEnv(build_topology(scenario.topology), max_steps=60,
                                         **parameters)
                env.reset(scenario.faults, seed=seed + scenario_index)
                result = runner(env)
                rows.append({
                    "scenario_id": scenario.scenario_id,
                    "split": scenario.split,
                    "topology": scenario.topology,
                    "fault_count": len(scenario.faults),
                    "initially_healthy": scenario.initially_healthy,
                    "condition": condition,
                    "algorithm": algorithm,
                    **result,
                })
    return rows


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["condition"], row.get("algorithm", "unknown"))].append(row)
    return {
        f"{split}:{condition}:{algorithm}": {
            "episodes": len(group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "mean_steps": sum(int(row["steps"]) for row in group) / len(group),
            "total_action_failures": sum(int(row["action_failures"]) for row in group),
            "total_new_policy_damage_events": sum(
                int(row["new_policy_damage_events"]) for row in group
            ),
            "safety_deadlocks": sum(bool(row.get("safety_deadlock", False)) for row in group),
        }
        for (split, condition, algorithm), group in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/expanded"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate(args.seed)
    csv_path = args.output_dir / "oracle_scenario_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.output_dir / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
