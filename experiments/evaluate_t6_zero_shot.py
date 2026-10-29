"""Representative zero-shot evaluation on the six-switch mesh."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import torch

from agents.graph_dqn import GraphDQNAgent
from env.expanded.actions import build_action_catalog
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault, atomic_faults
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.shield import SafetyMaskCache
from env.expanded.topologies import build_topology
from experiments.evaluate_graph_dqn import EVALUATION_CONDITIONS


@dataclass(frozen=True)
class T6Scenario:
    scenario_id: str
    faults: tuple[ExpandedFault, ...]


def representative_scenarios() -> tuple[T6Scenario, ...]:
    topology = build_topology("T6-Mesh")
    faults = atomic_faults(topology)
    by_kind: dict[str, list[ExpandedFault]] = {}
    for fault in faults:
        by_kind.setdefault(fault.kind, []).append(fault)
    cases: list[tuple[ExpandedFault, ...]] = [
        (sorted(group, key=lambda fault: fault.fault_id)[0],)
        for _, group in sorted(by_kind.items())
    ]
    mixed = [pair for pair in combinations(faults, 2) if pair[0].kind != pair[1].kind]
    mixed.sort(key=lambda pair: sha256("|".join(f.fault_id for f in pair).encode()).hexdigest())
    cases.extend(mixed[:6])
    return tuple(T6Scenario(
        "T6-Mesh-" + sha256("|".join(f.fault_id for f in case).encode()).hexdigest()[:12],
        case,
    ) for case in cases)


def evaluate(checkpoint_path: Path, max_steps: int = 30) -> list[dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    policy_features = bool(config.get("policy_features", False))
    agent = GraphDQNAgent(
        8, 5, 13 if policy_features else 11, hidden=int(config["hidden"])
    )
    agent.online_network.load_state_dict(checkpoint["model"])
    agent.online_network.eval()
    topology = build_topology("T6-Mesh")
    catalog = build_action_catalog(topology, acl_scope="policy")
    encoder = GraphObservationEncoder(
        topology, catalog, include_policy_features=policy_features
    )
    rows = []
    for index, scenario in enumerate(representative_scenarios()):
        for condition, parameters in EVALUATION_CONDITIONS.items():
            env = ExpandedNetworkEnv(topology, max_steps=max_steps, **parameters)
            observation, _ = env.reset(
                scenario.faults, seed=int(config["seed"]) + 20000 + index
            )
            initially_healthy = env.simulator.is_policy_healthy()
            shield = SafetyMaskCache(catalog)
            started = perf_counter()
            success = False
            for _ in range(max_steps):
                state = encoder.encode(observation)
                action_id = agent.select_action(state, shield.mask(env), epsilon=0.0)
                observation, _, terminated, truncated, info = env.step(
                    catalog[action_id].action
                )
                if terminated or truncated:
                    success = bool(info.get("recovery_success", False))
                    break
            rows.append({
                "scenario_id": scenario.scenario_id,
                "fault_ids": "|".join(fault.fault_id for fault in scenario.faults),
                "fault_count": len(scenario.faults),
                "condition": condition,
                "initially_healthy": initially_healthy,
                "success": success,
                "wall_seconds": perf_counter() - started,
                "shield_cache_hits": shield.hits,
                "shield_cache_misses": shield.misses,
                **env.metrics(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/expanded/t6_zero_shot"))
    args = parser.parse_args()
    rows = evaluate(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scenario_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        condition: {
            "episodes": len(group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "mean_steps": sum(int(row["steps"]) for row in group) / len(group),
            "damage": sum(int(row["new_policy_damage_events"]) for row in group),
            "wall_seconds": sum(float(row["wall_seconds"]) for row in group),
            "shield_cache_hit_rate": sum(int(row["shield_cache_hits"]) for row in group)
            / max(1, sum(int(row["shield_cache_hits"]) + int(row["shield_cache_misses"])
                         for row in group)),
            "impacting_episodes": len(impacting := [
                row for row in group if not bool(row["initially_healthy"])
            ]),
            "impacting_success_rate": (
                sum(bool(row["success"]) for row in impacting) / len(impacting)
            ),
            "no_impact_control_episodes": len(controls := [
                row for row in group if bool(row["initially_healthy"])
            ]),
            "no_impact_control_success_rate": (
                sum(bool(row["success"]) for row in controls) / len(controls)
            ),
        }
        for condition in EVALUATION_CONDITIONS
        if (group := [row for row in rows if row["condition"] == condition])
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
