"""Guidance-free held-out evaluation for an expanded Graph DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from agents.graph_dqn import GraphDQNAgent
from env.expanded.actions import build_action_catalog
from env.expanded.dataset import build_scenarios
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.shield import SafetyMaskCache
from env.expanded.topologies import build_topology


EVALUATION_CONDITIONS = {
    "nominal": {},
    "combined": {"command_failure_probability": 0.20, "actuation_delay": 2,
                 "telemetry_delay": 2, "stale_probability": 0.50},
}


def evaluate_checkpoint(
    checkpoint_path: Path, max_steps: int = 30, scenario_limit: int | None = None,
    shield_view: str = "true",
) -> list[dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    policy_features = bool(config.get("policy_features", False))
    agent = GraphDQNAgent(
        8, 5, 13 if policy_features else 11, hidden=int(config["hidden"])
    )
    agent.online_network.load_state_dict(checkpoint["model"])
    agent.online_network.eval()
    rows: list[dict] = []
    scenarios = [scenario for scenario in build_scenarios() if scenario.split == "test"]
    if scenario_limit is not None:
        scenarios = scenarios[:scenario_limit]
    for scenario_index, scenario in enumerate(scenarios):
        topology = build_topology(scenario.topology)
        catalog = build_action_catalog(topology)
        encoder = GraphObservationEncoder(
            topology, catalog, include_policy_features=policy_features
        )
        for condition, parameters in EVALUATION_CONDITIONS.items():
            env = ExpandedNetworkEnv(topology, max_steps=max_steps, **parameters)
            shield = SafetyMaskCache(catalog, shield_view=shield_view)
            observation, _ = env.reset(
                scenario.faults, seed=int(config["seed"]) + 10000 + scenario_index
            )
            terminated = truncated = success = False
            for _ in range(max_steps):
                state = encoder.encode(observation)
                mask = shield.mask(env)
                action_id = agent.select_action(state, mask, epsilon=0.0)
                observation, _, terminated, truncated, info = env.step(
                    catalog[action_id].action
                )
                if terminated or truncated:
                    success = bool(info.get("recovery_success", False))
                    break
            rows.append({
                "scenario_id": scenario.scenario_id,
                "topology": scenario.topology,
                "condition": condition,
                "shield_view": shield_view,
                "initially_healthy": scenario.initially_healthy,
                "success": success,
                **env.metrics(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/expanded/graph_dqn"))
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--shield-view", choices=("true", "observed"), default="true")
    args = parser.parse_args()
    rows = evaluate_checkpoint(
        args.checkpoint, scenario_limit=args.scenario_limit,
        shield_view=args.shield_view,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "heldout_evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        condition: {
            "episodes": len(group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "mean_steps": sum(int(row["steps"]) for row in group) / len(group),
            "damage": sum(int(row["new_policy_damage_events"]) for row in group),
        }
        for condition in EVALUATION_CONDITIONS
        if (group := [row for row in rows if row["condition"] == condition])
    }
    (args.output_dir / "heldout_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
