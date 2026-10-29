"""Record Q rankings and actions for the two T6 ACL zero-shot failures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from agents.graph_dqn import GraphDQNAgent
from baselines.expanded import reference_repairs
from env.expanded.actions import build_action_catalog
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.shield import SafetyMaskCache, policy_violation_count
from env.expanded.topologies import build_topology
from experiments.evaluate_t6_zero_shot import representative_scenarios


def diagnose(checkpoint_path: Path) -> dict:
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
    action_ids = {candidate.action: candidate.action_id for candidate in catalog}
    encoder = GraphObservationEncoder(
        topology, catalog, include_policy_features=policy_features
    )
    acl_scenarios = [scenario for scenario in representative_scenarios()
                     if len(scenario.faults) == 1 and scenario.faults[0].kind.startswith("acl_")]
    episodes = []
    for index, scenario in enumerate(acl_scenarios):
        env = ExpandedNetworkEnv(topology, max_steps=30)
        observation, _ = env.reset(scenario.faults, seed=int(config["seed"]) + 30000 + index)
        shield = SafetyMaskCache(catalog)
        trajectory = []
        for step in range(30):
            encoded = encoder.encode(observation)
            mask = shield.mask(env)
            with torch.no_grad():
                values = agent.online_network(encoded).cpu().numpy()
            valid_ids = np.flatnonzero(mask)
            ranked = valid_ids[np.argsort(values[valid_ids])[::-1]]
            selected = int(ranked[0])
            rank_by_id = {int(action_id): rank + 1 for rank, action_id in enumerate(ranked)}
            repairs = [action for action in reference_repairs(env) if action in action_ids]
            before = policy_violation_count(env.simulator)
            next_observation, _, terminated, truncated, info = env.step(
                catalog[selected].action
            )
            trajectory.append({
                "step": step + 1,
                "selected_id": selected,
                "selected_action": asdict(catalog[selected].action),
                "selected_q": float(values[selected]),
                "top_valid": [
                    {"action_id": int(action_id),
                     "action": asdict(catalog[int(action_id)].action),
                     "q": float(values[int(action_id)])}
                    for action_id in ranked[:5]
                ],
                "repair_candidates": [
                    {"action_id": action_ids[action], "action": asdict(action),
                     "q": float(values[action_ids[action]]),
                     "valid_rank": rank_by_id.get(action_ids[action])}
                    for action in repairs
                ],
                "valid_action_count": int(mask.sum()),
                "violations_before": before,
                "violations_after": policy_violation_count(env.simulator),
            })
            observation = next_observation
            if terminated or truncated:
                break
        episodes.append({
            "scenario_id": scenario.scenario_id,
            "fault_ids": [fault.fault_id for fault in scenario.faults],
            "success": env.simulator.is_policy_healthy() and bool(trajectory),
            "final_policy_healthy": env.simulator.is_policy_healthy(),
            "steps": env.metrics()["steps"],
            "cache_hits": shield.hits,
            "cache_misses": shield.misses,
            "trajectory": trajectory,
        })
    return {"checkpoint_seed": int(config["seed"]), "topology": "T6-Mesh",
            "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/expanded/t6_zero_shot/acl_failure_diagnostic.json"))
    args = parser.parse_args()
    result = diagnose(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "episodes": len(result["episodes"]),
        "steps": [episode["steps"] for episode in result["episodes"]],
        "healthy": [episode["final_policy_healthy"] for episode in result["episodes"]],
    }, indent=2))


if __name__ == "__main__":
    main()
