"""Produce step-level evidence for Safe DQN liveness failures.

This diagnostic is deliberately read-only: it loads a frozen checkpoint, runs
the fixed hold-out split with epsilon=0, and records Q-values, masks, progress
costs, and repeated configuration states.  It is intended to distinguish
``diagnose`` dominance from rollback cycles and shield-induced dead ends.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from experiments.evaluate_dqn import EvaluationConfig, action_name
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from safety.viability_shield import canonical_configuration


def _configuration_label(env: NetworkEnv) -> str:
    return "".join(str(int(value)) for value in canonical_configuration(env))


def diagnose_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = 2033,
    split_seed: int = 2026,
    max_episode_steps: int = 30,
) -> dict[str, object]:
    config = EvaluationConfig(
        checkpoint_path=str(checkpoint),
        split_seed=split_seed,
        reset_seed=split_seed,
        max_episode_steps=max_episode_steps,
    )
    evaluator = ShieldedDQNEvaluator(config)
    scenarios = FaultDataset().stratified_split("eval", split_seed)
    episodes: list[dict[str, object]] = []

    with torch.no_grad():
        for scenario in scenarios:
            env = NetworkEnv(
                split="eval", fault_scenario=scenario,
                max_steps=max_episode_steps,
            )
            observation, _ = env.reset(seed=split_seed)
            state = evaluator.encoder.encode(observation)
            state_visits: Counter[str] = Counter()
            steps: list[dict[str, object]] = []
            terminated = truncated = False
            final_info: dict[str, object] = {}

            for step_index in range(max_episode_steps):
                valid = np.asarray(env.get_action_mask(), dtype=np.bool_)
                decision = evaluator.shield.get_decision(env, valid)
                q_values = evaluator.agent.online_network(
                    torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                )[0].cpu().numpy()
                selected = int(
                    np.where(decision.combined_mask, q_values, -np.inf).argmax()
                )
                top_valid = int(np.where(valid, q_values, -np.inf).argmax())
                config_label = _configuration_label(env)
                state_visits[config_label] += 1
                admissible = np.flatnonzero(decision.combined_mask).tolist()
                ranked = sorted(admissible, key=lambda i: (-float(q_values[i]), i))
                candidate = decision.candidates[selected]
                steps.append({
                    "step": step_index + 1,
                    "configuration": config_label,
                    "configuration_visit": state_visits[config_label],
                    "policy_healthy": env.simulator.is_policy_healthy(),
                    "minimum_damage_cost": decision.current_cost,
                    "top_valid_action": action_name(ACTIONS[top_valid]),
                    "selected_action": action_name(ACTIONS[selected]),
                    "shield_intervened": not bool(decision.combined_mask[top_valid]),
                    "selected_q": float(q_values[selected]),
                    "selected_immediate_damage": candidate.immediate_damage,
                    "selected_successor_cost": candidate.successor_cost,
                    "admissible_ranking": [
                        {"action": action_name(ACTIONS[i]), "q": float(q_values[i])}
                        for i in ranked
                    ],
                })
                observation, _, terminated, truncated, final_info = env.step(selected)
                state = evaluator.encoder.encode(observation)
                if terminated or truncated:
                    break

            metrics = env.get_metrics()
            actions = [str(step["selected_action"]) for step in steps]
            max_same_config = max(state_visits.values(), default=0)
            functional_recovery = env.simulator.is_policy_healthy()
            episodes.append({
                "scenario_id": scenario.scenario_id,
                "recovery_success": functional_recovery,
                "declared_success": bool(final_info.get("recovery_success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "steps": len(steps),
                "diagnose_actions": actions.count("diagnose"),
                "rollback_actions": actions.count("rollback"),
                "max_configuration_revisits": max_same_config,
                "new_policy_damage_events": int(metrics["new_policy_damage_events"]),
                "trace": steps,
            })

    failures = [episode for episode in episodes if not episode["recovery_success"]]
    summary: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "split_seed": split_seed,
        "scenario_count": len(episodes),
        "recovered": len(episodes) - len(failures),
        "failed": len(failures),
        "failure_diagnose_fraction": (
            sum(int(e["diagnose_actions"]) for e in failures)
            / sum(int(e["steps"]) for e in failures)
            if failures else 0.0
        ),
        "failures_with_repeated_configuration": sum(
            int(e["max_configuration_revisits"]) > 1 for e in failures
        ),
        "episodes": episodes,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/multiseed/safe_seed_2033.pt",
    )
    parser.add_argument(
        "--output",
        default="results/multiseed/safe_seed_2033_liveness_diagnostic.json",
    )
    parser.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()
    result = diagnose_checkpoint(args.checkpoint, args.output, seed=args.seed)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
