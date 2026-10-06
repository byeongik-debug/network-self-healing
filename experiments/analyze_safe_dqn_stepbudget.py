"""Analysis and evaluation for the 35,747-transition Safe DQN control run."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from agents.dqn import DQNAgent
from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.evaluate_dqn import EvaluationConfig, action_name
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator
from safety.viability_shield import MinimumDamageViabilityShield


TRAIN_CSV = Path("results/safe_dqn_stepbudget_35747_seed_2026.csv")
CHECKPOINT = Path("results/safe_dqn_stepbudget_35747_seed_2026.pt")
SUMMARY = Path("results/safe_dqn_stepbudget_35747_seed_2026_summary.json")
EVAL_CSV = Path("results/safe_dqn_stepbudget_35747_seed_2026_eval.csv")
EVAL_SUMMARY = Path("results/safe_dqn_stepbudget_35747_seed_2026_eval_summary.json")
EVAL_TRAJECTORIES = Path("results/safe_dqn_stepbudget_35747_seed_2026_eval_trajectories.json")
FAILURE_ID = ("access_vlan:SW1:eth1:20+access_vlan:SW2:eth2:10+"
              "trunk_remove:SW1:eth3:10+trunk_remove:SW1:eth3:20+"
              "trunk_remove:SW2:eth3:10+trunk_remove:SW2:eth3:20")


def num(row, key):
    value = row[key]
    if value == "":
        return None
    if value in ("True", "False"):
        return float(value == "True")
    return float(value)


def summarize(rows):
    if not rows:
        return {"complete_episodes": 0}
    reasons = Counter(row["termination_reason"] for row in rows)
    losses = [num(row, "mean_loss") for row in rows if row["mean_loss"]]
    return {
        "complete_episodes": len(rows),
        "recovery_rate": fmean(num(r, "recovery_success") for r in rows),
        "mean_return": fmean(num(r, "episode_return") for r in rows),
        "mean_steps": fmean(num(r, "episode_steps") for r in rows),
        "incorrect_declare_rate": reasons["incorrect_declare_done"] / len(rows),
        "timeout_rate": reasons["step_limit"] / len(rows),
        "mean_new_damage": fmean(num(r, "new_policy_damage_events") for r in rows),
        "mean_loss": fmean(losses) if losses else None,
        "mean_interventions": fmean(num(r, "shield_interventions") for r in rows),
    }


def initial_q_ranking(checkpoint, scenario):
    cfg = checkpoint["training_config"]
    agent = DQNAgent(gamma=cfg["gamma"], learning_rate=cfg["learning_rate"])
    agent.online_network.load_state_dict(checkpoint["online_network_state_dict"])
    agent.online_network.eval()
    env = NetworkEnv(split="eval", fault_scenario=scenario, max_steps=30)
    observation, _ = env.reset(seed=2026)
    state = ObservationEncoder().encode(observation)
    shield = MinimumDamageViabilityShield()
    valid = np.asarray(env.get_action_mask(), dtype=np.bool_)
    combined = shield.get_combined_mask(env, valid)
    with torch.no_grad():
        q = agent.online_network(torch.as_tensor(state).unsqueeze(0))[0].numpy()
    ranking = sorted(np.flatnonzero(combined), key=lambda action: (-q[action], action))
    return [{"rank": rank + 1, "action_id": int(action),
             "action": action_name(ACTIONS[action]), "q_value": float(q[action]),
             "is_diagnose": ACTIONS[action].kind == "diagnose",
             "is_configuration_progress": ACTIONS[action].kind in (
                 "set_access_vlan", "set_trunk_vlan", "rollback")}
            for rank, action in enumerate(ranking)]


def analyze_and_evaluate():
    with TRAIN_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row["completed"] == "True"]
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    dataset = FaultDataset()
    train = dataset.stratified_split("train", 2026)
    evaluation = dataset.stratified_split("eval", 2026)
    sampled = Counter(row["scenario_id"] for row in rows)
    counts = [sampled[item.scenario_id] for item in train]

    edges = (0, 5000, 10000, 15000, 20000, 25000, 30000, 35747)
    intervals = []
    for lo, hi in zip(edges, edges[1:]):
        selected = [row for row in complete if lo < int(row["global_step"]) <= hi]
        item = summarize(selected)
        item.update({"range": f"{lo}-{hi}", "transitions": hi - lo,
                     "assignment": "complete episodes by ending global_step"})
        intervals.append(item)

    post = [row for row in complete if int(row["start_global_step"]) >= 20000]
    all_losses = [num(row, "mean_loss") for row in rows if row["mean_loss"]]
    tensors = list(checkpoint["online_network_state_dict"].values()) + list(
        checkpoint["target_network_state_dict"].values())
    failed_scenario = next(s for s in evaluation if s.scenario_id == FAILURE_ID)

    evaluator = ShieldedDQNEvaluator(
        EvaluationConfig(checkpoint_path=str(CHECKPOINT), output_csv=str(EVAL_CSV),
                         summary_json=str(EVAL_SUMMARY),
                         trajectories_json=str(EVAL_TRAJECTORIES)),
        output_csv=str(EVAL_CSV), output_summary=str(EVAL_SUMMARY),
        output_trajectories=str(EVAL_TRAJECTORIES))
    evaluated = evaluator.evaluate()
    failed_trajectory = next(t for t in evaluated["trajectories"]
                             if t["scenario_id"] == FAILURE_ID)
    ranking = initial_q_ranking(checkpoint, failed_scenario)

    summary = {
        "experiment": {"seed": 2026, "fresh_initialization": True,
                       "checkpoint_resume": False, "target_global_step": 35747,
                       "termination_policy": checkpoint["termination_policy"],
                       "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()},
        "budget": {"episodes_started": len(rows), "episodes_completed": len(complete),
                   "partial_episodes": sum(r["budget_cut"] == "True" for r in rows),
                   "global_steps": int(rows[-1]["global_step"]),
                   "optimizer_updates": sum(int(r["optimizer_updates"]) for r in rows),
                   "epsilon_floor_global_step": 20000,
                   "post_floor_transitions": 15747,
                   "post_floor_complete_episodes": len(post),
                   "target_sync_count": 35747 // 500, "replay_size": 10000},
        "coverage": {"unique": len(sampled), "total_train": len(train),
                     "min_samples": min(counts), "max_samples": max(counts),
                     "mean_samples": fmean(counts),
                     "eval_training_exposure": sum(sampled[s.scenario_id] for s in evaluation)},
        "step_intervals": intervals,
        "post_epsilon_floor": summarize(post),
        "safety_invariants": {
            "viability_violation_count": sum(int(r["viability_violation_count"]) for r in rows),
            "empty_combined_mask_count": sum(int(r["empty_combined_mask_count"]) for r in rows),
            "empty_progress_action_count": sum(int(r["empty_progress_action_count"]) for r in rows)},
        "numerical_stability": {"all_losses_finite": bool(np.isfinite(all_losses).all()),
            "max_loss": max(all_losses), "recent_mean_loss": fmean(all_losses[-500:]),
            "parameters_finite": all(torch.isfinite(t).all().item() for t in tensors),
            "max_abs_parameter": max(t.abs().max().item() for t in tensors)},
        "evaluation": evaluated["summary"],
        "phase_4_3_failure_scenario": {"trajectory": failed_trajectory,
                                      "initial_combined_q_ranking": ranking},
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze_and_evaluate(), indent=2, ensure_ascii=False))
