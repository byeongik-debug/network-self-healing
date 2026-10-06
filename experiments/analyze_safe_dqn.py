"""Training diagnostics for the 3000-episode shield-aware Safe DQN run."""

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


def number(row, field):
    value = row[field]
    if value == "": return None
    if value in ("True", "False"): return float(value == "True")
    return float(value)


def mean(rows, field):
    values = [number(row, field) for row in rows]
    return fmean(value for value in values if value is not None)


def summarize(rows):
    reasons = Counter(row["termination_reason"] for row in rows)
    return {
        "episodes": len(rows),
        "mean_return": mean(rows, "episode_return"),
        "recovery_rate": mean(rows, "recovery_success"),
        "mean_steps": mean(rows, "episode_steps"),
        "incorrect_declare_rate": reasons["incorrect_declare_done"] / len(rows),
        "timeout_rate": reasons["step_limit"] / len(rows),
        "mean_new_damage": mean(rows, "new_policy_damage_events"),
        "mean_unique_damage": mean(rows, "unique_newly_damaged_policies"),
        "mean_service_damage_actions": mean(rows, "service_damage_actions"),
        "mean_loss": mean(rows, "mean_loss"),
        "mean_shield_interventions": mean(rows, "shield_interventions"),
    }


def analyze():
    training = Path("results/safe_dqn_training_seed_2026_ep3000.csv")
    checkpoint_path = Path("results/safe_dqn_seed_2026_ep3000.pt")
    with training.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ranges = ((1,500),(501,1000),(1001,1500),(1501,2000),(2001,2500),(2501,3000))
    intervals = []
    for start, end in ranges:
        value = summarize(rows[start-1:end]); value["range"] = f"{start}-{end}"; intervals.append(value)
    final_rows = rows[-500:]
    final = summarize(final_rows)
    damage = np.asarray([float(row["new_policy_damage_events"]) for row in final_rows])
    losses = [float(row["mean_loss"]) for row in final_rows if row["mean_loss"]]
    final.update({
        "median_steps": median(float(row["episode_steps"]) for row in final_rows),
        "median_new_damage": float(np.median(damage)),
        "p95_new_damage": float(np.percentile(damage,95)),
        "max_new_damage": float(damage.max()),
        "max_loss": max(losses),
        "mean_admissible_actions": mean(final_rows,"mean_admissible_actions"),
        "mean_relaxations": mean(final_rows,"shield_relaxations"),
    })
    floor = 20000
    floor_index = next(i for i,row in enumerate(rows) if int(row["global_step"]) > floor)
    post = [row for i,row in enumerate(rows) if (int(rows[i-1]["global_step"]) if i else 0) >= floor]
    sampled = Counter(row["scenario_id"] for row in rows)
    dataset = FaultDataset(); train=dataset.stratified_split("train",2026); evaluation=dataset.stratified_split("eval",2026)
    train_ids={s.scenario_id for s in train}; eval_ids={s.scenario_id for s in evaluation}; counts=[sampled[i] for i in train_ids]
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    online=list(checkpoint["online_network_state_dict"].values()); target=list(checkpoint["target_network_state_dict"].values())
    all_losses=[float(row["mean_loss"]) for row in rows if row["mean_loss"]]
    network=QNetwork();network.load_state_dict(checkpoint["online_network_state_dict"]);encoder=ObservationEncoder()
    states=np.stack([encoder.encode(NetworkEnv(split="train",fault_scenario=s).reset(seed=2026)[0]) for s in train])
    with torch.no_grad():q=network(torch.as_tensor(states,dtype=torch.float32))
    summary={
        "episodes":len(rows),"global_steps":int(rows[-1]["global_step"]),
        "optimizer_updates":sum(int(row["optimizer_updates"]) for row in rows),
        "target_sync_count":int(rows[-1]["global_step"])//500,"final_epsilon":float(rows[-1]["epsilon"]),
        "epsilon_floor":{"global_step":floor,"episode":floor_index+1,"episodes_with_any_floor":len(rows)-floor_index,"complete_post_floor_episodes":len(post),"post_floor_transitions":int(rows[-1]["global_step"])-floor,"performance":summarize(post)},
        "intervals":intervals,"final_500":final,
        "coverage":{"unique":len(sampled),"total":len(train_ids),"percent":100*len(sampled)/len(train_ids),"min":min(counts),"max":max(counts),"mean":fmean(counts),"eval_samples":sum(sampled[i] for i in eval_ids)},
        "shield_integrity":{"viability_violations":sum(int(row["viability_violation_count"]) for row in rows),"empty_combined_masks":sum(int(row["empty_combined_mask_count"]) for row in rows),"empty_progress_actions":sum(int(row["empty_progress_action_count"]) for row in rows),"total_interventions":sum(int(row["shield_interventions"]) for row in rows),"total_relaxations":sum(int(row["shield_relaxations"]) for row in rows)},
        "stability":{"all_losses_finite":bool(np.isfinite(all_losses).all()),"max_loss":max(all_losses),"recent_mean_loss":fmean(all_losses[-500:]),"online_parameters_finite":all(torch.isfinite(x).all().item() for x in online),"target_parameters_finite":all(torch.isfinite(x).all().item() for x in target),"max_abs_parameter":max(x.abs().max().item() for x in online+target),"q_values_finite":bool(torch.isfinite(q).all()),"q_min":q.min().item(),"q_max":q.max().item()},
    }
    Path("results/safe_dqn_training_summary_seed_2026_ep3000.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


if __name__ == "__main__": print(json.dumps(analyze(),indent=2))
