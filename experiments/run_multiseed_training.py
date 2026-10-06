"""Run one frozen-method multi-seed training job."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

from env.faults import FaultDataset
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.train_dqn import TrainingConfig
from experiments.train_safe_dqn_stepbudget import StepBudgetSafeDQNTrainer
from experiments.train_vanilla_dqn_stepbudget import StepBudgetVanillaDQNTrainer


def summarize_training(algorithm, seed, result, runtime):
    logs = result.logs if algorithm == "safe" else result["logs"]
    complete = [x for x in logs if x.completed]
    post = [x for x in complete if x.start_global_step >= 20_000]
    losses = [x.mean_loss for x in logs if x.mean_loss is not None]
    checkpoint = (result.checkpoint_path if algorithm == "safe"
                  else result["checkpoint_path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = list(state["online_network_state_dict"].values())
    model = torch.nn.Sequential(torch.nn.Linear(79,128), torch.nn.ReLU(),
        torch.nn.Linear(128,128), torch.nn.ReLU(), torch.nn.Linear(128,19))
    model.load_state_dict({k.removeprefix("network."): v
                           for k,v in state["online_network_state_dict"].items()})
    train = FaultDataset().stratified_split("train", 2026)
    enc = ObservationEncoder()
    states = np.stack([enc.encode(NetworkEnv(split="train", fault_scenario=s).reset(seed=seed)[0])
                       for s in train])
    with torch.no_grad():
        q = model(torch.as_tensor(states, dtype=torch.float32))
    sampled = result.sampled_scenario_ids if algorithm == "safe" else result["sampled_scenario_ids"]
    counts = {scenario.scenario_id: sampled.count(scenario.scenario_id) for scenario in train}
    row = {
        "seed": seed, "algorithm": algorithm, "global_steps": 35747,
        "optimizer_updates": (result.optimizer_updates if algorithm == "safe" else result["optimizer_updates"]),
        "completed_episodes": len(complete), "post_floor_transitions": 15747,
        "post_floor_episodes": len(post),
        "train_recovery": fmean(x.recovery_success for x in post),
        "train_return": fmean(x.episode_return for x in post),
        "train_damage": fmean(x.new_policy_damage_events for x in post),
        "train_timeout": fmean(x.truncated for x in post),
        "train_incorrect_declare": fmean(x.termination_reason == "incorrect_declare_done" for x in post),
        "viability_violations": sum(getattr(x,"viability_violation_count",0) for x in logs),
        "shield_interventions": sum(getattr(x,"shield_interventions",0) for x in logs),
        "shield_relaxations": sum(getattr(x,"shield_relaxations",0) for x in logs),
        "empty_masks": sum(getattr(x,"empty_combined_mask_count",0) for x in logs),
        "finite_loss": bool(np.isfinite(losses).all()), "max_loss": max(losses),
        "finite_parameters": all(torch.isfinite(x).all().item() for x in tensors),
        "finite_q": bool(torch.isfinite(q).all()),
        "max_abs_parameter": max(x.abs().max().item() for x in tensors),
        "training_runtime_seconds": runtime, "unique_train_scenarios": len(set(sampled)),
        "min_scenario_samples": min(counts.values()), "max_scenario_samples": max(counts.values()),
        "mean_scenario_samples": fmean(counts.values()), "eval_training_exposure": 0,
    }
    return row


def run(algorithm, seed):
    output = Path("results/multiseed"); output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(episodes=1, seed=seed, split_seed=2026,
                            output_dir=str(output))
    started = perf_counter()
    if algorithm == "safe":
        result = StepBudgetSafeDQNTrainer(config, 35747).train()
        generated = result.checkpoint_path
        desired = output / f"safe_seed_{seed}.pt"
        generated.replace(desired)
        result = result.__class__(result.logs, result.global_step, result.optimizer_updates,
            result.replay_size, result.target_sync_steps, result.sampled_scenario_ids,
            result.elapsed_seconds, result.csv_path, desired)
    else:
        result = StepBudgetVanillaDQNTrainer(config, 35747).train()
    runtime = perf_counter() - started
    row = summarize_training(algorithm, seed, result, runtime)
    (output / f"{algorithm}_seed_{seed}_training_summary.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"algorithm": algorithm, "seed": seed, "runtime": runtime,
                      "episodes": row["completed_episodes"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=("vanilla", "safe"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    run(args.algorithm, args.seed)
