"""Vanilla DQN training with an exact environment-transition budget."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
import random

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.train_dqn import TrainingConfig, epsilon_at_step, seed_everything


@dataclass(frozen=True)
class VanillaBudgetLog:
    episode: int
    seed: int
    scenario_id: str
    start_global_step: int
    global_step: int
    epsilon: float
    episode_return: float
    episode_steps: int
    completed: bool
    budget_cut: bool
    terminated: bool
    truncated: bool
    recovery_success: bool
    termination_reason: str
    mean_loss: float | None
    optimizer_updates: int
    actual_configuration_changes: int
    diagnoses: int
    invalid_actions: int
    new_policy_damage_events: int
    unique_newly_damaged_policies: int
    service_damage_actions: int
    post_recovery_damage_events: int
    final_damaged_policies: int


class StepBudgetVanillaDQNTrainer:
    def __init__(self, config: TrainingConfig, target_steps: int = 35_747,
                 *, train_scenarios: tuple[FaultScenario, ...] | None = None):
        if target_steps <= 0:
            raise ValueError("target_steps must be positive")
        self.config = config
        self.target_steps = target_steps
        seed_everything(config.seed)
        self.agent = DQNAgent(gamma=config.gamma, learning_rate=config.learning_rate,
                              device=config.device)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)
        self.encoder = ObservationEncoder()
        self.train_scenarios = train_scenarios or FaultDataset().stratified_split(
            "train", config.split_seed)
        self._scenario_rng = random.Random(config.seed)
        self.global_step = self.optimizer_updates = 0
        self.target_sync_steps: list[int] = []

    def train(self):
        started = perf_counter()
        logs, sampled = [], []
        while self.global_step < self.target_steps:
            scenario = self._scenario_rng.choice(self.train_scenarios)
            sampled.append(scenario.scenario_id)
            logs.append(self._episode(len(logs), scenario))
        csv_path = self._save_csv(logs) if self.config.save_csv else None
        checkpoint_path = self._save_checkpoint() if self.config.save_checkpoint else None
        return {"logs": tuple(logs), "sampled_scenario_ids": tuple(sampled),
                "global_step": self.global_step, "optimizer_updates": self.optimizer_updates,
                "replay_size": len(self.replay_buffer),
                "target_sync_steps": tuple(self.target_sync_steps),
                "elapsed_seconds": perf_counter() - started,
                "csv_path": csv_path, "checkpoint_path": checkpoint_path}

    def _episode(self, episode, scenario):
        start = self.global_step
        episode_seed = self.config.seed + episode
        env = NetworkEnv(split="train", fault_scenario=scenario,
                         max_steps=self.config.max_episode_steps)
        observation, _ = env.reset(seed=episode_seed)
        state = self.encoder.encode(observation)
        losses, episode_return = [], 0.0
        terminated = truncated = False
        info = {}
        epsilon = epsilon_at_step(self.config, self.global_step)
        for _ in range(self.config.max_episode_steps):
            epsilon = epsilon_at_step(self.config, self.global_step)
            action = self.agent.select_action(
                state, np.asarray(env.get_action_mask(), dtype=np.bool_), epsilon)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode(observation)
            next_mask = np.asarray(env.get_action_mask(), dtype=np.bool_)
            self.replay_buffer.push(state, action, reward, next_state,
                                    terminated, truncated, next_mask)
            episode_return += reward
            self.global_step += 1
            if len(self.replay_buffer) >= max(self.config.replay_warmup,
                                              self.config.batch_size):
                losses.append(self.agent.update(self.replay_buffer, self.config.batch_size))
                self.optimizer_updates += 1
            if self.global_step % self.config.target_sync_interval == 0:
                self.agent.update_target_network()
                self.target_sync_steps.append(self.global_step)
            state = next_state
            if terminated or truncated or self.global_step == self.target_steps:
                break
        budget_cut = self.global_step == self.target_steps and not (terminated or truncated)
        completed = bool(terminated or truncated)
        success = bool(terminated and info.get("recovery_success", False))
        reason = ("budget_cut" if budget_cut else "recovery_success" if success else
                  "incorrect_declare_done" if terminated else "step_limit" if truncated
                  else "trainer_step_limit")
        metrics = env.get_metrics()
        return VanillaBudgetLog(
            episode, episode_seed, scenario.scenario_id, start, self.global_step,
            epsilon, episode_return, int(metrics["steps"]), completed, budget_cut,
            terminated, truncated, success, reason, fmean(losses) if losses else None,
            len(losses), int(metrics["actual_configuration_changes"]),
            int(metrics["diagnostic_actions"]), int(metrics["invalid_actions"]),
            int(metrics["new_policy_damage_events"]),
            int(metrics["unique_newly_damaged_policies"]),
            int(metrics["service_damage_actions"]),
            int(metrics["post_recovery_damage_events"]),
            int(metrics["final_damaged_policies"]))

    def _path(self, name):
        path = Path(self.config.output_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_csv(self, logs):
        path = self._path(f"vanilla_seed_{self.config.seed}_training.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(logs[0])))
            writer.writeheader(); writer.writerows(asdict(log) for log in logs)
        return path

    def _save_checkpoint(self):
        path = self._path(f"vanilla_seed_{self.config.seed}.pt")
        torch.save({"online_network_state_dict": self.agent.online_network.state_dict(),
                    "target_network_state_dict": self.agent.target_network.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "training_config": asdict(self.config), "seed": self.config.seed,
                    "global_step": self.global_step, "target_steps": self.target_steps,
                    "fresh_initialization": True,
                    "termination_policy": "stop_after_exact_target_transition; partial episode is budget_cut"}, path)
        return path
