"""Minimal reproducible training pipeline for the vanilla DQN agent."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from statistics import fmean
from typing import Callable

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1000
    batch_size: int = 64
    replay_capacity: int = 10_000
    replay_warmup: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_sync_interval: int = 500
    max_episode_steps: int = 30
    seed: int = 2026
    split_seed: int = 2026
    device: str = "cpu"
    output_dir: str = "results"
    save_csv: bool = True
    save_checkpoint: bool = True

    def __post_init__(self) -> None:
        positive = {
            "episodes": self.episodes,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "target_sync_interval": self.target_sync_interval,
            "max_episode_steps": self.max_episode_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.replay_warmup < 0:
            raise ValueError("replay_warmup must be non-negative")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon values must satisfy 0 <= epsilon_end <= epsilon_start <= 1"
            )


@dataclass(frozen=True)
class EpisodeLog:
    episode: int
    seed: int
    scenario_id: str
    global_step: int
    epsilon: float
    episode_return: float
    episode_steps: int
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


@dataclass(frozen=True)
class TrainingResult:
    logs: tuple[EpisodeLog, ...]
    global_step: int
    optimizer_updates: int
    replay_size: int
    target_sync_steps: tuple[int, ...]
    sampled_scenario_ids: tuple[str, ...]
    csv_path: Path | None
    checkpoint_path: Path | None


def epsilon_at_step(config: TrainingConfig, global_step: int) -> float:
    """Linear epsilon schedule indexed by environment transitions."""
    if global_step >= config.epsilon_decay_steps:
        return config.epsilon_end
    progress = max(global_step, 0) / config.epsilon_decay_steps
    return config.epsilon_start - (
        config.epsilon_start - config.epsilon_end
    ) * progress


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the current CPU-first training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DQNTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        agent: DQNAgent | None = None,
        replay_buffer: ReplayBuffer | None = None,
        encoder: ObservationEncoder | None = None,
        train_scenarios: tuple[FaultScenario, ...] | None = None,
        env_factory: Callable[[FaultScenario], NetworkEnv] | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.seed)
        self.agent = agent if agent is not None else DQNAgent(
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            device=config.device,
        )
        self.replay_buffer = (
            replay_buffer
            if replay_buffer is not None
            else ReplayBuffer(config.replay_capacity)
        )
        self.encoder = encoder if encoder is not None else ObservationEncoder()
        self.train_scenarios = (
            train_scenarios
            if train_scenarios is not None
            else FaultDataset().stratified_split("train", seed=config.split_seed)
        )
        if not self.train_scenarios:
            raise ValueError("training scenario set must not be empty")
        self._env_factory = env_factory or (
            lambda scenario: NetworkEnv(
                split="train",
                fault_scenario=scenario,
                max_steps=config.max_episode_steps,
            )
        )
        self._scenario_rng = random.Random(config.seed)
        self.global_step = 0
        self.optimizer_updates = 0
        self.target_sync_steps: list[int] = []

    def train(self) -> TrainingResult:
        logs: list[EpisodeLog] = []
        sampled_ids: list[str] = []
        for episode in range(self.config.episodes):
            scenario = self._scenario_rng.choice(self.train_scenarios)
            sampled_ids.append(scenario.scenario_id)
            logs.append(self._run_episode(episode, scenario))

        csv_path = self._save_csv(logs) if self.config.save_csv else None
        checkpoint_path = (
            self._save_checkpoint() if self.config.save_checkpoint else None
        )
        return TrainingResult(
            logs=tuple(logs),
            global_step=self.global_step,
            optimizer_updates=self.optimizer_updates,
            replay_size=len(self.replay_buffer),
            target_sync_steps=tuple(self.target_sync_steps),
            sampled_scenario_ids=tuple(sampled_ids),
            csv_path=csv_path,
            checkpoint_path=checkpoint_path,
        )

    def _run_episode(self, episode: int, scenario: FaultScenario) -> EpisodeLog:
        episode_seed = self.config.seed + episode
        env = self._env_factory(scenario)
        observation, _ = env.reset(seed=episode_seed)
        state = self.encoder.encode(observation)
        losses: list[float] = []
        episode_return = 0.0
        terminated = truncated = recovery_success = False
        final_info: dict[str, object] = {}
        last_epsilon = epsilon_at_step(self.config, self.global_step)

        for _ in range(self.config.max_episode_steps):
            last_epsilon = epsilon_at_step(self.config, self.global_step)
            action = self.agent.select_action(
                state, np.asarray(env.get_action_mask(), dtype=np.bool_), last_epsilon
            )
            next_observation, reward, terminated, truncated, final_info = env.step(
                int(action)
            )
            next_state = self.encoder.encode(next_observation)
            next_action_mask = np.asarray(env.get_action_mask(), dtype=np.bool_)
            self.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                terminated,
                truncated,
                next_action_mask,
            )

            episode_return += reward
            self.global_step += 1
            if len(self.replay_buffer) >= max(
                self.config.replay_warmup, self.config.batch_size
            ):
                losses.append(
                    self.agent.update(self.replay_buffer, self.config.batch_size)
                )
                self.optimizer_updates += 1
            if self.global_step % self.config.target_sync_interval == 0:
                self.agent.update_target_network()
                self.target_sync_steps.append(self.global_step)

            state = next_state
            if terminated or truncated:
                break

        if terminated:
            recovery_success = bool(final_info.get("recovery_success", False))
            reason = "recovery_success" if recovery_success else "incorrect_declare_done"
        elif truncated:
            reason = "step_limit"
        else:
            # Defensive fallback for a custom environment whose limit differs
            # from TrainingConfig.max_episode_steps.
            reason = "trainer_step_limit"
        metrics = env.get_metrics()
        return EpisodeLog(
            episode=episode,
            seed=episode_seed,
            scenario_id=scenario.scenario_id,
            global_step=self.global_step,
            epsilon=last_epsilon,
            episode_return=episode_return,
            episode_steps=int(metrics["steps"]),
            terminated=terminated,
            truncated=truncated,
            recovery_success=recovery_success,
            termination_reason=reason,
            mean_loss=fmean(losses) if losses else None,
            optimizer_updates=len(losses),
            actual_configuration_changes=int(metrics["actual_configuration_changes"]),
            diagnoses=int(metrics["diagnostic_actions"]),
            invalid_actions=int(metrics["invalid_actions"]),
            new_policy_damage_events=int(metrics["new_policy_damage_events"]),
            unique_newly_damaged_policies=int(metrics["unique_newly_damaged_policies"]),
            service_damage_actions=int(metrics["service_damage_actions"]),
            post_recovery_damage_events=int(metrics["post_recovery_damage_events"]),
            final_damaged_policies=int(metrics["final_damaged_policies"]),
        )

    def _output_path(self, filename: str) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    def _save_csv(self, logs: list[EpisodeLog]) -> Path:
        path = self._output_path(f"dqn_training_seed_{self.config.seed}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(logs[0])))
            writer.writeheader()
            writer.writerows(asdict(log) for log in logs)
        return path

    def _save_checkpoint(self) -> Path:
        path = self._output_path(f"dqn_vanilla_seed_{self.config.seed}.pt")
        torch.save(
            {
                "online_network_state_dict": self.agent.online_network.state_dict(),
                "target_network_state_dict": self.agent.target_network.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "training_config": asdict(self.config),
                "seed": self.config.seed,
                "global_step": self.global_step,
            },
            path,
        )
        return path


def train_dqn(config: TrainingConfig | None = None) -> TrainingResult:
    """Train vanilla DQN with defaults or a caller-provided smoke config."""
    return DQNTrainer(config or TrainingConfig()).train()
