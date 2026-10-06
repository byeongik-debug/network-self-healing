"""Shield-aware training pipeline using the minimum-damage viability mask."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.actions import ACTIONS
from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.train_dqn import TrainingConfig, epsilon_at_step, seed_everything
from safety.viability_shield import MinimumDamageViabilityShield


@dataclass(frozen=True)
class SafeEpisodeLog:
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
    shield_interventions: int
    unsafe_top_q_attempts: int
    prevented_unsafe_exploration_mass: float
    shield_relaxations: int
    mean_admissible_actions: float
    empty_progress_action_count: int
    empty_combined_mask_count: int
    minimum_required_damage: int
    realized_damage_minus_minimum: int
    viability_violation_count: int


@dataclass(frozen=True)
class SafeTrainingResult:
    logs: tuple[SafeEpisodeLog, ...]
    global_step: int
    optimizer_updates: int
    replay_size: int
    target_sync_steps: tuple[int, ...]
    sampled_scenario_ids: tuple[str, ...]
    elapsed_seconds: float
    csv_path: Path | None
    checkpoint_path: Path | None


class SafeDQNTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        agent: DQNAgent | None = None,
        replay_buffer: ReplayBuffer | None = None,
        encoder: ObservationEncoder | None = None,
        shield: MinimumDamageViabilityShield | None = None,
        train_scenarios: tuple[FaultScenario, ...] | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.seed)
        self.agent = agent if agent is not None else DQNAgent(
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            device=config.device,
        )
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(
            config.replay_capacity
        )
        self.encoder = encoder if encoder is not None else ObservationEncoder()
        self.shield = shield if shield is not None else MinimumDamageViabilityShield()
        self.train_scenarios = (
            train_scenarios
            if train_scenarios is not None
            else FaultDataset().stratified_split("train", config.split_seed)
        )
        if not self.train_scenarios:
            raise ValueError("training scenario set must not be empty")
        self._scenario_rng = __import__("random").Random(config.seed)
        self.global_step = 0
        self.optimizer_updates = 0
        self.target_sync_steps: list[int] = []

    def train(self) -> SafeTrainingResult:
        started = perf_counter()
        logs: list[SafeEpisodeLog] = []
        sampled: list[str] = []
        for episode in range(self.config.episodes):
            scenario = self._scenario_rng.choice(self.train_scenarios)
            sampled.append(scenario.scenario_id)
            logs.append(self._run_episode(episode, scenario))
        elapsed = perf_counter() - started
        csv_path = self._save_csv(logs) if self.config.save_csv else None
        checkpoint_path = self._save_checkpoint() if self.config.save_checkpoint else None
        return SafeTrainingResult(
            tuple(logs), self.global_step, self.optimizer_updates,
            len(self.replay_buffer), tuple(self.target_sync_steps), tuple(sampled),
            elapsed, csv_path, checkpoint_path,
        )

    def _run_episode(self, episode: int, scenario: FaultScenario) -> SafeEpisodeLog:
        episode_seed = self.config.seed + episode
        env = NetworkEnv(
            split="train", fault_scenario=scenario,
            max_steps=self.config.max_episode_steps,
        )
        observation, _ = env.reset(seed=episode_seed)
        state = self.encoder.encode(observation)
        minimum_required = self.shield.cost_to_goal(env)
        losses: list[float] = []
        admissible_counts: list[int] = []
        episode_return = exploration_mass = 0.0
        interventions = relaxations = empty_progress = empty_combined = violations = 0
        terminated = truncated = recovery_success = False
        final_info: dict[str, object] = {}
        last_epsilon = epsilon_at_step(self.config, self.global_step)

        for _ in range(self.config.max_episode_steps):
            last_epsilon = epsilon_at_step(self.config, self.global_step)
            valid = np.asarray(env.get_action_mask(), dtype=np.bool_)
            decision = self.shield.get_decision(env, valid)
            combined = decision.combined_mask
            admissible_counts.append(decision.admissible_action_count)
            if not np.any(combined):
                empty_combined += 1
                raise RuntimeError("empty combined action mask during Safe DQN training")
            progress = [
                action.id for action in ACTIONS
                if action.kind in ("set_access_vlan", "set_trunk_vlan", "rollback")
                and combined[action.id]
            ]
            if not env.simulator.is_policy_healthy() and not progress:
                empty_progress += 1

            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.agent.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.agent.online_network(state_tensor)[0]
                valid_tensor = torch.as_tensor(valid, device=self.agent.device)
                top_valid = int(
                    q_values.masked_fill(~valid_tensor, -torch.inf).argmax().item()
                )
            blocked_top = not bool(combined[top_valid])
            interventions += int(blocked_top)
            unsafe_valid = int(np.count_nonzero(valid & ~combined))
            exploration_mass += last_epsilon * unsafe_valid / int(np.count_nonzero(valid))

            action = self.agent.select_action(state, combined, last_epsilon)
            candidate = decision.candidates[action]
            relaxations += int(bool(candidate.immediate_damage and candidate.immediate_damage > 0))
            if ACTIONS[action].kind in ("set_access_vlan", "set_trunk_vlan", "rollback"):
                if candidate.immediate_damage is None or candidate.successor_cost is None:
                    violations += 1
                elif candidate.immediate_damage + candidate.successor_cost != decision.current_cost:
                    violations += 1

            next_observation, reward, terminated, truncated, final_info = env.step(action)
            next_state = self.encoder.encode(next_observation)
            if terminated:
                next_mask = np.zeros(len(ACTIONS), dtype=np.bool_)
            else:
                next_mask = self.shield.get_combined_mask(
                    env, np.asarray(env.get_action_mask(), dtype=np.bool_)
                )
            self.replay_buffer.push(
                state, action, reward, next_state, terminated, truncated, next_mask
            )
            episode_return += reward
            self.global_step += 1
            if len(self.replay_buffer) >= max(
                self.config.replay_warmup, self.config.batch_size
            ):
                losses.append(self.agent.update(self.replay_buffer, self.config.batch_size))
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
            reason = "trainer_step_limit"
        metrics = env.get_metrics()
        realized_damage = int(metrics["new_policy_damage_events"])
        return SafeEpisodeLog(
            episode, episode_seed, scenario.scenario_id, self.global_step,
            last_epsilon, episode_return, int(metrics["steps"]), terminated,
            truncated, recovery_success, reason,
            fmean(losses) if losses else None, len(losses),
            int(metrics["actual_configuration_changes"]),
            int(metrics["diagnostic_actions"]), int(metrics["invalid_actions"]),
            realized_damage, int(metrics["unique_newly_damaged_policies"]),
            int(metrics["service_damage_actions"]),
            int(metrics["post_recovery_damage_events"]),
            int(metrics["final_damaged_policies"]), interventions, interventions,
            exploration_mass, relaxations, fmean(admissible_counts), empty_progress,
            empty_combined, minimum_required,
            realized_damage - minimum_required, violations,
        )

    def _output(self, filename: str) -> Path:
        path = Path(self.config.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_csv(self, logs: list[SafeEpisodeLog]) -> Path:
        path = self._output(f"safe_dqn_training_seed_{self.config.seed}_ep{self.config.episodes}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(logs[0])))
            writer.writeheader()
            writer.writerows(asdict(log) for log in logs)
        return path

    def _save_checkpoint(self) -> Path:
        path = self._output(f"safe_dqn_seed_{self.config.seed}_ep{self.config.episodes}.pt")
        torch.save(
            {
                "online_network_state_dict": self.agent.online_network.state_dict(),
                "target_network_state_dict": self.agent.target_network.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "training_config": asdict(self.config),
                "seed": self.config.seed,
                "global_step": self.global_step,
                "safety_mechanism": "minimum_damage_viability_shield",
                "shield_state_count": len(self.shield.states),
            }, path,
        )
        return path


def train_safe_dqn(config: TrainingConfig | None = None) -> SafeTrainingResult:
    return SafeDQNTrainer(config or TrainingConfig()).train()


def save_safe_learning_curve(
    logs: tuple[SafeEpisodeLog, ...], path: str | Path,
) -> Path:
    fields = (
        "episode_return", "recovery_success", "episode_steps",
        "new_policy_damage_events", "unique_newly_damaged_policies",
        "service_damage_actions", "mean_loss", "shield_interventions",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = ["episode", "global_step", "epsilon", "window_start", "window_size"]
    names += [f"{field}_rolling_100" for field in fields]
    names += ["incorrect_declare_rolling_100", "timeout_rolling_100"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for index, log in enumerate(logs):
            window = logs[max(0, index - 99):index + 1]
            row: dict[str, object] = {
                "episode": index + 1, "global_step": log.global_step,
                "epsilon": log.epsilon, "window_start": max(1, index - 98),
                "window_size": len(window),
                "incorrect_declare_rolling_100": fmean(
                    item.termination_reason == "incorrect_declare_done" for item in window
                ),
                "timeout_rolling_100": fmean(item.truncated for item in window),
            }
            for field in fields:
                values = [getattr(item, field) for item in window]
                present = [value for value in values if value is not None]
                row[f"{field}_rolling_100"] = fmean(present) if present else None
            writer.writerow(row)
    return destination


def save_safe_training_summary(summary: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
