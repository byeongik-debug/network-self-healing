"""Fresh Safe DQN training controlled by an exact environment-step budget."""

from __future__ import annotations

import csv
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


TARGET_STEPS = 35_747


@dataclass(frozen=True)
class StepBudgetEpisodeLog:
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
class StepBudgetTrainingResult:
    logs: tuple[StepBudgetEpisodeLog, ...]
    global_step: int
    optimizer_updates: int
    replay_size: int
    target_sync_steps: tuple[int, ...]
    sampled_scenario_ids: tuple[str, ...]
    elapsed_seconds: float
    csv_path: Path | None
    checkpoint_path: Path | None

    @property
    def completed_episodes(self) -> int:
        return sum(log.completed for log in self.logs)


class StepBudgetSafeDQNTrainer:
    """Safe DQN trainer that stops immediately after ``target_steps`` transitions."""

    def __init__(
        self,
        config: TrainingConfig,
        target_steps: int = TARGET_STEPS,
        *,
        agent: DQNAgent | None = None,
        replay_buffer: ReplayBuffer | None = None,
        encoder: ObservationEncoder | None = None,
        shield: MinimumDamageViabilityShield | None = None,
        train_scenarios: tuple[FaultScenario, ...] | None = None,
    ) -> None:
        if target_steps <= 0:
            raise ValueError("target_steps must be positive")
        self.config = config
        self.target_steps = target_steps
        seed_everything(config.seed)
        self.agent = agent or DQNAgent(
            gamma=config.gamma, learning_rate=config.learning_rate, device=config.device
        )
        self.replay_buffer = replay_buffer or ReplayBuffer(config.replay_capacity)
        self.encoder = encoder or ObservationEncoder()
        self.shield = shield or MinimumDamageViabilityShield()
        self.train_scenarios = train_scenarios or FaultDataset().stratified_split(
            "train", config.split_seed
        )
        if not self.train_scenarios:
            raise ValueError("training scenario set must not be empty")
        self._scenario_rng = __import__("random").Random(config.seed)
        self.global_step = 0
        self.optimizer_updates = 0
        self.target_sync_steps: list[int] = []

    def train(self) -> StepBudgetTrainingResult:
        started = perf_counter()
        logs: list[StepBudgetEpisodeLog] = []
        sampled: list[str] = []
        episode = 0
        while self.global_step < self.target_steps:
            scenario = self._scenario_rng.choice(self.train_scenarios)
            sampled.append(scenario.scenario_id)
            logs.append(self._run_episode(episode, scenario))
            episode += 1
        if self.global_step != self.target_steps:
            raise AssertionError("exact transition budget was not respected")
        elapsed = perf_counter() - started
        csv_path = self._save_csv(logs) if self.config.save_csv else None
        checkpoint_path = self._save_checkpoint() if self.config.save_checkpoint else None
        return StepBudgetTrainingResult(
            tuple(logs), self.global_step, self.optimizer_updates,
            len(self.replay_buffer), tuple(self.target_sync_steps), tuple(sampled),
            elapsed, csv_path, checkpoint_path,
        )

    def _run_episode(self, episode: int, scenario: FaultScenario) -> StepBudgetEpisodeLog:
        start_step = self.global_step
        episode_seed = self.config.seed + episode
        env = NetworkEnv(split="train", fault_scenario=scenario,
                         max_steps=self.config.max_episode_steps)
        observation, _ = env.reset(seed=episode_seed)
        state = self.encoder.encode(observation)
        minimum_required = self.shield.cost_to_goal(env)
        losses: list[float] = []
        admissible_counts: list[int] = []
        episode_return = exploration_mass = 0.0
        interventions = relaxations = empty_progress = empty_combined = violations = 0
        terminated = truncated = False
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
            progress = [a.id for a in ACTIONS if a.kind in (
                "set_access_vlan", "set_trunk_vlan", "rollback") and combined[a.id]]
            if not env.simulator.is_policy_healthy() and not progress:
                empty_progress += 1

            state_tensor = torch.as_tensor(state, dtype=torch.float32,
                                           device=self.agent.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.agent.online_network(state_tensor)[0]
                valid_tensor = torch.as_tensor(valid, device=self.agent.device)
                top_valid = int(q_values.masked_fill(~valid_tensor, -torch.inf).argmax())
            interventions += int(not bool(combined[top_valid]))
            unsafe_valid = int(np.count_nonzero(valid & ~combined))
            exploration_mass += last_epsilon * unsafe_valid / int(np.count_nonzero(valid))

            action = self.agent.select_action(state, combined, last_epsilon)
            candidate = decision.candidates[action]
            relaxations += int(bool(candidate.immediate_damage and candidate.immediate_damage > 0))
            if ACTIONS[action].kind in ("set_access_vlan", "set_trunk_vlan", "rollback"):
                if candidate.immediate_damage is None or candidate.successor_cost is None \
                        or candidate.immediate_damage + candidate.successor_cost != decision.current_cost:
                    violations += 1

            next_observation, reward, terminated, truncated, final_info = env.step(action)
            next_state = self.encoder.encode(next_observation)
            next_mask = (np.zeros(len(ACTIONS), dtype=np.bool_) if terminated else
                         self.shield.get_combined_mask(
                             env, np.asarray(env.get_action_mask(), dtype=np.bool_)))
            # A budget cut is an external collection boundary, not an MDP terminal.
            # Store the environment's own flags and keep its viable next mask.
            self.replay_buffer.push(
                state, action, reward, next_state, terminated, truncated, next_mask
            )
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
        recovery_success = bool(terminated and final_info.get("recovery_success", False))
        reason = ("budget_cut" if budget_cut else "recovery_success" if recovery_success
                  else "incorrect_declare_done" if terminated else "step_limit" if truncated
                  else "trainer_step_limit")
        metrics = env.get_metrics()
        damage = int(metrics["new_policy_damage_events"])
        return StepBudgetEpisodeLog(
            episode, episode_seed, scenario.scenario_id, start_step, self.global_step,
            last_epsilon, episode_return, int(metrics["steps"]), completed, budget_cut,
            terminated, truncated, recovery_success, reason,
            fmean(losses) if losses else None, len(losses),
            int(metrics["actual_configuration_changes"]), int(metrics["diagnostic_actions"]),
            int(metrics["invalid_actions"]), damage,
            int(metrics["unique_newly_damaged_policies"]),
            int(metrics["service_damage_actions"]), int(metrics["post_recovery_damage_events"]),
            int(metrics["final_damaged_policies"]), interventions, interventions,
            exploration_mass, relaxations, fmean(admissible_counts), empty_progress,
            empty_combined, minimum_required, damage - minimum_required, violations,
        )

    def _output(self, filename: str) -> Path:
        path = Path(self.config.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_csv(self, logs: list[StepBudgetEpisodeLog]) -> Path:
        path = self._output(f"safe_dqn_stepbudget_{self.target_steps}_seed_{self.config.seed}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(logs[0])))
            writer.writeheader()
            writer.writerows(asdict(log) for log in logs)
        return path

    def _save_checkpoint(self) -> Path:
        path = self._output(f"safe_dqn_stepbudget_{self.target_steps}_seed_{self.config.seed}.pt")
        torch.save({
            "online_network_state_dict": self.agent.online_network.state_dict(),
            "target_network_state_dict": self.agent.target_network.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "training_config": asdict(self.config), "seed": self.config.seed,
            "global_step": self.global_step, "target_steps": self.target_steps,
            "fresh_initialization": True,
            "termination_policy": "stop_after_exact_target_transition; partial episode is budget_cut",
            "safety_mechanism": "minimum_damage_viability_shield",
            "shield_state_count": len(self.shield.states),
        }, path)
        return path


def train_stepbudget_safe_dqn(
    config: TrainingConfig | None = None, target_steps: int = TARGET_STEPS,
) -> StepBudgetTrainingResult:
    config = config or TrainingConfig(episodes=1)
    return StepBudgetSafeDQNTrainer(config, target_steps).train()


if __name__ == "__main__":
    result = train_stepbudget_safe_dqn(TrainingConfig(episodes=1))
    print({"episodes_started": len(result.logs),
           "episodes_completed": result.completed_episodes,
           "global_step": result.global_step, "updates": result.optimizer_updates,
           "elapsed_seconds": result.elapsed_seconds})
