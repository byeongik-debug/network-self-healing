"""Reproducible mixed-topology Graph DQN training for expanded experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch

from agents.graph_dqn import GraphDQNAgent
from agents.graph_replay_buffer import GraphReplayBuffer
from baselines.expanded import reference_repairs
from env.expanded.actions import build_action_catalog
from env.expanded.dataset import ExpandedScenario, build_scenarios
from env.expanded.environment import ExpandedAction, ExpandedNetworkEnv
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.shield import SafetyMaskCache, policy_violation_count
from env.expanded.topologies import build_topology


@dataclass(frozen=True)
class GraphTrainingConfig:
    episodes: int = 300
    batch_size: int = 16
    replay_capacity: int = 5000
    replay_warmup: int = 64
    max_steps: int = 30
    hidden: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4000
    guidance_start: float = 0.8
    guidance_end: float = 0.1
    target_sync_interval: int = 200
    seed: int = 2026
    output_dir: str = "results/expanded/graph_dqn"
    save_outputs: bool = True
    policy_features: bool = False


@dataclass(frozen=True)
class GraphEpisodeLog:
    episode: int
    scenario_id: str
    topology: str
    steps: int
    success: bool
    episode_return: float
    epsilon: float
    guidance_probability: float
    guided_actions: int
    optimizer_updates: int
    mean_loss: float | None
    new_policy_damage_events: int


def _schedule(start: float, end: float, step: int, duration: int) -> float:
    if step >= duration:
        return end
    progress = min(max(step / max(duration, 1), 0.0), 1.0)
    return start + (end - start) * progress


class GraphDQNTrainer:
    def __init__(self, config: GraphTrainingConfig) -> None:
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = random.Random(config.seed)
        action_dim = 13 if config.policy_features else 11
        self.agent = GraphDQNAgent(
            8, 5, action_dim, hidden=config.hidden, gamma=config.gamma,
            learning_rate=config.learning_rate,
        )
        self.replay = GraphReplayBuffer(config.replay_capacity)
        self.scenarios = tuple(
            scenario for scenario in build_scenarios()
            if scenario.split == "train" and not scenario.initially_healthy
        )
        self.global_step = 0

    def train(self) -> tuple[GraphEpisodeLog, ...]:
        logs = tuple(self._episode(index, self.rng.choice(self.scenarios))
                     for index in range(self.config.episodes))
        if self.config.save_outputs:
            self._save(logs)
        return logs

    def _episode(self, episode: int, scenario: ExpandedScenario) -> GraphEpisodeLog:
        topology = build_topology(scenario.topology)
        catalog = build_action_catalog(topology)
        action_ids = {candidate.action: candidate.action_id for candidate in catalog}
        encoder = GraphObservationEncoder(
            topology, catalog, include_policy_features=self.config.policy_features
        )
        env = ExpandedNetworkEnv(topology, max_steps=self.config.max_steps)
        shield = SafetyMaskCache(catalog)
        observation, _ = env.reset(scenario.faults, seed=self.config.seed + episode)
        state = encoder.encode(observation)
        losses: list[float] = []
        total_reward, guided = 0.0, 0
        success = False
        epsilon = guidance_probability = 0.0
        for _ in range(self.config.max_steps):
            epsilon = _schedule(
                self.config.epsilon_start, self.config.epsilon_end,
                self.global_step, self.config.epsilon_decay_steps,
            )
            guidance_probability = _schedule(
                self.config.guidance_start, self.config.guidance_end,
                self.global_step, self.config.epsilon_decay_steps,
            )
            mask = shield.mask(env)
            repairs = [action for action in reference_repairs(env)
                       if mask[action_ids[action]]]
            if repairs and self.rng.random() < guidance_probability:
                action_id = action_ids[self.rng.choice(repairs)]
                guided += 1
            else:
                action_id = self.agent.select_action(state, mask, epsilon)
            before = policy_violation_count(env.simulator)
            next_observation, base_reward, terminated, truncated, info = env.step(
                catalog[action_id].action
            )
            after = policy_violation_count(env.simulator)
            reward = base_reward + float(before - after) - 2.0 * float(
                info.get("policy_healthy", False) is False and after > before
            )
            next_state = encoder.encode(next_observation)
            next_mask = shield.mask(env)
            self.replay.push(
                state, action_id, reward, next_state, terminated, truncated, next_mask
            )
            total_reward += reward
            self.global_step += 1
            if len(self.replay) >= max(self.config.replay_warmup, self.config.batch_size):
                losses.append(self.agent.update(self.replay, self.config.batch_size))
            if self.global_step % self.config.target_sync_interval == 0:
                self.agent.update_target_network()
            state = next_state
            if terminated or truncated:
                success = bool(info.get("recovery_success", False))
                break
        metrics = env.metrics()
        return GraphEpisodeLog(
            episode, scenario.scenario_id, scenario.topology, int(metrics["steps"]),
            success, total_reward, epsilon, guidance_probability, guided, len(losses),
            float(np.mean(losses)) if losses else None,
            int(metrics["new_policy_damage_events"]),
        )

    def _save(self, logs: tuple[GraphEpisodeLog, ...]) -> None:
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        suffix = "_policy" if self.config.policy_features else ""
        if self.config.guidance_start == 0.0 and self.config.guidance_end == 0.0:
            suffix += "_noguidance"
        run_name = f"seed_{self.config.seed}_ep{self.config.episodes}{suffix}"
        with (output / f"training_{run_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(logs[0])))
            writer.writeheader()
            writer.writerows(asdict(log) for log in logs)
        torch.save({
            "model": self.agent.online_network.state_dict(),
            "config": asdict(self.config),
            "global_step": self.global_step,
        }, output / f"model_{run_name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--policy-features", action="store_true")
    parser.add_argument("--output-dir", default="results/expanded/graph_dqn")
    parser.add_argument(
        "--no-guidance", action="store_true",
        help="Disable reference-repair exploration while keeping all other budgets fixed",
    )
    args = parser.parse_args()
    config = GraphTrainingConfig(
        episodes=args.episodes, seed=args.seed, policy_features=args.policy_features,
        guidance_start=0.0 if args.no_guidance else 0.8,
        guidance_end=0.0 if args.no_guidance else 0.1,
        output_dir=args.output_dir,
    )
    logs = GraphDQNTrainer(config).train()
    print({
        "episodes": len(logs),
        "success_rate": sum(log.success for log in logs) / len(logs),
        "last_50_success_rate": sum(log.success for log in logs[-50:]) / min(50, len(logs)),
        "global_step": sum(log.steps for log in logs),
        "damage": sum(log.new_policy_damage_events for log in logs),
    })


if __name__ == "__main__":
    main()
