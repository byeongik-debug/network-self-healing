"""Deterministic hold-out evaluation for a trained vanilla DQN checkpoint."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

import numpy as np
import torch

from agents.dqn import DQNAgent
from env.actions import ACTIONS, ActionSpec
from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder
from experiments.evaluation import configuration_recovery, functional_recovery


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str = "results/dqn_vanilla_seed_2026_ep3000.pt"
    output_csv: str = "results/dqn_eval_seed_2026_ep3000.csv"
    summary_json: str = "results/dqn_eval_summary_seed_2026_ep3000.json"
    trajectories_json: str = "results/dqn_eval_trajectories_seed_2026_ep3000.json"
    split_seed: int = 2026
    reset_seed: int = 2026
    max_episode_steps: int = 30
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.epsilon != 0.0:
            raise ValueError("hold-out evaluation requires epsilon=0.0")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")


@dataclass(frozen=True)
class EvaluationRow:
    scenario_id: str
    fault_type: str
    severity: str
    fault_count: int
    episode_return: float
    recovery_success: bool
    configuration_recovery: bool
    steps: int
    termination_reason: str
    incorrect_declare: bool
    timeout: bool
    actual_configuration_changes: int
    diagnoses: int
    invalid_actions: int
    initial_damaged_policies: int
    new_policy_damage_events: int
    unique_newly_damaged_policies: int
    service_damage_actions: int
    continuously_healthy_policies: int
    final_damaged_policies: int
    post_recovery_damage_events: int
    action_sequence: str


def action_name(action: ActionSpec) -> str:
    if action.kind in ("diagnose", "rollback", "declare_done"):
        return action.kind
    if action.kind == "set_access_vlan":
        return f"set_access_vlan({action.switch},{action.port},vlan={action.vlan})"
    operation = "allow" if action.allowed else "remove"
    return f"set_trunk_vlan({action.switch},{action.port},vlan={action.vlan},{operation})"


class DQNEvaluator:
    def __init__(
        self,
        config: EvaluationConfig | None = None,
        *,
        scenarios: tuple[FaultScenario, ...] | None = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.dataset = FaultDataset()
        self.train_scenarios = self.dataset.stratified_split(
            "train", self.config.split_seed
        )
        self.scenarios = scenarios if scenarios is not None else self.dataset.stratified_split(
            "eval", self.config.split_seed
        )
        if not self.scenarios:
            raise ValueError("evaluation scenario set must not be empty")
        train_ids = {scenario.scenario_id for scenario in self.train_scenarios}
        evaluated_ids = {scenario.scenario_id for scenario in self.scenarios}
        if train_ids & evaluated_ids:
            raise ValueError("evaluation scenarios overlap the training split")

        self.checkpoint = torch.load(
            self.config.checkpoint_path, map_location="cpu", weights_only=False
        )
        training_config = self.checkpoint["training_config"]
        self.agent = DQNAgent(
            gamma=float(training_config["gamma"]),
            learning_rate=float(training_config["learning_rate"]),
            device="cpu",
        )
        self.agent.online_network.load_state_dict(
            self.checkpoint["online_network_state_dict"]
        )
        self.agent.online_network.eval()
        self.encoder = ObservationEncoder()

    def evaluate(self, *, save: bool = True) -> dict[str, Any]:
        before = {
            name: tensor.detach().clone()
            for name, tensor in self.agent.online_network.state_dict().items()
        }
        rows: list[EvaluationRow] = []
        trajectories: list[dict[str, Any]] = []
        with torch.no_grad():
            for scenario in self.scenarios:
                row, trajectory = self._run_scenario(scenario)
                rows.append(row)
                trajectories.append(trajectory)
        parameters_unchanged = all(
            torch.equal(before[name], tensor)
            for name, tensor in self.agent.online_network.state_dict().items()
        )
        summary = self._summary(rows, parameters_unchanged)
        if save:
            self._save(rows, summary, trajectories)
        return {"rows": rows, "summary": summary, "trajectories": trajectories}

    def _run_scenario(
        self, scenario: FaultScenario
    ) -> tuple[EvaluationRow, dict[str, Any]]:
        env = NetworkEnv(
            split="eval",
            fault_scenario=scenario,
            max_steps=self.config.max_episode_steps,
        )
        observation, _ = env.reset(seed=self.config.reset_seed)
        state = self.encoder.encode(observation)
        actions: list[str] = []
        episode_return = 0.0
        terminated = truncated = False
        final_info: dict[str, Any] = {}
        for _ in range(self.config.max_episode_steps):
            action_id = self.agent.select_action(
                state,
                np.asarray(env.get_action_mask(), dtype=np.bool_),
                self.config.epsilon,
            )
            actions.append(action_name(ACTIONS[action_id]))
            observation, reward, terminated, truncated, final_info = env.step(action_id)
            episode_return += reward
            state = self.encoder.encode(observation)
            if terminated or truncated:
                break

        functional = functional_recovery(env.simulator)
        exact_configuration = configuration_recovery(env.simulator.snapshot())
        metrics = env.get_metrics()
        incorrect = bool(terminated and not final_info.get("recovery_success", False))
        if truncated:
            reason = "step_limit"
        elif incorrect:
            reason = "incorrect_declare_done"
        elif terminated and functional:
            reason = "recovery_success"
        else:
            reason = "trainer_step_limit"
        row = EvaluationRow(
            scenario_id=scenario.scenario_id,
            fault_type=scenario.fault_type,
            severity=scenario.severity,
            fault_count=scenario.fault_count,
            episode_return=episode_return,
            recovery_success=functional,
            configuration_recovery=exact_configuration,
            steps=int(metrics["steps"]),
            termination_reason=reason,
            incorrect_declare=incorrect,
            timeout=bool(truncated),
            actual_configuration_changes=int(metrics["actual_configuration_changes"]),
            diagnoses=int(metrics["diagnostic_actions"]),
            invalid_actions=int(metrics["invalid_actions"]),
            initial_damaged_policies=int(metrics["initial_damaged_policies"]),
            new_policy_damage_events=int(metrics["new_policy_damage_events"]),
            unique_newly_damaged_policies=int(metrics["unique_newly_damaged_policies"]),
            service_damage_actions=int(metrics["service_damage_actions"]),
            continuously_healthy_policies=int(metrics["continuously_healthy_policies"]),
            final_damaged_policies=int(metrics["final_damaged_policies"]),
            post_recovery_damage_events=int(metrics["post_recovery_damage_events"]),
            action_sequence=" -> ".join(actions),
        )
        trajectory = {
            "scenario_id": scenario.scenario_id,
            "severity": scenario.severity,
            "faults": [asdict(mutation) for mutation in scenario.mutations],
            "actions": actions,
            "termination_reason": reason,
            "recovery_success": functional,
            "configuration_recovery": exact_configuration,
            "final_damaged_policies": row.final_damaged_policies,
            "episode_return": episode_return,
            "new_policy_damage_events": row.new_policy_damage_events,
            "unique_newly_damaged_policies": row.unique_newly_damaged_policies,
            "service_damage_actions": row.service_damage_actions,
        }
        return row, trajectory

    def _summary(
        self, rows: list[EvaluationRow], parameters_unchanged: bool
    ) -> dict[str, Any]:
        checkpoint_state = self.checkpoint["online_network_state_dict"]
        architecture = [
            int(checkpoint_state["network.0.weight"].shape[1]),
            int(checkpoint_state["network.0.weight"].shape[0]),
            int(checkpoint_state["network.2.weight"].shape[0]),
            int(checkpoint_state["network.4.weight"].shape[0]),
        ]
        return {
            "evaluation": {
                "epsilon": self.config.epsilon,
                "scenario_count": len(rows),
                "split_seed": self.config.split_seed,
                "train_scenario_count": len(self.train_scenarios),
                "train_eval_overlap": 0,
                "parameters_unchanged": parameters_unchanged,
            },
            "checkpoint": {
                "path": self.config.checkpoint_path,
                "seed": self.checkpoint["seed"],
                "global_step": self.checkpoint["global_step"],
                "architecture": architecture,
                "training_config": self.checkpoint["training_config"],
            },
            "overall": summarize_rows(rows),
            "by_severity": {
                severity: summarize_rows([row for row in rows if row.severity == severity])
                for severity in ("low", "medium", "high")
            },
            "failed_scenarios": [
                asdict(row) for row in rows if not row.recovery_success
            ],
            "highest_safety_damage_successes": [
                asdict(row)
                for row in sorted(
                    (row for row in rows if row.recovery_success),
                    key=lambda row: (
                        -row.new_policy_damage_events,
                        -row.unique_newly_damaged_policies,
                        row.scenario_id,
                    ),
                )[:5]
            ],
            "baseline_comparison": load_baseline_comparison(),
        }

    def _save(
        self,
        rows: list[EvaluationRow],
        summary: dict[str, Any],
        trajectories: list[dict[str, Any]],
    ) -> None:
        csv_path = Path(self.config.output_csv)
        summary_path = Path(self.config.summary_json)
        trajectory_path = Path(self.config.trajectories_json)
        for path in (csv_path, summary_path, trajectory_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(EvaluationRow.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        trajectory_path.write_text(
            json.dumps(trajectories, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def summarize_rows(rows: list[EvaluationRow]) -> dict[str, float | int]:
    if not rows:
        return {"scenario_count": 0}
    steps = np.asarray([row.steps for row in rows], dtype=np.float64)
    damage = np.asarray(
        [row.new_policy_damage_events for row in rows], dtype=np.float64
    )
    unique_damage = np.asarray(
        [row.unique_newly_damaged_policies for row in rows], dtype=np.float64
    )
    returns = np.asarray([row.episode_return for row in rows], dtype=np.float64)
    return {
        "scenario_count": len(rows),
        "functional_recovery_rate": fmean(row.recovery_success for row in rows),
        "configuration_recovery_rate": fmean(
            row.configuration_recovery for row in rows
        ),
        "incorrect_declare_rate": fmean(row.incorrect_declare for row in rows),
        "timeout_rate": fmean(row.timeout for row in rows),
        "mean_steps": float(steps.mean()),
        "median_steps": float(np.median(steps)),
        "p95_steps": float(np.percentile(steps, 95)),
        "mean_actual_configuration_changes": fmean(
            row.actual_configuration_changes for row in rows
        ),
        "mean_diagnoses": fmean(row.diagnoses for row in rows),
        "mean_new_policy_damage_events": float(damage.mean()),
        "median_new_policy_damage_events": float(np.median(damage)),
        "p95_new_policy_damage_events": float(np.percentile(damage, 95)),
        "max_new_policy_damage_events": float(damage.max()),
        "mean_unique_newly_damaged_policies": float(unique_damage.mean()),
        "median_unique_newly_damaged_policies": float(np.median(unique_damage)),
        "mean_service_damage_actions": fmean(
            row.service_damage_actions for row in rows
        ),
        "mean_continuously_healthy_policies": fmean(
            row.continuously_healthy_policies for row in rows
        ),
        "mean_final_damaged_policies": fmean(
            row.final_damaged_policies for row in rows
        ),
        "mean_post_recovery_damage_events": fmean(
            row.post_recovery_damage_events for row in rows
        ),
        "mean_return": float(returns.mean()),
        "median_return": float(median(returns)),
        "minimum_return": float(returns.min()),
        "maximum_return": float(returns.max()),
    }


def load_baseline_comparison(
    path: str | Path = "results/stratified_baseline_results.csv",
) -> dict[str, dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["row_type"] == "episode" and row["split"] == "eval"
        ]
    output: dict[str, dict[str, float]] = {}
    for algorithm in ("rule_based", "configuration_diff"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        output[algorithm] = {
            "scenario_count": float(len(selected)),
            "functional_recovery_rate": fmean(
                float(row["functional_recovery_rate"]) for row in selected
            ),
            "configuration_recovery_rate": fmean(
                float(row["configuration_recovery_rate"]) for row in selected
            ),
            "mean_steps": fmean(float(row["steps"]) for row in selected),
            "mean_actual_configuration_changes": fmean(
                float(row["actual_configuration_changes"]) for row in selected
            ),
            "mean_new_policy_damage_events": fmean(
                float(row["new_policy_damage_events"]) for row in selected
            ),
            "mean_unique_newly_damaged_policies": fmean(
                float(row["unique_newly_damaged_policies"]) for row in selected
            ),
        }
    return output


def evaluate_dqn(config: EvaluationConfig | None = None) -> dict[str, Any]:
    return DQNEvaluator(config).evaluate()


if __name__ == "__main__":
    result = evaluate_dqn()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
