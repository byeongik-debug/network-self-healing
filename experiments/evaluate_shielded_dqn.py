"""Shield-only evaluation of the fixed Vanilla DQN checkpoint."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch

from env.actions import ACTIONS
from env.network_env import NetworkEnv
from experiments.evaluate_dqn import (
    DQNEvaluator,
    EvaluationConfig,
    EvaluationRow,
    action_name,
    summarize_rows,
)
from experiments.evaluation import configuration_recovery, functional_recovery
from safety.viability_shield import MinimumDamageViabilityShield


@dataclass(frozen=True)
class ShieldedEvaluationRow(EvaluationRow):
    theoretical_minimum_damage: int
    shield_interventions: int
    unsafe_top_q_actions: int
    mean_admissible_actions: float
    shield_relaxations: int
    empty_progress_actions: int
    empty_combined_masks: int


class ShieldedDQNEvaluator(DQNEvaluator):
    def __init__(
        self,
        config: EvaluationConfig | None = None,
        *,
        scenarios=None,
        output_csv: str = "results/dqn_shield_only_eval_seed_2026_ep3000.csv",
        output_summary: str = "results/dqn_shield_only_eval_summary_seed_2026_ep3000.json",
        output_trajectories: str = "results/dqn_shield_only_eval_trajectories_seed_2026_ep3000.json",
    ) -> None:
        super().__init__(config, scenarios=scenarios)
        self.shield = MinimumDamageViabilityShield()
        self.output_csv = Path(output_csv)
        self.output_summary = Path(output_summary)
        self.output_trajectories = Path(output_trajectories)

    def evaluate(self, *, save: bool = True) -> dict[str, Any]:
        before = {
            name: tensor.detach().clone()
            for name, tensor in self.agent.online_network.state_dict().items()
        }
        rows: list[ShieldedEvaluationRow] = []
        trajectories: list[dict[str, Any]] = []
        with torch.no_grad():
            for scenario in self.scenarios:
                row, trajectory = self._run_shielded_scenario(scenario)
                rows.append(row)
                trajectories.append(trajectory)
        unchanged = all(
            torch.equal(before[name], tensor)
            for name, tensor in self.agent.online_network.state_dict().items()
        )
        summary = self._shielded_summary(rows, unchanged)
        if save:
            self._save_shielded(rows, summary, trajectories)
        return {"rows": rows, "summary": summary, "trajectories": trajectories}

    def _run_shielded_scenario(self, scenario):
        env = NetworkEnv(
            split="eval",
            fault_scenario=scenario,
            max_steps=self.config.max_episode_steps,
        )
        observation, _ = env.reset(seed=self.config.reset_seed)
        state = self.encoder.encode(observation)
        theoretical = self.shield.cost_to_goal(env)
        actions: list[str] = []
        decisions: list[dict[str, Any]] = []
        episode_return = 0.0
        interventions = unsafe_top_q = relaxations = 0
        empty_progress = empty_combined = 0
        admissible_counts: list[int] = []
        terminated = truncated = False
        final_info: dict[str, Any] = {}

        for _ in range(self.config.max_episode_steps):
            valid = np.asarray(env.get_action_mask(), dtype=np.bool_)
            decision = self.shield.get_decision(env, valid)
            admissible_counts.append(decision.admissible_action_count)
            if not np.any(decision.combined_mask):
                empty_combined += 1
                break
            progress_ids = [
                action.id
                for action in ACTIONS
                if action.kind in ("set_access_vlan", "set_trunk_vlan", "rollback")
                and decision.combined_mask[action.id]
            ]
            if not env.simulator.is_policy_healthy() and not progress_ids:
                empty_progress += 1

            proposed = self.agent.select_action(state, valid, self.config.epsilon)
            selected = self.agent.select_action(
                state, decision.combined_mask, self.config.epsilon
            )
            intervened = not bool(decision.combined_mask[proposed])
            interventions += int(intervened)
            unsafe_top_q += int(intervened)
            candidate = decision.candidates[selected]
            relaxed = bool(candidate.immediate_damage and candidate.immediate_damage > 0)
            relaxations += int(relaxed)
            action_label = action_name(ACTIONS[selected])
            actions.append(action_label)
            decisions.append(
                {
                    "step": len(actions),
                    "current_minimum_damage": decision.current_cost,
                    "proposed_action": action_name(ACTIONS[proposed]),
                    "selected_action": action_label,
                    "intervened": intervened,
                    "selected_immediate_damage": candidate.immediate_damage,
                    "selected_successor_cost": candidate.successor_cost,
                    "admissible_action_count": decision.admissible_action_count,
                    "shielded_action_count": decision.shielded_action_count,
                }
            )
            observation, reward, terminated, truncated, final_info = env.step(selected)
            episode_return += reward
            state = self.encoder.encode(observation)
            if terminated or truncated:
                break

        functional = functional_recovery(env.simulator)
        exact = configuration_recovery(env.simulator.snapshot())
        metrics = env.get_metrics()
        incorrect = bool(terminated and not final_info.get("recovery_success", False))
        reason = (
            "step_limit" if truncated
            else "incorrect_declare_done" if incorrect
            else "recovery_success" if terminated and functional
            else "shield_empty_set"
        )
        row = ShieldedEvaluationRow(
            scenario_id=scenario.scenario_id,
            fault_type=scenario.fault_type,
            severity=scenario.severity,
            fault_count=scenario.fault_count,
            episode_return=episode_return,
            recovery_success=functional,
            configuration_recovery=exact,
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
            theoretical_minimum_damage=theoretical,
            shield_interventions=interventions,
            unsafe_top_q_actions=unsafe_top_q,
            mean_admissible_actions=fmean(admissible_counts),
            shield_relaxations=relaxations,
            empty_progress_actions=empty_progress,
            empty_combined_masks=empty_combined,
        )
        trajectory = {
            "scenario_id": scenario.scenario_id,
            "severity": scenario.severity,
            "theoretical_minimum_damage": theoretical,
            "actions": actions,
            "shield_decisions": decisions,
            "termination_reason": reason,
            "recovery_success": functional,
            "new_policy_damage_events": row.new_policy_damage_events,
        }
        return row, trajectory

    def _shielded_summary(self, rows, unchanged):
        overall = summarize_rows(rows)
        theoretical = sum(row.theoretical_minimum_damage for row in rows)
        realized = sum(row.new_policy_damage_events for row in rows)
        vanilla = load_vanilla_summary()
        return {
            "evaluation": {
                "epsilon": self.config.epsilon,
                "scenario_count": len(rows),
                "train_scenario_count": len(self.train_scenarios),
                "train_eval_overlap": 0,
                "parameters_unchanged": unchanged,
            },
            "checkpoint": {
                "path": self.config.checkpoint_path,
                "seed": self.checkpoint["seed"],
                "global_step": self.checkpoint["global_step"],
            },
            "overall": overall,
            "shield": {
                "total_interventions": sum(row.shield_interventions for row in rows),
                "episodes_with_intervention": sum(row.shield_interventions > 0 for row in rows),
                "mean_interventions": fmean(row.shield_interventions for row in rows),
                "unsafe_top_q_action_count": sum(row.unsafe_top_q_actions for row in rows),
                "unsafe_top_q_action_rate": (
                    sum(row.unsafe_top_q_actions for row in rows)
                    / sum(row.steps for row in rows)
                ),
                "mean_admissible_action_count": fmean(
                    row.mean_admissible_actions for row in rows
                ),
                "relaxation_count": sum(row.shield_relaxations for row in rows),
                "empty_progress_action_count": sum(
                    row.empty_progress_actions for row in rows
                ),
                "empty_combined_mask_count": sum(
                    row.empty_combined_masks for row in rows
                ),
            },
            "optimality": {
                "theoretical_minimum_total_damage": theoretical,
                "realized_total_damage": realized,
                "realized_minus_theoretical": realized - theoretical,
            },
            "vanilla_comparison": vanilla,
            "changed_avoidable_scenarios": compare_avoidable(rows),
        }

    def _save_shielded(self, rows, summary, trajectories):
        csv_path = self.output_csv
        summary_path = self.output_summary
        trajectories_path = self.output_trajectories
        for path in (csv_path, summary_path, trajectories_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(ShieldedEvaluationRow.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        trajectories_path.write_text(
            json.dumps(trajectories, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def load_vanilla_summary():
    with Path("results/dqn_eval_seed_2026_ep3000.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return {
        "functional_recovery_rate": fmean(row["recovery_success"] == "True" for row in rows),
        "configuration_recovery_rate": fmean(
            row["configuration_recovery"] == "True" for row in rows
        ),
        "mean_steps": fmean(float(row["steps"]) for row in rows),
        "total_new_policy_damage_events": sum(
            int(row["new_policy_damage_events"]) for row in rows
        ),
        "mean_new_policy_damage_events": fmean(
            int(row["new_policy_damage_events"]) for row in rows
        ),
    }


def compare_avoidable(shielded_rows):
    by_id = {row.scenario_id: row for row in shielded_rows}
    with Path("results/dqn_eval_seed_2026_ep3000.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        vanilla = list(csv.DictReader(handle))
    return [
        {
            "scenario_id": row["scenario_id"],
            "vanilla_actions": row["action_sequence"],
            "shielded_actions": by_id[row["scenario_id"]].action_sequence,
            "vanilla_damage": int(row["new_policy_damage_events"]),
            "shielded_damage": by_id[row["scenario_id"]].new_policy_damage_events,
            "vanilla_steps": int(row["steps"]),
            "shielded_steps": by_id[row["scenario_id"]].steps,
        }
        for row in vanilla
        if row["scenario_id"] in by_id
        and int(row["new_policy_damage_events"])
        > by_id[row["scenario_id"]].theoretical_minimum_damage
    ]


def evaluate_shielded_dqn(config: EvaluationConfig | None = None):
    return ShieldedDQNEvaluator(config).evaluate()


if __name__ == "__main__":
    result = evaluate_shielded_dqn()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
