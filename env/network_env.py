"""RL-compatible network environment without reward or learning logic."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Literal

from config.scenarios import EVAL_SCENARIOS, TRAIN_SCENARIOS
from .actions import ACTIONS, ActionSpec
from .faults import FaultScenario
from .simulator import NetworkSimulator

Split = Literal["train", "eval"]
DiagnosticStatus = Literal["unknown", "success", "failure"]


class NetworkEnv:
    """Deterministic VLAN recovery environment with 19 discrete actions."""

    def __init__(
        self,
        split: Split = "train",
        scenario: str | None = None,
        max_steps: int = 30,
        fault_scenario: FaultScenario | None = None,
    ) -> None:
        if split not in ("train", "eval"):
            raise ValueError("split must be 'train' or 'eval'")
        self.split, self.requested_scenario, self.max_steps = split, scenario, max_steps
        if scenario is not None and fault_scenario is not None:
            raise ValueError("scenario and fault_scenario are mutually exclusive")
        self.fault_scenario = fault_scenario
        self.simulator = NetworkSimulator()
        self._rng = random.Random()
        self._initial_faulty_state: dict[str, Any] = {}
        self._scenario_name = ""
        self._metrics: dict[str, int] = {}
        self._observed_diagnostics: dict[str, dict[str, DiagnosticStatus]] = {}
        self._change_history: list[dict[str, Any]] = []
        self._initial_damaged_policies: set[tuple[str, str]] = set()
        self._newly_damaged_policies: set[tuple[str, str]] = set()
        self._continuously_healthy_policies: set[tuple[str, str]] = set()
        self._ever_functionally_healthy = False
        self._terminated = self._truncated = False

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._rng.seed(seed)
        catalog = TRAIN_SCENARIOS if self.split == "train" else EVAL_SCENARIOS
        self.simulator.reset()
        if self.fault_scenario is not None:
            for mutation in self.fault_scenario.mutations:
                self.simulator.apply_fault(mutation.as_environment_mutation())
            self._scenario_name = self.fault_scenario.scenario_id
        else:
            scenario_name = self.requested_scenario or self._rng.choice(sorted(catalog))
            if scenario_name not in catalog:
                raise ValueError(f"Unknown {self.split} scenario: {scenario_name}")
            for mutation in catalog[scenario_name]:
                self.simulator.apply_fault(mutation)
            self._scenario_name = scenario_name
        self._initial_faulty_state = self.simulator.snapshot()
        self._change_history = []
        self._metrics = {
            "steps": 0,
            "diagnostic_actions": 0,
            "configuration_changes": 0,  # compatibility alias
            "actual_configuration_changes": 0,
            "unnecessary_actions": 0,
            "rollbacks": 0,
            "invalid_actions": 0,
            "healthy_network_impacts": 0,
            "new_policy_damage_events": 0,
            "service_damage_actions": 0,
            "post_recovery_damage_events": 0,
        }
        initial_policy = self.simulator.policy_results()
        self._initial_damaged_policies = {pair for pair, healthy in initial_policy.items() if not healthy}
        self._newly_damaged_policies = set()
        self._continuously_healthy_policies = {pair for pair, healthy in initial_policy.items() if healthy}
        self._ever_functionally_healthy = all(initial_policy.values())
        self._clear_diagnostics()
        self._terminated = self._truncated = False
        return self.get_observation(), {"split": self.split}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has ended; call reset() before step().")
        self._metrics["steps"] += 1
        info: dict[str, Any] = {"valid_action": True}
        spec = self._get_action(action)
        if spec is None:
            self._metrics["invalid_actions"] += 1
            info.update(valid_action=False, action_applied=False, error="invalid action id")
        else:
            self._apply_action(spec, info)
        self._truncated = not self._terminated and self._metrics["steps"] >= self.max_steps
        info["policy_healthy"] = self.simulator.is_policy_healthy()
        return self.get_observation(), self.calculate_reward(), self._terminated, self._truncated, info

    def calculate_reward(self) -> float:
        """Researcher-owned extension point; deliberately has no reward design."""
        return 0.0

    def get_observation(self) -> dict[str, Any]:
        return {
            "ports": self.simulator.observable_ports(),
            "link_states": {
                f"{switch}:{port}": bool(settings["link_up"])
                for switch, ports in self.simulator.ports.items()
                for port, settings in ports.items()
            },
            "connectivity": deepcopy(self._observed_diagnostics),
            "step_count": self._metrics.get("steps", 0),
        }

    def get_valid_actions(self) -> list[dict[str, Any]]:
        """Return actions enabled by current observable configuration/history."""
        return [self._serialize_action(a) for a in ACTIONS if self._action_availability(a)[0]]

    def get_action_mask(self) -> list[bool]:
        """Return a stable 19-position mask corresponding to action IDs."""
        return [self._action_availability(a)[0] for a in ACTIONS]

    def check_connectivity(self) -> dict[str, dict[str, bool]]:
        """Diagnose all host pairs and retain the legacy boolean return value."""
        actual = self.simulator.connectivity()
        self._observed_diagnostics = {
            source: {destination: "success" if value else "failure" for destination, value in row.items()}
            for source, row in actual.items()
        }
        return actual

    def get_metrics(self) -> dict[str, Any]:
        final_policy = self.simulator.policy_results()
        return {
            **self._metrics,
            "policy_healthy": all(final_policy.values()),
            "rollback_depth": len(self._change_history),
            "initial_damaged_policies": len(self._initial_damaged_policies),
            "unique_newly_damaged_policies": len(self._newly_damaged_policies),
            "continuously_healthy_policies": len(self._continuously_healthy_policies),
            "final_damaged_policies": sum(not healthy for healthy in final_policy.values()),
        }

    def restore_initial_state(self) -> bool:
        """Restore the initially injected state, separate from last-change rollback."""
        if not self._initial_faulty_state:
            return False
        changed = self.simulator.snapshot() != self._initial_faulty_state
        before = self.simulator.policy_results()
        self.simulator.restore(self._initial_faulty_state)
        self._record_service_transition(before, self.simulator.policy_results())
        self._change_history.clear()
        self._clear_diagnostics()
        return changed

    def _get_action(self, action: int) -> ActionSpec | None:
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(ACTIONS):
            return None
        return ACTIONS[action]

    @staticmethod
    def _serialize_action(action: ActionSpec) -> dict[str, Any]:
        return {"id": action.id, "kind": action.kind, "switch": action.switch, "port": action.port, "vlan": action.vlan, "allowed": action.allowed}

    def _apply_action(self, action: ActionSpec, info: dict[str, Any]) -> None:
        available, reason = self._action_availability(action)
        if not available:
            self._metrics["unnecessary_actions"] += 1
            info.update(valid_action=False, action_applied=False, reason=reason)
            return
        if action.kind == "diagnose":
            self._metrics["diagnostic_actions"] += 1
            self.check_connectivity()
            info["action_applied"] = True
        elif action.kind in ("set_access_vlan", "set_trunk_vlan"):
            before = self.simulator.policy_results()
            self._change_history.append(self.simulator.snapshot())
            if action.kind == "set_access_vlan":
                self.simulator.set_access_vlan(action.switch or "", action.port or "", action.vlan or 0)
            else:
                self.simulator.set_trunk_vlan(action.switch or "", action.port or "", action.vlan or 0, bool(action.allowed))
            after = self.simulator.policy_results()
            impact = any(ok and not after[pair] for pair, ok in before.items())
            self._metrics["configuration_changes"] += 1
            self._metrics["actual_configuration_changes"] += 1
            self._metrics["healthy_network_impacts"] += int(impact)
            self._record_service_transition(before, after)
            self._clear_diagnostics()
            info.update(action_applied=True, healthy_network_impact=impact)
        elif action.kind == "rollback":
            before = self.simulator.policy_results()
            self.simulator.restore(self._change_history.pop())
            self._record_service_transition(before, self.simulator.policy_results())
            self._metrics["rollbacks"] += 1
            self._clear_diagnostics()
            info.update(action_applied=True, rollback_type="last_change")
        elif action.kind == "declare_done":
            self._terminated = True
            info.update(action_applied=True, recovery_success=self.simulator.is_policy_healthy())

    def _action_availability(self, action: ActionSpec) -> tuple[bool, str | None]:
        if action.kind in ("diagnose", "declare_done"):
            return True, None
        if action.kind == "rollback":
            return bool(self._change_history), None if self._change_history else "no_change_to_rollback"
        ports = self.simulator.ports.get(action.switch or "")
        if ports is None or action.port not in ports:
            return False, "target_does_not_exist"
        port = ports[action.port]
        if action.kind == "set_access_vlan":
            if port.get("mode") != "access" or action.vlan not in (10, 20):
                return False, "invalid_access_setting"
            return (False, "setting_already_applied") if port.get("access_vlan") == action.vlan else (True, None)
        if action.kind == "set_trunk_vlan":
            if port.get("mode") != "trunk" or action.vlan not in (10, 20) or action.allowed is None:
                return False, "invalid_trunk_setting"
            present = action.vlan in port.get("allowed_vlans", [])
            return (False, "setting_already_applied") if present == action.allowed else (True, None)
        return False, "unsupported_action"

    def _clear_diagnostics(self) -> None:
        self._observed_diagnostics = {
            source: {destination: "unknown" for destination in row}
            for source, row in self.simulator.connectivity().items()
        }

    def _record_service_transition(
        self,
        before: dict[tuple[str, str], bool],
        after: dict[tuple[str, str], bool],
    ) -> None:
        newly_damaged = {pair for pair, was_healthy in before.items() if was_healthy and not after[pair]}
        self._metrics["new_policy_damage_events"] += len(newly_damaged)
        self._metrics["service_damage_actions"] += int(bool(newly_damaged))
        self._newly_damaged_policies.update(newly_damaged)
        self._continuously_healthy_policies.difference_update(
            pair for pair, healthy in after.items() if not healthy
        )
        if all(before.values()) and not all(after.values()):
            self._metrics["post_recovery_damage_events"] += 1
        self._ever_functionally_healthy = self._ever_functionally_healthy or all(after.values())
