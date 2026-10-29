"""Expanded environment with delayed, failed, and stale control/telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

from env.expanded.faults import ExpandedFault
from env.expanded.schema import TopologySpec
from env.expanded.simulator import ExpandedNetworkSimulator, ExpandedState


ActionKind = Literal[
    "set_access_vlan", "set_trunk_vlan", "set_link", "set_port", "set_acl",
    "diagnose", "declare_done",
]


@dataclass(frozen=True)
class ExpandedAction:
    kind: ActionKind
    target: str | None = None
    secondary: str | None = None
    vlan: int | None = None
    enabled: bool | None = None


@dataclass
class PendingAction:
    remaining_steps: int
    action: ExpandedAction


class ExpandedNetworkEnv:
    def __init__(
        self,
        topology: TopologySpec,
        *,
        max_steps: int = 60,
        command_failure_probability: float = 0.0,
        actuation_delay: int = 0,
        telemetry_delay: int = 0,
        stale_probability: float = 0.0,
    ) -> None:
        for name, value in (
            ("command_failure_probability", command_failure_probability),
            ("stale_probability", stale_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if actuation_delay < 0 or telemetry_delay < 0 or max_steps <= 0:
            raise ValueError("delays must be non-negative and max_steps positive")
        self.topology = topology
        self.max_steps = max_steps
        self.command_failure_probability = command_failure_probability
        self.actuation_delay = actuation_delay
        self.telemetry_delay = telemetry_delay
        self.stale_probability = stale_probability
        self.simulator = ExpandedNetworkSimulator(topology)
        self._rng = Random()
        self._pending: list[PendingAction] = []
        self._history: list[ExpandedState] = []
        self._telemetry: deque[dict[str, Any]] = deque()
        self._steps = self._action_failures = self._damage = 0
        self._terminated = self._truncated = False
        self._last_observation: dict[str, Any] | None = None

    def reset(
        self, faults: tuple[ExpandedFault, ...] = (), seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._rng.seed(seed)
        self.simulator.reset()
        for fault in faults:
            fault.apply(self.simulator)
        self._pending, self._history = [], []
        self._steps = self._action_failures = self._damage = 0
        self._terminated = self._truncated = False
        initial = self._ground_truth_observation()
        self._telemetry = deque(
            [initial] * (self.telemetry_delay + 2),
            maxlen=self.telemetry_delay + 3,
        )
        observation = self._observation()
        self._last_observation = observation
        return observation, {"fault_ids": [fault.fault_id for fault in faults]}

    def step(self, action: ExpandedAction):
        if self._terminated or self._truncated:
            raise RuntimeError("episode has ended")
        self._steps += 1
        applied_pending = self._advance_pending()
        info: dict[str, Any] = {"applied_pending": applied_pending}
        if action.kind == "declare_done":
            self._terminated = True
            info["recovery_success"] = self.simulator.is_policy_healthy()
        elif action.kind == "diagnose":
            info["diagnostic"] = self.simulator.policy_results()
        elif self._rng.random() < self.command_failure_probability:
            self._action_failures += 1
            info.update(action_failed=True, action_applied=False)
        elif self.actuation_delay:
            self._pending.append(PendingAction(self.actuation_delay, action))
            info.update(action_queued=True, action_applied=False)
        else:
            self._apply_with_metrics(action)
            info["action_applied"] = True
        self._truncated = not self._terminated and self._steps >= self.max_steps
        self._telemetry.append(self._ground_truth_observation())
        info.update(policy_healthy=self.simulator.is_policy_healthy())
        reward = 10.0 if self._terminated and info["policy_healthy"] else -0.05
        observation = self._observation()
        self._last_observation = observation
        return observation, reward, self._terminated, self._truncated, info

    def _advance_pending(self) -> int:
        applied = 0
        remaining: list[PendingAction] = []
        for pending in self._pending:
            pending.remaining_steps -= 1
            if pending.remaining_steps <= 0:
                self._apply_with_metrics(pending.action)
                applied += 1
            else:
                remaining.append(pending)
        self._pending = remaining
        return applied

    def _apply_with_metrics(self, action: ExpandedAction) -> None:
        before = self.simulator.policy_results()
        self._history.append(self.simulator.snapshot())
        if not self._apply(action):
            self._history.pop()
            raise ValueError(f"invalid expanded action: {action}")
        after = self.simulator.policy_results()
        self._damage += sum(before[pair] and not after[pair] for pair in before)

    def _apply(self, action: ExpandedAction) -> bool:
        if action.kind == "set_access_vlan":
            return action.target is not None and action.vlan is not None \
                and self.simulator.set_access_vlan(action.target, action.vlan)
        if action.kind == "set_trunk_vlan":
            return action.target is not None and action.secondary is not None \
                and action.vlan is not None and action.enabled is not None \
                and self.simulator.set_trunk_vlan(
                    action.target, action.secondary, action.vlan, action.enabled
                )
        if action.kind == "set_link":
            return action.target is not None and action.enabled is not None \
                and self.simulator.set_link(action.target, action.enabled)
        if action.kind == "set_port":
            if action.target is None or action.enabled is None:
                return False
            return (self.simulator.set_host_port(action.target, action.enabled)
                    if action.secondary is None else
                    self.simulator.set_link_port(action.target, action.secondary, action.enabled))
        if action.kind == "set_acl":
            return action.target is not None and action.secondary is not None \
                and action.enabled is not None \
                and self.simulator.set_acl(action.target, action.secondary, action.enabled)
        return False

    def _ground_truth_observation(self) -> dict[str, Any]:
        return {
            **self.simulator.observation(),
            "step": self._steps,
            "pending_commands": len(self._pending),
        }

    def _observation(self) -> dict[str, Any]:
        delay = self.telemetry_delay
        if self.stale_probability and self._rng.random() < self.stale_probability:
            delay += 1
        index = max(0, len(self._telemetry) - 1 - delay)
        observation = dict(list(self._telemetry)[index])
        observation["telemetry_age"] = delay
        observation["pending_commands"] = len(self._pending)
        return observation

    def metrics(self) -> dict[str, Any]:
        return {
            "steps": self._steps,
            "action_failures": self._action_failures,
            "pending_commands": len(self._pending),
            "new_policy_damage_events": self._damage,
            "policy_healthy": self.simulator.is_policy_healthy(),
        }

    def shield_simulator(self, view: Literal["true", "observed"] = "true") -> ExpandedNetworkSimulator:
        """Return the state model available to a shield under ``view``.

        ``observed`` reconstructs exactly the last observation returned to the
        policy; it does not draw another stale-telemetry sample.
        """
        if view == "true":
            return self.simulator
        if view != "observed":
            raise ValueError("shield_view must be 'true' or 'observed'")
        if self._last_observation is None:
            raise RuntimeError("reset must be called before requesting shield state")
        observation = self._last_observation
        simulator = ExpandedNetworkSimulator(self.topology)
        simulator.state = ExpandedState(
            access_vlans={key: int(value) for key, value in observation["access_vlans"].items()},
            host_ports_up={key: bool(value) for key, value in observation["host_ports_up"].items()},
            link_up={key: bool(value) for key, value in observation["link_up"].items()},
            link_ports_up={tuple(key.split(":", 1)): bool(value)
                           for key, value in observation["link_ports_up"].items()},
            trunk_vlans={tuple(key.split(":", 1)): set(map(int, values))
                         for key, values in observation["trunk_vlans"].items()},
            acl_denied={tuple(sorted(pair)) for pair in observation["acl_denied"]},
        )
        return simulator
