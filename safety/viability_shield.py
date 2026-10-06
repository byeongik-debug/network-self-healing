"""Exact minimum-damage viability shield for the finite VLAN simulator."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from math import inf
from typing import Any, Final, Iterable

import numpy as np
from numpy.typing import NDArray

from env.actions import ACTIONS, ActionSpec
from env.network_env import NetworkEnv
from env.simulator import NetworkSimulator


ConfigurationKey = tuple[int, int, bool, bool, int, int, bool, bool]
CONFIGURATION_KINDS: Final = frozenset(("set_access_vlan", "set_trunk_vlan"))


@dataclass(frozen=True)
class CandidateSafety:
    action_id: int
    valid: bool
    immediate_damage: int | None
    successor_cost: int | None
    admissible: bool


@dataclass(frozen=True)
class ShieldDecision:
    current_cost: int
    validity_mask: NDArray[np.bool_]
    safety_mask: NDArray[np.bool_]
    combined_mask: NDArray[np.bool_]
    candidates: tuple[CandidateSafety, ...]

    @property
    def shielded_action_count(self) -> int:
        return int(np.count_nonzero(self.validity_mask & ~self.combined_mask))

    @property
    def admissible_action_count(self) -> int:
        return int(np.count_nonzero(self.combined_mask))


def canonical_configuration(source: NetworkEnv | NetworkSimulator | dict[str, Any]) -> ConfigurationKey:
    """Return the deterministic hash key used only by the shield graph."""
    if isinstance(source, NetworkEnv):
        ports = source.simulator.ports
    elif isinstance(source, NetworkSimulator):
        ports = source.ports
    else:
        ports = source
    return (
        int(ports["SW1"]["eth1"]["access_vlan"]),
        int(ports["SW1"]["eth2"]["access_vlan"]),
        10 in ports["SW1"]["eth3"]["allowed_vlans"],
        20 in ports["SW1"]["eth3"]["allowed_vlans"],
        int(ports["SW2"]["eth1"]["access_vlan"]),
        int(ports["SW2"]["eth2"]["access_vlan"]),
        10 in ports["SW2"]["eth3"]["allowed_vlans"],
        20 in ports["SW2"]["eth3"]["allowed_vlans"],
    )


def simulator_from_key(key: ConfigurationKey) -> NetworkSimulator:
    simulator = NetworkSimulator()
    simulator.set_access_vlan("SW1", "eth1", key[0])
    simulator.set_access_vlan("SW1", "eth2", key[1])
    simulator.set_trunk_vlan("SW1", "eth3", 10, key[2])
    simulator.set_trunk_vlan("SW1", "eth3", 20, key[3])
    simulator.set_access_vlan("SW2", "eth1", key[4])
    simulator.set_access_vlan("SW2", "eth2", key[5])
    simulator.set_trunk_vlan("SW2", "eth3", 10, key[6])
    simulator.set_trunk_vlan("SW2", "eth3", 20, key[7])
    return simulator


def safety_cost(before: NetworkSimulator, after: NetworkSimulator) -> int:
    """Count policies that transition from healthy to damaged."""
    before_policy = before.policy_results()
    after_policy = after.policy_results()
    return sum(
        was_healthy and not after_policy[pair]
        for pair, was_healthy in before_policy.items()
    )


class MinimumDamageViabilityShield:
    """Allow exactly the actions lying on a minimum cumulative-damage path."""

    def __init__(self) -> None:
        self.configuration_actions = tuple(
            action for action in ACTIONS if action.kind in CONFIGURATION_KINDS
        )
        self.states = tuple(self._all_states())
        self._successors: dict[tuple[ConfigurationKey, int], ConfigurationKey] = {}
        self._edge_costs: dict[tuple[ConfigurationKey, int], int] = {}
        self._build_graph()
        self.minimum_damage = self._reverse_dijkstra()

    @staticmethod
    def _all_states() -> Iterable[ConfigurationKey]:
        return product((10, 20), (10, 20), (False, True), (False, True),
                       (10, 20), (10, 20), (False, True), (False, True))

    def _build_graph(self) -> None:
        for key in self.states:
            before = simulator_from_key(key)
            for action in self.configuration_actions:
                after = self._apply_configuration(before, action)
                successor = canonical_configuration(after)
                self._successors[(key, action.id)] = successor
                self._edge_costs[(key, action.id)] = safety_cost(before, after)

    def _reverse_dijkstra(self) -> dict[ConfigurationKey, int]:
        predecessors: dict[ConfigurationKey, list[tuple[ConfigurationKey, int]]] = {
            key: [] for key in self.states
        }
        for (source, action_id), target in self._successors.items():
            if target != source:
                predecessors[target].append(
                    (source, self._edge_costs[(source, action_id)])
                )
        distance = {key: inf for key in self.states}
        queue: list[tuple[int, ConfigurationKey]] = []
        for key in self.states:
            if simulator_from_key(key).is_policy_healthy():
                distance[key] = 0
                heappush(queue, (0, key))
        while queue:
            current_cost, target = heappop(queue)
            if current_cost != distance[target]:
                continue
            for source, edge_cost in predecessors[target]:
                candidate = current_cost + edge_cost
                if candidate < distance[source]:
                    distance[source] = candidate
                    heappush(queue, (candidate, source))
        if any(value == inf for value in distance.values()):
            raise RuntimeError("configuration graph contains a state with no functional path")
        return {key: int(value) for key, value in distance.items()}

    @staticmethod
    def _apply_configuration(
        before: NetworkSimulator, action: ActionSpec
    ) -> NetworkSimulator:
        after = NetworkSimulator()
        after.restore(before.snapshot())
        if action.kind == "set_access_vlan":
            after.set_access_vlan(
                action.switch or "", action.port or "", action.vlan or 0
            )
        elif action.kind == "set_trunk_vlan":
            after.set_trunk_vlan(
                action.switch or "",
                action.port or "",
                action.vlan or 0,
                bool(action.allowed),
            )
        else:
            raise ValueError(f"not a configuration action: {action.kind}")
        return after

    def cost_to_goal(self, source: NetworkEnv | NetworkSimulator | dict[str, Any]) -> int:
        return self.minimum_damage[canonical_configuration(source)]

    def transition(
        self, key: ConfigurationKey, action_id: int
    ) -> tuple[ConfigurationKey, int]:
        return (
            self._successors[(key, action_id)],
            self._edge_costs[(key, action_id)],
        )

    def get_safety_mask(self, env: NetworkEnv) -> NDArray[np.bool_]:
        return self.get_decision(env).safety_mask

    def get_combined_mask(
        self,
        env: NetworkEnv,
        valid_mask: NDArray[np.bool_] | list[bool] | None = None,
    ) -> NDArray[np.bool_]:
        return self.get_decision(env, valid_mask).combined_mask

    def get_decision(
        self,
        env: NetworkEnv,
        valid_mask: NDArray[np.bool_] | list[bool] | None = None,
    ) -> ShieldDecision:
        """Evaluate candidates without mutating any environment state."""
        validity = np.asarray(
            env.get_action_mask() if valid_mask is None else valid_mask,
            dtype=np.bool_,
        )
        if validity.shape != (len(ACTIONS),):
            raise ValueError(f"valid_mask must have shape ({len(ACTIONS)},)")
        key = canonical_configuration(env)
        current_cost = self.minimum_damage[key]
        safety = np.zeros(len(ACTIONS), dtype=np.bool_)
        candidates: list[CandidateSafety] = []
        for action in ACTIONS:
            immediate: int | None = None
            successor_cost: int | None = None
            admissible = False
            if validity[action.id]:
                if action.kind in ("diagnose", "declare_done"):
                    immediate, successor_cost, admissible = 0, current_cost, True
                elif action.kind in CONFIGURATION_KINDS:
                    successor, immediate = self.transition(key, action.id)
                    successor_cost = self.minimum_damage[successor]
                    admissible = immediate + successor_cost == current_cost
                elif action.kind == "rollback":
                    immediate, successor_cost = self._rollback_cost(env)
                    admissible = immediate + successor_cost == current_cost
            safety[action.id] = admissible
            candidates.append(
                CandidateSafety(
                    action.id,
                    bool(validity[action.id]),
                    immediate,
                    successor_cost,
                    admissible,
                )
            )
        combined = validity & safety
        if not np.any(combined):
            raise RuntimeError("shield produced an empty combined action set")
        return ShieldDecision(
            current_cost,
            validity.copy(),
            safety,
            combined,
            tuple(candidates),
        )

    def _rollback_cost(self, env: NetworkEnv) -> tuple[int, int]:
        if not env._change_history:  # guarded by the validity mask
            raise RuntimeError("rollback evaluated without history")
        before = env.simulator
        after = NetworkSimulator()
        after.restore(env._change_history[-1])
        return safety_cost(before, after), self.cost_to_goal(after)
