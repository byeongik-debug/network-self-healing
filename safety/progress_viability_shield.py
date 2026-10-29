"""Lexicographic safety/liveness shield.

The original viability shield preserves minimum cumulative damage but admits
zero-progress control actions and reversible cycles.  This ablation refines the
objective lexicographically: first minimize damage, then minimize the number of
configuration transitions to a healthy state.  Once healthy, exactly one fresh
diagnosis is permitted before declaration.
"""

from __future__ import annotations

from dataclasses import replace
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

from env.actions import ACTIONS
from env.network_env import NetworkEnv
from safety.viability_shield import (
    CONFIGURATION_KINDS,
    CandidateSafety,
    ConfigurationKey,
    MinimumDamageViabilityShield,
    ShieldDecision,
    canonical_configuration,
)


class ProgressViabilityShield(MinimumDamageViabilityShield):
    """Guarantee finite progress without weakening minimum-damage safety."""

    def __init__(self) -> None:
        super().__init__()
        self.minimum_steps = self._minimum_steps_on_safe_paths()

    def _minimum_steps_on_safe_paths(self) -> dict[ConfigurationKey, int]:
        """Shortest path length among paths attaining ``minimum_damage``."""
        predecessors: dict[ConfigurationKey, list[ConfigurationKey]] = {
            key: [] for key in self.states
        }
        for (source, action_id), target in self._successors.items():
            if target == source:
                continue
            edge_cost = self._edge_costs[(source, action_id)]
            if edge_cost + self.minimum_damage[target] == self.minimum_damage[source]:
                predecessors[target].append(source)

        infinity = len(self.states) + 1
        distance = {key: infinity for key in self.states}
        queue: list[tuple[int, ConfigurationKey]] = []
        for key in self.states:
            if self.minimum_damage[key] == 0 and self.cost_to_goal_key_is_healthy(key):
                distance[key] = 0
                heappush(queue, (0, key))
        while queue:
            current, target = heappop(queue)
            if current != distance[target]:
                continue
            for source in predecessors[target]:
                candidate = current + 1
                if candidate < distance[source]:
                    distance[source] = candidate
                    heappush(queue, (candidate, source))
        if any(value == infinity for value in distance.values()):
            raise RuntimeError("a minimum-damage state has no finite progress path")
        return distance

    @staticmethod
    def cost_to_goal_key_is_healthy(key: ConfigurationKey) -> bool:
        from safety.viability_shield import simulator_from_key

        return simulator_from_key(key).is_policy_healthy()

    @staticmethod
    def _diagnostics_unknown(env: NetworkEnv) -> bool:
        return any(
            value == "unknown"
            for row in env._observed_diagnostics.values()
            for value in row.values()
        )

    def get_decision(
        self,
        env: NetworkEnv,
        valid_mask: NDArray[np.bool_] | list[bool] | None = None,
    ) -> ShieldDecision:
        base = super().get_decision(env, valid_mask)
        key = canonical_configuration(env)
        current_steps = self.minimum_steps[key]
        healthy = env.simulator.is_policy_healthy()
        progress = np.zeros(len(ACTIONS), dtype=np.bool_)
        candidates: list[CandidateSafety] = []

        for action, candidate in zip(ACTIONS, base.candidates):
            admissible = False
            if candidate.admissible:
                if healthy:
                    if action.kind == "diagnose":
                        admissible = self._diagnostics_unknown(env)
                    elif action.kind == "declare_done":
                        admissible = not self._diagnostics_unknown(env)
                elif action.kind in CONFIGURATION_KINDS:
                    successor, _ = self.transition(key, action.id)
                    admissible = self.minimum_steps[successor] == current_steps - 1
                elif action.kind == "rollback" and env._change_history:
                    successor = canonical_configuration(env._change_history[-1])
                    admissible = self.minimum_steps[successor] == current_steps - 1
            progress[action.id] = admissible
            candidates.append(replace(candidate, admissible=admissible))

        combined = base.validity_mask & progress
        if not np.any(combined):
            raise RuntimeError("progress shield produced an empty combined action set")
        return ShieldDecision(
            base.current_cost,
            base.validity_mask,
            progress,
            combined,
            tuple(candidates),
        )
