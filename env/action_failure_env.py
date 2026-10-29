"""Network environment with stochastic no-op configuration command failures."""

from __future__ import annotations

from typing import Any

from env.actions import ActionSpec
from env.network_env import NetworkEnv


class ActionFailureNetworkEnv(NetworkEnv):
    """Model command rejection/loss while preserving observable actual state.

    A failed configuration command consumes one environment step but leaves the
    switch configuration unchanged. The next observation therefore exposes the
    unchanged port state and permits the runtime shield to re-plan.
    """

    def __init__(self, *args: Any, action_failure_probability: float, **kwargs: Any) -> None:
        if not 0.0 <= action_failure_probability <= 1.0:
            raise ValueError("action_failure_probability must be in [0, 1]")
        super().__init__(*args, **kwargs)
        self.action_failure_probability = action_failure_probability

    def reset(self, seed: int | None = None):
        observation, info = super().reset(seed=seed)
        self._metrics["action_failures"] = 0
        return observation, info

    def _apply_action(self, action: ActionSpec, info: dict[str, Any]) -> None:
        if (
            action.kind in ("set_access_vlan", "set_trunk_vlan")
            and self._rng.random() < self.action_failure_probability
        ):
            self._metrics["action_failures"] += 1
            info.update(
                action_applied=False,
                action_failed=True,
                failure_mode="configuration_command_noop",
            )
            return
        super()._apply_action(action, info)
