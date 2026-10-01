"""Privileged reference-configuration comparison baseline."""

from __future__ import annotations

from typing import Any

from config.topology import REFERENCE_PORTS
from .common import ActionExecutor, RecoveryEnvironment, RecoveryResult


class ConfigurationDiffRecovery:
    """Apply only settings that differ from the known reference configuration."""

    name = "configuration_diff"
    information_access = "observation_plus_reference_configuration"

    def recover(self, env: RecoveryEnvironment, max_steps: int = 30) -> RecoveryResult:
        executor = ActionExecutor(env, max_steps)
        current = env.get_observation()["ports"]
        for switch, reference_ports in sorted(REFERENCE_PORTS.items()):
            for port, reference in sorted(reference_ports.items()):
                actual = current.get(switch, {}).get(port)
                if actual is None or actual.get("mode") != reference.get("mode"):
                    continue  # No action in the fixed space can create/change a port mode.
                if reference["mode"] == "access" and actual.get("access_vlan") != reference["access_vlan"]:
                    executor.run(executor.find(
                        "set_access_vlan", switch=switch, port=port,
                        vlan=int(reference["access_vlan"]), allowed=None,
                    ))
                elif reference["mode"] == "trunk":
                    executor = self._repair_trunk(executor, switch, port, actual, reference)
                if executor.done:
                    return executor.result(self.name)
        executor.run(executor.find("declare_done"))
        return executor.result(self.name)

    @staticmethod
    def _repair_trunk(
        executor: ActionExecutor,
        switch: str,
        port: str,
        actual: dict[str, Any],
        reference: dict[str, Any],
    ) -> ActionExecutor:
        actual_vlans = set(actual.get("allowed_vlans", []))
        reference_vlans = set(reference.get("allowed_vlans", []))
        for vlan in sorted(actual_vlans - reference_vlans):
            executor.run(executor.find("set_trunk_vlan", switch=switch, port=port, vlan=vlan, allowed=False))
        for vlan in sorted(reference_vlans - actual_vlans):
            executor.run(executor.find("set_trunk_vlan", switch=switch, port=port, vlan=vlan, allowed=True))
        return executor
