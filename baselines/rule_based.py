"""Observation-only heuristic recovery baseline."""

from __future__ import annotations

from typing import Any

from .common import ActionExecutor, RecoveryEnvironment, RecoveryResult


class RuleBasedRecovery:
    """Repair common VLAN inconsistencies without reference config or fault labels."""

    name = "rule_based"
    information_access = "observation_and_diagnostics_only"

    def recover(self, env: RecoveryEnvironment, max_steps: int = 30) -> RecoveryResult:
        executor = ActionExecutor(env, max_steps)
        executor.run(executor.find("diagnose"))
        if executor.done:
            return executor.result(self.name)

        observation = env.get_observation()
        for switch, port, vlan in self._access_repairs(observation):
            executor.run(executor.find("set_access_vlan", switch=switch, port=port, vlan=vlan, allowed=None))
            if executor.done:
                return executor.result(self.name)

        observation = env.get_observation()
        for switch, port, vlan in self._trunk_repairs(observation):
            executor.run(executor.find("set_trunk_vlan", switch=switch, port=port, vlan=vlan, allowed=True))
            if executor.done:
                return executor.result(self.name)

        executor.run(executor.find("diagnose"))
        if not executor.done:
            executor.run(executor.find("declare_done"))
        return executor.result(self.name)

    @staticmethod
    def _access_repairs(observation: dict[str, Any]) -> list[tuple[str, str, int]]:
        ports = observation["ports"]
        switches = sorted(ports)
        access_names = sorted(
            set.intersection(*[
                {name for name, data in ports[switch].items() if data.get("mode") == "access"}
                for switch in switches
            ])
        )
        repairs: list[tuple[str, str, int]] = []
        planned = deepcopy_ports(ports)
        for port_name in access_names:
            values = {int(planned[switch][port_name]["access_vlan"]) for switch in switches}
            if len(values) <= 1:
                continue
            # Prefer the candidate that preserves distinct access VLANs within each switch.
            def score(candidate: int) -> tuple[int, int]:
                unique_score = sum(
                    all(
                        name == port_name or data.get("mode") != "access" or data.get("access_vlan") != candidate
                        for name, data in planned[switch].items()
                    )
                    for switch in switches
                )
                frequency = sum(
                    data.get("access_vlan") == candidate
                    for switch in switches
                    for data in planned[switch].values()
                    if data.get("mode") == "access"
                )
                return unique_score, frequency

            chosen = max(sorted(values), key=score)
            for switch in switches:
                if planned[switch][port_name]["access_vlan"] != chosen:
                    repairs.append((switch, port_name, chosen))
                    planned[switch][port_name]["access_vlan"] = chosen
        return repairs

    @staticmethod
    def _trunk_repairs(observation: dict[str, Any]) -> list[tuple[str, str, int]]:
        ports = observation["ports"]
        active_vlans = {
            int(data["access_vlan"])
            for switch_ports in ports.values()
            for data in switch_ports.values()
            if data.get("mode") == "access"
        }
        return [
            (switch, port_name, vlan)
            for switch, switch_ports in sorted(ports.items())
            for port_name, data in sorted(switch_ports.items())
            if data.get("mode") == "trunk"
            for vlan in sorted(active_vlans)
            if vlan not in data.get("allowed_vlans", [])
        ]


def deepcopy_ports(ports: dict[str, Any]) -> dict[str, Any]:
    """Copy only observable port data without reaching into environment internals."""
    return {
        switch: {port: dict(settings) for port, settings in switch_ports.items()}
        for switch, switch_ports in ports.items()
    }

