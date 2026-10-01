"""Small deterministic layer-2 VLAN simulator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from config.topology import HOSTS, REFERENCE_PORTS


class NetworkSimulator:
    """Models access VLANs, one inter-switch trunk, and physical link state."""

    def __init__(self) -> None:
        self.ports: dict[str, dict[str, dict[str, Any]]] = {}
        self.reset()

    def reset(self) -> None:
        self.ports = deepcopy(REFERENCE_PORTS)

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        return deepcopy(self.ports)

    def restore(self, snapshot: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
        self.ports = deepcopy(snapshot)  # type: ignore[assignment]

    def set_access_vlan(self, switch: str, port: str, vlan: int) -> bool:
        target = self._port(switch, port)
        if target is None or target.get("mode") != "access" or vlan not in (10, 20):
            return False
        target["access_vlan"] = vlan
        return True

    def set_trunk_vlan(self, switch: str, port: str, vlan: int, allowed: bool) -> bool:
        target = self._port(switch, port)
        if target is None or target.get("mode") != "trunk" or vlan not in (10, 20):
            return False
        current = set(target["allowed_vlans"])
        current.add(vlan) if allowed else current.discard(vlan)
        target["allowed_vlans"] = sorted(current)
        return True

    def apply_fault(self, mutation: Mapping[str, object]) -> None:
        kind = mutation["kind"]
        switch, port, value = str(mutation["switch"]), str(mutation["port"]), int(mutation["value"])
        if kind == "access_vlan":
            self.set_access_vlan(switch, port, value)
        elif kind == "trunk_remove":
            self.set_trunk_vlan(switch, port, value, False)
        else:
            raise ValueError(f"Unsupported fault mutation: {kind}")

    def connectivity(self) -> dict[str, dict[str, bool]]:
        return {
            source: {
                destination: self.can_reach(source, destination)
                for destination in HOSTS
                if destination != source
            }
            for source in HOSTS
        }

    def can_reach(self, source: str, destination: str) -> bool:
        if source not in HOSTS or destination not in HOSTS:
            return False
        src, dst = HOSTS[source], HOSTS[destination]
        src_port = self.ports[str(src["switch"])][str(src["port"])]
        dst_port = self.ports[str(dst["switch"])][str(dst["port"])]
        if not src_port["link_up"] or not dst_port["link_up"]:
            return False
        src_vlan, dst_vlan = src_port["access_vlan"], dst_port["access_vlan"]
        if src_vlan != dst_vlan:
            return False
        if src["switch"] == dst["switch"]:
            return True
        left = self.ports["SW1"]["eth3"]
        right = self.ports["SW2"]["eth3"]
        return bool(
            left["link_up"]
            and right["link_up"]
            and src_vlan in left["allowed_vlans"]
            and src_vlan in right["allowed_vlans"]
        )

    def is_policy_healthy(self) -> bool:
        return all(self.policy_results().values())

    def policy_results(self) -> dict[tuple[str, str], bool]:
        """Return whether each unordered host pair currently follows VLAN policy."""
        results: dict[tuple[str, str], bool] = {}
        for source, src_data in HOSTS.items():
            for destination, dst_data in HOSTS.items():
                if source >= destination:
                    continue
                expected = src_data["vlan"] == dst_data["vlan"]
                results[(source, destination)] = self.can_reach(source, destination) == expected
        return results

    def observable_ports(self) -> dict[str, dict[str, dict[str, Any]]]:
        return deepcopy(self.ports)

    def _port(self, switch: str, port: str) -> dict[str, Any] | None:
        return self.ports.get(switch, {}).get(port)
