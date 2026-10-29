"""Heterogeneous fault definitions for expanded topologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from env.expanded.schema import TopologySpec, canonical_pair
from env.expanded.simulator import ExpandedNetworkSimulator


FaultKind = Literal[
    "access_vlan", "trunk_remove", "link_down", "port_shutdown",
    "acl_deny_required", "acl_allow_forbidden",
]


@dataclass(frozen=True, order=True)
class ExpandedFault:
    kind: FaultKind
    target: str
    secondary: str | None = None
    vlan: int | None = None

    @property
    def fault_id(self) -> str:
        return ":".join(str(value) for value in (self.kind, self.target, self.secondary, self.vlan) if value is not None)

    def apply(self, simulator: ExpandedNetworkSimulator) -> None:
        if self.kind == "access_vlan":
            if self.vlan is None or not simulator.set_access_vlan(self.target, self.vlan):
                raise ValueError(f"invalid access fault: {self}")
        elif self.kind == "trunk_remove":
            if self.secondary is None or self.vlan is None or not simulator.set_trunk_vlan(
                self.target, self.secondary, self.vlan, False
            ):
                raise ValueError(f"invalid trunk fault: {self}")
        elif self.kind == "link_down":
            if not simulator.set_link(self.target, False):
                raise ValueError(f"invalid link fault: {self}")
        elif self.kind == "port_shutdown":
            success = (simulator.set_host_port(self.target, False) if self.secondary is None
                       else simulator.set_link_port(self.target, self.secondary, False))
            if not success:
                raise ValueError(f"invalid port fault: {self}")
        elif self.kind in ("acl_deny_required", "acl_allow_forbidden"):
            if self.secondary is None or not simulator.set_acl(
                self.target, self.secondary, self.kind == "acl_allow_forbidden"
            ):
                raise ValueError(f"invalid ACL fault: {self}")


def atomic_faults(topology: TopologySpec) -> tuple[ExpandedFault, ...]:
    faults: list[ExpandedFault] = []
    for host in topology.hosts:
        alternative = next(vlan for vlan in topology.vlans if vlan != host.vlan)
        faults.append(ExpandedFault("access_vlan", host.name, vlan=alternative))
        faults.append(ExpandedFault("port_shutdown", host.name))
    for link in topology.links:
        if link.nominal_enabled:
            faults.append(ExpandedFault("link_down", link.name))
            for switch in (link.left_switch, link.right_switch):
                faults.append(ExpandedFault("port_shutdown", link.name, switch))
                for vlan in topology.vlans:
                    faults.append(ExpandedFault("trunk_remove", link.name, switch, vlan))
    simulator = ExpandedNetworkSimulator(topology)
    required = [pair for pair, healthy in simulator.policy_results().items()
                if healthy and simulator.expected_reachability(*pair)]
    if required:
        faults.append(ExpandedFault("acl_deny_required", *required[0]))
    for pair in sorted(topology.policy.normalized_forbidden()):
        faults.append(ExpandedFault("acl_allow_forbidden", *pair))
    return tuple(sorted(faults, key=lambda fault: fault.fault_id))
