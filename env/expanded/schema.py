"""Validated immutable schemas for variable-size network topologies."""

from __future__ import annotations

from dataclasses import dataclass


Pair = tuple[str, str]


def canonical_pair(left: str, right: str) -> Pair:
    if left == right:
        raise ValueError("a policy pair requires two distinct hosts")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


@dataclass(frozen=True)
class SwitchSpec:
    name: str


@dataclass(frozen=True)
class HostSpec:
    name: str
    switch: str
    port: str
    vlan: int


@dataclass(frozen=True)
class LinkSpec:
    name: str
    left_switch: str
    left_port: str
    right_switch: str
    right_port: str
    allowed_vlans: tuple[int, ...]
    nominal_enabled: bool = True


@dataclass(frozen=True)
class PolicySpec:
    forbidden_pairs: tuple[Pair, ...] = ()
    required_exceptions: tuple[Pair, ...] = ()

    def normalized_forbidden(self) -> frozenset[Pair]:
        return frozenset(canonical_pair(*pair) for pair in self.forbidden_pairs)

    def normalized_required(self) -> frozenset[Pair]:
        return frozenset(canonical_pair(*pair) for pair in self.required_exceptions)


@dataclass(frozen=True)
class TopologySpec:
    name: str
    switches: tuple[SwitchSpec, ...]
    hosts: tuple[HostSpec, ...]
    links: tuple[LinkSpec, ...]
    vlans: tuple[int, ...]
    policy: PolicySpec = PolicySpec()

    def __post_init__(self) -> None:
        switch_names = [switch.name for switch in self.switches]
        host_names = [host.name for host in self.hosts]
        link_names = [link.name for link in self.links]
        if len(set(switch_names)) != len(switch_names):
            raise ValueError("switch names must be unique")
        if len(set(host_names)) != len(host_names):
            raise ValueError("host names must be unique")
        if len(set(link_names)) != len(link_names):
            raise ValueError("link names must be unique")
        if not self.switches or not self.hosts or not self.vlans:
            raise ValueError("topology requires switches, hosts, and VLANs")
        if len(set(self.vlans)) != len(self.vlans):
            raise ValueError("VLAN IDs must be unique")
        known_switches, known_hosts = set(switch_names), set(host_names)
        used_ports: set[tuple[str, str]] = set()
        for host in self.hosts:
            if host.switch not in known_switches or host.vlan not in self.vlans:
                raise ValueError(f"invalid host attachment: {host}")
            endpoint = (host.switch, host.port)
            if endpoint in used_ports:
                raise ValueError(f"port is used more than once: {endpoint}")
            used_ports.add(endpoint)
        for link in self.links:
            if link.left_switch not in known_switches or link.right_switch not in known_switches:
                raise ValueError(f"invalid link endpoint: {link}")
            if link.left_switch == link.right_switch:
                raise ValueError("self-links are not supported")
            if not set(link.allowed_vlans) <= set(self.vlans):
                raise ValueError(f"link uses an unknown VLAN: {link}")
            for endpoint in ((link.left_switch, link.left_port), (link.right_switch, link.right_port)):
                if endpoint in used_ports:
                    raise ValueError(f"port is used more than once: {endpoint}")
                used_ports.add(endpoint)
        pairs = self.policy.normalized_forbidden() | self.policy.normalized_required()
        if any(not set(pair) <= known_hosts for pair in pairs):
            raise ValueError("policy references an unknown host")
        overlap = self.policy.normalized_forbidden() & self.policy.normalized_required()
        if overlap:
            raise ValueError(f"policy pair is both required and forbidden: {overlap}")
