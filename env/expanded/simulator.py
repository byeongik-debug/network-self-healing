"""Topology-independent VLAN, link, port, and ACL simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from env.expanded.schema import Pair, TopologySpec, canonical_pair


@dataclass
class ExpandedState:
    access_vlans: dict[str, int]
    host_ports_up: dict[str, bool]
    link_up: dict[str, bool]
    link_ports_up: dict[tuple[str, str], bool]
    trunk_vlans: dict[tuple[str, str], set[int]]
    acl_denied: set[Pair]


class ExpandedNetworkSimulator:
    def __init__(self, topology: TopologySpec) -> None:
        self.topology = topology
        self.reset()

    def reset(self) -> None:
        self.state = ExpandedState(
            access_vlans={host.name: host.vlan for host in self.topology.hosts},
            host_ports_up={host.name: True for host in self.topology.hosts},
            link_up={link.name: link.nominal_enabled for link in self.topology.links},
            link_ports_up={
                (link.name, switch): True
                for link in self.topology.links
                for switch in (link.left_switch, link.right_switch)
            },
            trunk_vlans={
                (link.name, switch): set(link.allowed_vlans)
                for link in self.topology.links
                for switch in (link.left_switch, link.right_switch)
            },
            acl_denied=set(self.topology.policy.normalized_forbidden()),
        )

    def snapshot(self) -> ExpandedState:
        return deepcopy(self.state)

    def restore(self, state: ExpandedState) -> None:
        self.state = deepcopy(state)

    def set_access_vlan(self, host: str, vlan: int) -> bool:
        if host not in self.state.access_vlans or vlan not in self.topology.vlans:
            return False
        self.state.access_vlans[host] = vlan
        return True

    def set_host_port(self, host: str, up: bool) -> bool:
        if host not in self.state.host_ports_up:
            return False
        self.state.host_ports_up[host] = up
        return True

    def set_link(self, link: str, up: bool) -> bool:
        if link not in self.state.link_up:
            return False
        self.state.link_up[link] = up
        return True

    def set_link_port(self, link: str, switch: str, up: bool) -> bool:
        key = (link, switch)
        if key not in self.state.link_ports_up:
            return False
        self.state.link_ports_up[key] = up
        return True

    def set_trunk_vlan(self, link: str, switch: str, vlan: int, allowed: bool) -> bool:
        key = (link, switch)
        if key not in self.state.trunk_vlans or vlan not in self.topology.vlans:
            return False
        self.state.trunk_vlans[key].add(vlan) if allowed else self.state.trunk_vlans[key].discard(vlan)
        return True

    def set_acl(self, left: str, right: str, allowed: bool) -> bool:
        hosts = {host.name for host in self.topology.hosts}
        if left not in hosts or right not in hosts or left == right:
            return False
        pair = canonical_pair(left, right)
        self.state.acl_denied.discard(pair) if allowed else self.state.acl_denied.add(pair)
        return True

    def can_reach(self, source: str, destination: str) -> bool:
        hosts = {host.name: host for host in self.topology.hosts}
        if source not in hosts or destination not in hosts or source == destination:
            return False
        if not self.state.host_ports_up[source] or not self.state.host_ports_up[destination]:
            return False
        if canonical_pair(source, destination) in self.state.acl_denied:
            return False
        vlan = self.state.access_vlans[source]
        if vlan != self.state.access_vlans[destination]:
            return False
        start, goal = hosts[source].switch, hosts[destination].switch
        if start == goal:
            return True
        adjacency: dict[str, set[str]] = {switch.name: set() for switch in self.topology.switches}
        for link in self.topology.links:
            endpoints = (link.left_switch, link.right_switch)
            if not self.state.link_up[link.name]:
                continue
            if not all(self.state.link_ports_up[(link.name, switch)] for switch in endpoints):
                continue
            if not all(vlan in self.state.trunk_vlans[(link.name, switch)] for switch in endpoints):
                continue
            adjacency[link.left_switch].add(link.right_switch)
            adjacency[link.right_switch].add(link.left_switch)
        frontier, visited = [start], {start}
        while frontier:
            current = frontier.pop()
            if current == goal:
                return True
            for neighbor in adjacency[current] - visited:
                visited.add(neighbor)
                frontier.append(neighbor)
        return False

    def expected_reachability(self, left: str, right: str) -> bool:
        pair = canonical_pair(left, right)
        if pair in self.topology.policy.normalized_required():
            return True
        if pair in self.topology.policy.normalized_forbidden():
            return False
        reference = {host.name: host.vlan for host in self.topology.hosts}
        return reference[left] == reference[right]

    def policy_results(self) -> dict[Pair, bool]:
        names = sorted(host.name for host in self.topology.hosts)
        return {
            canonical_pair(left, right): self.can_reach(left, right)
            == self.expected_reachability(left, right)
            for left, right in combinations(names, 2)
        }

    def is_policy_healthy(self) -> bool:
        return all(self.policy_results().values())

    def observation(self) -> dict[str, Any]:
        return {
            "topology": self.topology.name,
            "access_vlans": dict(self.state.access_vlans),
            "host_ports_up": dict(self.state.host_ports_up),
            "link_up": dict(self.state.link_up),
            "link_ports_up": {f"{link}:{switch}": up for (link, switch), up in self.state.link_ports_up.items()},
            "trunk_vlans": {f"{link}:{switch}": sorted(vlans) for (link, switch), vlans in self.state.trunk_vlans.items()},
            "acl_denied": [list(pair) for pair in sorted(self.state.acl_denied)],
        }
