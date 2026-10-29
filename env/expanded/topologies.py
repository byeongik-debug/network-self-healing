"""Frozen topology suite for scale and generalization experiments."""

from __future__ import annotations

from env.expanded.schema import HostSpec, LinkSpec, PolicySpec, SwitchSpec, TopologySpec


def _hosts(count: int, switch_count: int, vlans: tuple[int, ...]) -> tuple[HostSpec, ...]:
    per_switch = [0] * switch_count
    hosts = []
    for index in range(count):
        switch_index = index % switch_count
        per_switch[switch_index] += 1
        vlan = vlans[(index // switch_count + switch_index) % len(vlans)]
        hosts.append(HostSpec(
            name=f"H{index + 1}", switch=f"SW{switch_index + 1}",
            port=f"eth{per_switch[switch_index]}", vlan=vlan,
        ))
    return tuple(hosts)


def _links(edges, host_count: int, switch_count: int, vlans: tuple[int, ...]) -> tuple[LinkSpec, ...]:
    host_ports = [sum(index % switch_count == switch for index in range(host_count))
                  for switch in range(switch_count)]
    next_port = [value + 1 for value in host_ports]
    links = []
    for index, (left, right, enabled) in enumerate(edges, start=1):
        left_port, right_port = next_port[left - 1], next_port[right - 1]
        next_port[left - 1] += 1
        next_port[right - 1] += 1
        links.append(LinkSpec(
            name=f"L{index}", left_switch=f"SW{left}", left_port=f"eth{left_port}",
            right_switch=f"SW{right}", right_port=f"eth{right_port}",
            allowed_vlans=vlans, nominal_enabled=enabled,
        ))
    return tuple(links)


def build_topology(name: str) -> TopologySpec:
    definitions = {
        "T2": (2, 4, (10, 20), ((1, 2, True),)),
        "T3": (3, 8, (10, 20, 30), ((1, 2, True), (2, 3, True), (3, 1, False))),
        "T4-Line": (4, 16, (10, 20, 30, 40), ((1, 2, True), (2, 3, True), (3, 4, True))),
        "T4-Ring": (4, 16, (10, 20, 30, 40), ((1, 2, True), (2, 3, True), (3, 4, True), (4, 1, True))),
        "T5-Mesh": (
            5, 20, (10, 20, 30, 40, 50),
            ((1, 2, True), (2, 3, True), (3, 4, True), (4, 5, True),
             (5, 1, True), (1, 3, True), (2, 5, True)),
        ),
        "T6-Mesh": (
            6, 24, (10, 20, 30, 40, 50, 60),
            ((1, 2, True), (2, 3, True), (3, 4, True), (4, 5, True),
             (5, 6, True), (6, 1, True), (1, 4, True), (2, 5, True),
             (3, 6, True)),
        ),
    }
    if name not in definitions:
        raise KeyError(f"unknown topology: {name}")
    switch_count, host_count, vlans, edges = definitions[name]
    hosts = (
        (
            HostSpec("H1", "SW1", "eth1", 10),
            HostSpec("H2", "SW1", "eth2", 20),
            HostSpec("H3", "SW2", "eth1", 10),
            HostSpec("H4", "SW2", "eth2", 20),
        )
        if name == "T2" else _hosts(host_count, switch_count, vlans)
    )
    same_vlan = next((left.name, right.name)
                     for index, left in enumerate(hosts)
                     for right in hosts[index + 1:] if left.vlan == right.vlan)
    return TopologySpec(
        name=name,
        switches=tuple(SwitchSpec(f"SW{i}") for i in range(1, switch_count + 1)),
        hosts=hosts,
        links=_links(edges, host_count, switch_count, vlans),
        vlans=vlans,
        policy=PolicySpec(forbidden_pairs=() if name == "T2" else (same_vlan,)),
    )


TOPOLOGIES = {
    name: build_topology(name)
    for name in ("T2", "T3", "T4-Line", "T4-Ring", "T5-Mesh", "T6-Mesh")
}
