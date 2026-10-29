"""Variable-size action catalog and state-dependent validity masks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from env.expanded.environment import ExpandedAction
from env.expanded.schema import TopologySpec
from env.expanded.simulator import ExpandedNetworkSimulator


@dataclass(frozen=True)
class ActionCandidate:
    action_id: int
    action: ExpandedAction
    target_node: str | None
    secondary_node: str | None = None


def build_action_catalog(
    topology: TopologySpec, *, acl_scope: str = "all"
) -> tuple[ActionCandidate, ...]:
    if acl_scope not in ("all", "policy"):
        raise ValueError("acl_scope must be 'all' or 'policy'")
    candidates: list[ActionCandidate] = []

    def add(action: ExpandedAction, target: str | None, secondary: str | None = None) -> None:
        candidates.append(ActionCandidate(len(candidates), action, target, secondary))

    add(ExpandedAction("diagnose"), None)
    for host in topology.hosts:
        for vlan in topology.vlans:
            add(ExpandedAction("set_access_vlan", host.name, vlan=vlan), host.name)
        for enabled in (False, True):
            add(ExpandedAction("set_port", host.name, enabled=enabled), host.name)
    for link in topology.links:
        for enabled in (False, True):
            add(ExpandedAction("set_link", link.name, enabled=enabled), link.left_switch, link.right_switch)
        for switch in (link.left_switch, link.right_switch):
            for vlan in topology.vlans:
                for enabled in (False, True):
                    add(
                        ExpandedAction("set_trunk_vlan", link.name, switch, vlan, enabled),
                        switch,
                    )
            for enabled in (False, True):
                add(ExpandedAction("set_port", link.name, switch, enabled=enabled), switch)
    host_names = sorted(host.name for host in topology.hosts)
    acl_pairs = list(combinations(host_names, 2))
    if acl_scope == "policy":
        nominal = ExpandedNetworkSimulator(topology)
        required = [pair for pair in acl_pairs if nominal.expected_reachability(*pair)]
        acl_pairs = sorted(set(topology.policy.normalized_forbidden()) | set(required[:1]))
    for left, right in acl_pairs:
        for enabled in (False, True):
            add(ExpandedAction("set_acl", left, right, enabled=enabled), left, right)
    add(ExpandedAction("declare_done"), None)
    return tuple(candidates)


def validity_mask(
    simulator: ExpandedNetworkSimulator,
    catalog: tuple[ActionCandidate, ...],
) -> NDArray[np.bool_]:
    state = simulator.state
    mask = np.ones(len(catalog), dtype=np.bool_)
    for candidate in catalog:
        action = candidate.action
        valid = True
        if action.kind == "set_access_vlan":
            valid = state.access_vlans.get(action.target or "") != action.vlan
        elif action.kind == "set_link":
            valid = state.link_up.get(action.target or "") != action.enabled
        elif action.kind == "set_port":
            if action.secondary is None:
                valid = state.host_ports_up.get(action.target or "") != action.enabled
            else:
                valid = state.link_ports_up.get(
                    (action.target or "", action.secondary)
                ) != action.enabled
        elif action.kind == "set_trunk_vlan":
            present = action.vlan in state.trunk_vlans.get(
                (action.target or "", action.secondary or ""), set()
            )
            valid = present != action.enabled
        elif action.kind == "set_acl":
            pair = tuple(sorted((action.target or "", action.secondary or "")))
            currently_allowed = pair not in state.acl_denied
            valid = currently_allowed != action.enabled
        mask[candidate.action_id] = valid
    return mask
