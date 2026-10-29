"""Topology-size-independent graph observation and action feature encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from env.expanded.actions import ActionCandidate
from env.expanded.schema import TopologySpec, canonical_pair


ACTION_KINDS = (
    "diagnose", "set_access_vlan", "set_trunk_vlan", "set_link",
    "set_port", "set_acl", "declare_done",
)


@dataclass(frozen=True)
class GraphObservation:
    node_names: tuple[str, ...]
    node_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    edge_features: NDArray[np.float32]
    action_features: NDArray[np.float32]
    action_targets: NDArray[np.int64]


class GraphObservationEncoder:
    node_dim = 8
    edge_dim = 5
    base_action_dim = len(ACTION_KINDS) + 4
    action_dim = base_action_dim

    def __init__(
        self, topology: TopologySpec, catalog: tuple[ActionCandidate, ...],
        *, include_policy_features: bool = False,
    ) -> None:
        self.topology, self.catalog = topology, catalog
        self.include_policy_features = include_policy_features
        self.action_dim = self.base_action_dim + (2 if include_policy_features else 0)
        self.node_names = tuple(
            [switch.name for switch in topology.switches]
            + [host.name for host in topology.hosts]
        )
        self.node_index = {name: index for index, name in enumerate(self.node_names)}

    def encode(self, observation: dict) -> GraphObservation:
        max_vlan = max(self.topology.vlans)
        access = observation["access_vlans"]
        host_up = observation["host_ports_up"]
        nodes = np.zeros((len(self.node_names), self.node_dim), dtype=np.float32)
        for switch in self.topology.switches:
            index = self.node_index[switch.name]
            nodes[index, 0] = 1.0
            nodes[index, 4] = sum(
                host.switch == switch.name for host in self.topology.hosts
            ) / max(1, len(self.topology.hosts))
        for host in self.topology.hosts:
            index = self.node_index[host.name]
            nodes[index, 1] = 1.0
            nodes[index, 2] = float(host_up[host.name])
            nodes[index, 3] = float(access[host.name]) / max_vlan
            nodes[index, 5] = float(host.vlan) / max_vlan
        nodes[:, 6] = float(observation.get("step", 0)) / 60.0
        nodes[:, 7] = float(observation.get("telemetry_age", 0)) / 4.0

        edges: list[tuple[int, int]] = []
        edge_features: list[list[float]] = []
        for host in self.topology.hosts:
            left, right = self.node_index[host.name], self.node_index[host.switch]
            feature = [1.0, 0.0, float(host_up[host.name]), 1.0, 0.0]
            edges.extend(((left, right), (right, left)))
            edge_features.extend((feature, feature))
        for link in self.topology.links:
            left, right = self.node_index[link.left_switch], self.node_index[link.right_switch]
            endpoint_up = all(observation["link_ports_up"][f"{link.name}:{switch}"]
                              for switch in (link.left_switch, link.right_switch))
            allowed = [
                set(observation["trunk_vlans"][f"{link.name}:{switch}"])
                for switch in (link.left_switch, link.right_switch)
            ]
            fraction = len(set.intersection(*allowed)) / len(self.topology.vlans)
            feature = [0.0, 1.0, float(observation["link_up"][link.name]),
                       float(endpoint_up), fraction]
            edges.extend(((left, right), (right, left)))
            edge_features.extend((feature, feature))

        actions = np.zeros((len(self.catalog), self.action_dim), dtype=np.float32)
        targets = np.full((len(self.catalog), 2), -1, dtype=np.int64)
        denied = {canonical_pair(*pair) for pair in observation["acl_denied"]}
        for candidate in self.catalog:
            action, row = candidate.action, candidate.action_id
            actions[row, ACTION_KINDS.index(action.kind)] = 1.0
            actions[row, len(ACTION_KINDS)] = float(action.enabled) if action.enabled is not None else 0.0
            actions[row, len(ACTION_KINDS) + 1] = (
                float(action.vlan) / max_vlan if action.vlan is not None else 0.0
            )
            actions[row, len(ACTION_KINDS) + 2] = float(observation.get("pending_commands", 0)) / 8.0
            if action.kind == "set_acl" and action.target and action.secondary:
                actions[row, len(ACTION_KINDS) + 3] = float(
                    canonical_pair(action.target, action.secondary) in denied
                )
                if self.include_policy_features:
                    pair = canonical_pair(action.target, action.secondary)
                    actions[row, self.base_action_dim] = float(
                        self._expected_required(*pair)
                    )
                    actions[row, self.base_action_dim + 1] = float(
                        pair in self.topology.policy.normalized_forbidden()
                    )
            if candidate.target_node is not None:
                targets[row, 0] = self.node_index[candidate.target_node]
            if candidate.secondary_node is not None:
                targets[row, 1] = self.node_index[candidate.secondary_node]
        return GraphObservation(
            self.node_names, nodes,
            np.asarray(edges, dtype=np.int64).T,
            np.asarray(edge_features, dtype=np.float32),
            actions, targets,
        )

    def _expected_required(self, left: str, right: str) -> bool:
        hosts = {host.name: host for host in self.topology.hosts}
        pair = canonical_pair(left, right)
        if pair in self.topology.policy.normalized_forbidden():
            return False
        return (pair in self.topology.policy.normalized_required()
                or hosts[left].vlan == hosts[right].vlan)
