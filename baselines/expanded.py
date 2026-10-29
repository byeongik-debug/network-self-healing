"""Privileged reference planner for expanded-environment upper bounds."""

from __future__ import annotations

from env.expanded.environment import ExpandedAction, ExpandedNetworkEnv
from env.expanded.shield import is_action_safe


def reference_repairs(env: ExpandedNetworkEnv) -> list[ExpandedAction]:
    """Return the exact configuration diff to the topology's nominal state."""
    topology, state = env.topology, env.simulator.state
    repairs: list[ExpandedAction] = []
    for host in topology.hosts:
        if state.access_vlans[host.name] != host.vlan:
            repairs.append(ExpandedAction("set_access_vlan", host.name, vlan=host.vlan))
        if not state.host_ports_up[host.name]:
            repairs.append(ExpandedAction("set_port", host.name, enabled=True))
    for link in topology.links:
        if state.link_up[link.name] != link.nominal_enabled:
            repairs.append(ExpandedAction("set_link", link.name, enabled=link.nominal_enabled))
        for switch in (link.left_switch, link.right_switch):
            if not state.link_ports_up[(link.name, switch)]:
                repairs.append(ExpandedAction("set_port", link.name, switch, enabled=True))
            nominal = set(link.allowed_vlans)
            current = state.trunk_vlans[(link.name, switch)]
            for vlan in sorted(nominal - current):
                repairs.append(ExpandedAction("set_trunk_vlan", link.name, switch, vlan, True))
            for vlan in sorted(current - nominal):
                repairs.append(ExpandedAction("set_trunk_vlan", link.name, switch, vlan, False))
    nominal_denied = topology.policy.normalized_forbidden()
    for left, right in sorted(state.acl_denied - nominal_denied):
        repairs.append(ExpandedAction("set_acl", left, right, enabled=True))
    for left, right in sorted(nominal_denied - state.acl_denied):
        repairs.append(ExpandedAction("set_acl", left, right, enabled=False))
    return repairs


def run_reference_baseline(env: ExpandedNetworkEnv) -> dict[str, int | bool]:
    """Replan after every command so failures and delays are observable."""
    while env.metrics()["steps"] < env.max_steps:
        metrics = env.metrics()
        if metrics["policy_healthy"] and metrics["pending_commands"] == 0:
            _, _, terminated, truncated, info = env.step(ExpandedAction("declare_done"))
            return {**env.metrics(), "success": bool(info["recovery_success"]),
                    "terminated": terminated, "truncated": truncated,
                    "safety_deadlock": False}
        repairs = reference_repairs(env)
        action = repairs[0] if repairs else ExpandedAction("diagnose")
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return {**env.metrics(), "success": False, "terminated": False,
            "truncated": True, "safety_deadlock": False}


def run_shielded_reference_baseline(env: ExpandedNetworkEnv) -> dict[str, int | bool]:
    """Reference diff planner filtered by the non-worsening safety shield."""
    while env.metrics()["steps"] < env.max_steps:
        metrics = env.metrics()
        if metrics["policy_healthy"] and metrics["pending_commands"] == 0:
            _, _, terminated, truncated, info = env.step(ExpandedAction("declare_done"))
            return {**env.metrics(), "success": bool(info["recovery_success"]),
                    "terminated": terminated, "truncated": truncated,
                    "safety_deadlock": False}
        repairs = reference_repairs(env)
        safe_repairs = [action for action in repairs if is_action_safe(env.simulator, action)]
        if not safe_repairs and repairs and metrics["pending_commands"] == 0:
            return {**env.metrics(), "success": False, "terminated": False,
                    "truncated": False, "safety_deadlock": True}
        action = safe_repairs[0] if safe_repairs else ExpandedAction("diagnose")
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return {**env.metrics(), "success": False, "terminated": False,
            "truncated": True, "safety_deadlock": False}
