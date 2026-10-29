"""Scalable one-step counterfactual safety shield for variable topologies."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from env.expanded.actions import ActionCandidate, validity_mask
from env.expanded.environment import ExpandedAction
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.simulator import ExpandedNetworkSimulator


def _apply(simulator: ExpandedNetworkSimulator, action: ExpandedAction) -> bool:
    if action.kind == "set_access_vlan":
        return simulator.set_access_vlan(action.target or "", action.vlan or -1)
    if action.kind == "set_trunk_vlan":
        return simulator.set_trunk_vlan(
            action.target or "", action.secondary or "", action.vlan or -1,
            bool(action.enabled),
        )
    if action.kind == "set_link":
        return simulator.set_link(action.target or "", bool(action.enabled))
    if action.kind == "set_port":
        return (simulator.set_host_port(action.target or "", bool(action.enabled))
                if action.secondary is None else simulator.set_link_port(
                    action.target or "", action.secondary, bool(action.enabled)))
    if action.kind == "set_acl":
        return simulator.set_acl(
            action.target or "", action.secondary or "", bool(action.enabled)
        )
    return action.kind in ("diagnose", "declare_done")


def policy_violation_count(simulator: ExpandedNetworkSimulator) -> int:
    return sum(not healthy for healthy in simulator.policy_results().values())


def _policy_violations(simulator: ExpandedNetworkSimulator) -> frozenset[tuple[str, str]]:
    return frozenset(pair for pair, healthy in simulator.policy_results().items() if not healthy)


def _affected_pairs(
    simulator: ExpandedNetworkSimulator, action: ExpandedAction
) -> frozenset[tuple[str, str]]:
    """Return policy pairs whose reachability can change after ``action``.

    Host-local changes can only affect pairs containing that host, while an ACL
    edit affects exactly its target pair.  Fabric changes may alter arbitrary
    paths, so they deliberately retain the full-pair check.
    """
    # Build pair identities only.  Calling policy_results() here would perform
    # a full reachability sweep once per candidate and erase the fast-path win.
    names = sorted(host.name for host in simulator.topology.hosts)
    pairs = frozenset(combinations(names, 2))
    if action.kind == "set_acl":
        return frozenset({tuple(sorted((action.target or "", action.secondary or "")))})
    if action.kind in ("set_access_vlan", "set_port") and action.secondary is None:
        host = action.target
        return frozenset(pair for pair in pairs if host in pair)
    if action.kind in ("set_link", "set_trunk_vlan", "set_port"):
        # Fabric edits cannot make differently tagged access hosts reachable.
        # A trunk-VLAN edit is narrower still: only that VLAN can change.
        access = simulator.state.access_vlans
        return frozenset(
            pair for pair in pairs
            if access[pair[0]] == access[pair[1]]
            and (action.kind != "set_trunk_vlan" or access[pair[0]] == action.vlan)
        )
    return pairs


def _undo_token(simulator: ExpandedNetworkSimulator, action: ExpandedAction) -> object:
    """Capture the single state cell changed by an already-valid action."""
    state = simulator.state
    if action.kind == "set_access_vlan":
        return state.access_vlans[action.target or ""]
    if action.kind == "set_link":
        return state.link_up[action.target or ""]
    if action.kind == "set_port":
        return (state.host_ports_up[action.target or ""] if action.secondary is None
                else state.link_ports_up[(action.target or "", action.secondary)])
    if action.kind == "set_trunk_vlan":
        return action.vlan in state.trunk_vlans[(action.target or "", action.secondary or "")]
    if action.kind == "set_acl":
        pair = tuple(sorted((action.target or "", action.secondary or "")))
        return pair not in state.acl_denied
    return None


def _restore_action(
    simulator: ExpandedNetworkSimulator, action: ExpandedAction, token: object
) -> None:
    """Restore a state cell without allocating a simulator or deep-copying state."""
    if action.kind == "set_access_vlan":
        simulator.set_access_vlan(action.target or "", int(token))
    elif action.kind == "set_link":
        simulator.set_link(action.target or "", bool(token))
    elif action.kind == "set_port":
        (simulator.set_host_port(action.target or "", bool(token))
         if action.secondary is None else simulator.set_link_port(
             action.target or "", action.secondary, bool(token)))
    elif action.kind == "set_trunk_vlan":
        simulator.set_trunk_vlan(
            action.target or "", action.secondary or "", action.vlan or -1, bool(token)
        )
    elif action.kind == "set_acl":
        simulator.set_acl(action.target or "", action.secondary or "", bool(token))


def is_action_safe(
    simulator: ExpandedNetworkSimulator,
    action: ExpandedAction,
    *,
    require_progress: bool = False,
) -> bool:
    """Check one action without mutating the supplied simulator."""
    before = _policy_violations(simulator)
    if action.kind == "diagnose":
        return True
    if action.kind == "declare_done":
        return not before
    clone = ExpandedNetworkSimulator(simulator.topology)
    clone.restore(simulator.snapshot())
    if not _apply(clone, action):
        return False
    after = _policy_violations(clone)
    return after < before if require_progress else after <= before


def counterfactual_safety_mask(
    simulator: ExpandedNetworkSimulator,
    catalog: tuple[ActionCandidate, ...],
    *,
    require_progress: bool = False,
) -> NDArray[np.bool_]:
    """Mask actions that increase violations; optionally require strict progress.

    Diagnosis remains available for liveness. Declaration is available only once
    the policy is healthy, preventing false-success termination.
    """
    mask = validity_mask(simulator, catalog)
    before = _policy_violations(simulator)
    for candidate in catalog:
        if not mask[candidate.action_id]:
            continue
        action = candidate.action
        if action.kind == "diagnose":
            continue
        if action.kind == "declare_done":
            mask[candidate.action_id] = not before
            continue

        affected = _affected_pairs(simulator, action)
        token = _undo_token(simulator, action)
        try:
            if not _apply(simulator, action):
                mask[candidate.action_id] = False
                continue
            after_affected = frozenset(
                pair for pair in affected
                if simulator.can_reach(*pair) != simulator.expected_reachability(*pair)
            )
            after = (before - affected) | after_affected
            mask[candidate.action_id] = (
                after < before if require_progress else after <= before
            )
        finally:
            _restore_action(simulator, action, token)
    return mask


def environment_safety_mask(
    env: ExpandedNetworkEnv,
    catalog: tuple[ActionCandidate, ...],
    *,
    require_progress: bool = False,
    shield_view: str = "true",
) -> NDArray[np.bool_]:
    """Pending-aware shield that serializes delayed configuration changes."""
    mask = counterfactual_safety_mask(
        env.shield_simulator(shield_view), catalog, require_progress=require_progress
    )
    if env.metrics()["pending_commands"]:
        for candidate in catalog:
            mask[candidate.action_id] = candidate.action.kind == "diagnose"
    return mask


class SafetyMaskCache:
    """Episode-local cache for repeated configurations and diagnostic self-loops."""

    def __init__(
        self, catalog: tuple[ActionCandidate, ...], *, shield_view: str = "true"
    ) -> None:
        if shield_view not in ("true", "observed"):
            raise ValueError("shield_view must be 'true' or 'observed'")
        self.catalog = catalog
        self.shield_view = shield_view
        self._cache: dict[tuple, NDArray[np.bool_]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(
        env: ExpandedNetworkEnv, require_progress: bool, shield_view: str = "true"
    ) -> tuple:
        state = env.shield_simulator(shield_view).state
        return (
            tuple(sorted(state.access_vlans.items())),
            tuple(sorted(state.host_ports_up.items())),
            tuple(sorted(state.link_up.items())),
            tuple(sorted(state.link_ports_up.items())),
            tuple(sorted((key, tuple(sorted(value))) for key, value in state.trunk_vlans.items())),
            tuple(sorted(state.acl_denied)),
            env.metrics()["pending_commands"],
            require_progress,
            shield_view,
        )

    def mask(
        self, env: ExpandedNetworkEnv, *, require_progress: bool = False
    ) -> NDArray[np.bool_]:
        key = self._key(env, require_progress, self.shield_view)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached.copy()
        self.misses += 1
        value = environment_safety_mask(
            env, self.catalog, require_progress=require_progress,
            shield_view=self.shield_view,
        )
        self._cache[key] = value.copy()
        return value
