import numpy as np
import pytest

from env.expanded.actions import build_action_catalog, validity_mask
from env.expanded.environment import ExpandedAction, ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault, atomic_faults
from env.expanded.shield import (
    SafetyMaskCache, counterfactual_safety_mask, environment_safety_mask, is_action_safe,
    policy_violation_count,
)
from env.expanded.topologies import build_topology


@pytest.mark.parametrize("name", ["T2", "T3", "T4-Line", "T4-Ring"])
def test_shield_is_non_mutating_and_never_all_false(name):
    topology = build_topology(name)
    env = ExpandedNetworkEnv(topology)
    env.reset((atomic_faults(topology)[0],))
    before = env.simulator.observation()
    mask = counterfactual_safety_mask(env.simulator, build_action_catalog(topology))
    assert mask.any()
    assert env.simulator.observation() == before


def test_shield_blocks_new_damage_and_false_declaration():
    topology = build_topology("T3")
    env = ExpandedNetworkEnv(topology)
    env.reset((ExpandedFault("port_shutdown", "H1"),))
    catalog = build_action_catalog(topology)
    mask = counterfactual_safety_mask(env.simulator, catalog)
    declare = next(c.action_id for c in catalog if c.action.kind == "declare_done")
    harmful = next(c.action_id for c in catalog if c.action.kind == "set_link"
                   and c.action.target == "L1" and not c.action.enabled)
    repair = next(c.action_id for c in catalog if c.action.kind == "set_port"
                  and c.action.target == "H1" and c.action.enabled)
    assert not mask[declare]
    assert not mask[harmful]
    assert mask[repair]


def test_progress_mask_contains_strict_repair_for_each_atomic_fault():
    topology = build_topology("T4-Ring")
    catalog = build_action_catalog(topology)
    checked = 0
    for fault in atomic_faults(topology):
        env = ExpandedNetworkEnv(topology)
        env.reset((fault,))
        before = policy_violation_count(env.simulator)
        if before == 0:  # redundant ring paths make some link faults no-impact controls
            continue
        checked += 1
        mask = counterfactual_safety_mask(env.simulator, catalog, require_progress=True)
        progress_exists = False
        for candidate in np.asarray(catalog, dtype=object)[mask]:
            if candidate.action.kind not in ("diagnose", "declare_done"):
                progress_exists = True
                break
        assert progress_exists, fault.fault_id
    assert checked > 0


def test_pending_aware_shield_serializes_delayed_commands():
    topology = build_topology("T3")
    catalog = build_action_catalog(topology)
    env = ExpandedNetworkEnv(topology, actuation_delay=2)
    env.reset((ExpandedFault("port_shutdown", "H1"),))
    repair = next(c for c in catalog if c.action.kind == "set_port"
                  and c.action.target == "H1" and c.action.enabled)
    env.step(repair.action)
    mask = environment_safety_mask(env, catalog)
    allowed_kinds = {candidate.action.kind for candidate in np.asarray(catalog, dtype=object)[mask]}
    assert allowed_kinds == {"diagnose"}


def test_observed_shield_uses_same_delayed_state_as_policy():
    topology = build_topology("T2")
    catalog = build_action_catalog(topology)
    env = ExpandedNetworkEnv(topology, telemetry_delay=1)
    env.reset((ExpandedFault("access_vlan", "H1", vlan=20),), seed=1)
    env.step(ExpandedAction("set_access_vlan", "H1", vlan=10))
    assert env.simulator.is_policy_healthy()
    assert not env.shield_simulator("observed").is_policy_healthy()
    declare = next(c.action_id for c in catalog if c.action.kind == "declare_done")
    assert environment_safety_mask(env, catalog, shield_view="true")[declare]
    assert not environment_safety_mask(env, catalog, shield_view="observed")[declare]


def test_shield_view_default_is_true_and_invalid_view_is_rejected():
    topology = build_topology("T2")
    catalog = build_action_catalog(topology)
    env = ExpandedNetworkEnv(topology)
    env.reset((ExpandedFault("access_vlan", "H1", vlan=20),), seed=1)
    assert np.array_equal(
        environment_safety_mask(env, catalog),
        environment_safety_mask(env, catalog, shield_view="true"),
    )
    with pytest.raises(ValueError, match="shield_view"):
        SafetyMaskCache(catalog, shield_view="invalid")


def test_safety_mask_cache_reuses_diagnostic_self_loop_state():
    topology = build_topology("T3")
    catalog = build_action_catalog(topology)
    env = ExpandedNetworkEnv(topology)
    env.reset((ExpandedFault("port_shutdown", "H1"),))
    cache = SafetyMaskCache(catalog)
    first = cache.mask(env)
    env.step(next(c.action for c in catalog if c.action.kind == "diagnose"))
    second = cache.mask(env)
    assert np.array_equal(first, second)
    assert cache.misses == 1 and cache.hits == 1


@pytest.mark.parametrize("name", ["T2", "T3", "T4-Line", "T4-Ring", "T5-Mesh", "T6-Mesh"])
@pytest.mark.parametrize("require_progress", [False, True])
def test_optimized_mask_matches_independent_clone_reference(name, require_progress):
    """Guard the fast path against the simple clone-per-action definition."""
    topology = build_topology(name)
    catalog = build_action_catalog(topology)
    faults = atomic_faults(topology)
    samples = [(), (faults[0],), (faults[len(faults) // 2],), (faults[-1],)]
    for scenario in samples:
        env = ExpandedNetworkEnv(topology)
        env.reset(scenario)
        before = env.simulator.observation()
        actual = counterfactual_safety_mask(
            env.simulator, catalog, require_progress=require_progress
        )
        valid = validity_mask(env.simulator, catalog)
        expected = np.array([
            is_action_safe(env.simulator, candidate.action, require_progress=require_progress)
            and bool(valid[candidate.action_id])
            for candidate in catalog
        ], dtype=np.bool_)
        assert np.array_equal(actual, expected), (name, scenario, require_progress)
        assert env.simulator.observation() == before
