from collections import Counter

import pytest

from baselines.expanded import (
    reference_repairs, run_reference_baseline, run_shielded_reference_baseline,
)
from env.expanded.dataset import (
    HELD_OUT_TOPOLOGIES, TRAIN_TOPOLOGIES, build_scenarios,
)
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault, atomic_faults
from env.expanded.topologies import build_topology


def test_scenario_manifest_is_deterministic_unique_and_leak_free():
    first, second = build_scenarios(), build_scenarios()
    assert first == second
    assert len({scenario.scenario_id for scenario in first}) == len(first)
    assert {s.topology for s in first if s.split == "test"} == set(HELD_OUT_TOPOLOGIES)
    assert {s.topology for s in first if s.split != "test"} == set(TRAIN_TOPOLOGIES)
    counts = Counter(s.split for s in first)
    assert counts["train"] and counts["validation"] and counts["test"]
    ring = [s for s in first if s.topology == "T4-Ring"]
    assert any(s.initially_healthy for s in ring)  # resilience/no-impact controls
    assert any(not s.initially_healthy for s in ring)


@pytest.mark.parametrize("topology_name", ["T2", "T3", "T4-Line", "T4-Ring"])
def test_reference_baseline_repairs_every_atomic_fault(topology_name):
    topology = build_topology(topology_name)
    for fault in atomic_faults(topology):
        env = ExpandedNetworkEnv(topology, max_steps=20)
        env.reset((fault,), seed=1)
        result = run_reference_baseline(env)
        assert result["success"], fault.fault_id


def test_reference_baseline_handles_failure_delay_and_stale_telemetry():
    topology = build_topology("T4-Ring")
    faults = (ExpandedFault("link_down", "L1"), ExpandedFault("port_shutdown", "H1"))
    env = ExpandedNetworkEnv(
        topology, max_steps=60, command_failure_probability=0.25,
        actuation_delay=2, telemetry_delay=2, stale_probability=0.5,
    )
    env.reset(faults, seed=11)
    result = run_reference_baseline(env)
    assert result["success"]
    assert result["action_failures"] >= 0


def test_reference_repairs_is_empty_in_nominal_state():
    env = ExpandedNetworkEnv(build_topology("T3"))
    env.reset()
    assert reference_repairs(env) == []


def test_shielded_reference_reports_unavoidable_one_step_safety_deadlock():
    topology = build_topology("T3")
    faults = (ExpandedFault("access_vlan", "H1", vlan=20),
              ExpandedFault("access_vlan", "H4", vlan=10))
    env = ExpandedNetworkEnv(topology, max_steps=30)
    env.reset(faults, seed=3)
    result = run_shielded_reference_baseline(env)
    assert not result["success"]
    assert result["safety_deadlock"]
    assert result["new_policy_damage_events"] == 0
