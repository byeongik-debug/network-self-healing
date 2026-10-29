from __future__ import annotations

from env.expanded.environment import ExpandedAction, ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault, atomic_faults
from env.expanded.simulator import ExpandedNetworkSimulator
from env.expanded.topologies import TOPOLOGIES
from env.simulator import NetworkSimulator


def test_topology_suite_has_frozen_scale_and_distinct_generalization_target() -> None:
    expected = {
        "T2": (2, 4, 2, 1),
        "T3": (3, 8, 3, 3),
        "T4-Line": (4, 16, 4, 3),
        "T4-Ring": (4, 16, 4, 4),
    }
    for name, counts in expected.items():
        topology = TOPOLOGIES[name]
        assert (len(topology.switches), len(topology.hosts), len(topology.vlans), len(topology.links)) == counts
        assert ExpandedNetworkSimulator(topology).is_policy_healthy()
    assert TOPOLOGIES["T4-Line"].links != TOPOLOGIES["T4-Ring"].links


def test_expanded_t2_matches_legacy_reference_policy() -> None:
    legacy = NetworkSimulator().policy_results()
    expanded = ExpandedNetworkSimulator(TOPOLOGIES["T2"]).policy_results()
    renamed = {(left.upper(), right.upper()): value for (left, right), value in expanded.items()}
    assert renamed == legacy


def test_all_requested_fault_families_are_generated() -> None:
    kinds = {fault.kind for fault in atomic_faults(TOPOLOGIES["T4-Ring"])}
    assert kinds == {
        "access_vlan", "trunk_remove", "link_down", "port_shutdown",
        "acl_deny_required", "acl_allow_forbidden",
    }


def test_representative_faults_change_policy_and_are_repairable() -> None:
    topology = TOPOLOGIES["T3"]
    cases = [
        (ExpandedFault("access_vlan", "H1", vlan=20),
         ExpandedAction("set_access_vlan", "H1", vlan=10)),
        (ExpandedFault("trunk_remove", "L1", "SW1", 10),
         ExpandedAction("set_trunk_vlan", "L1", "SW1", 10, True)),
        (ExpandedFault("link_down", "L1"),
         ExpandedAction("set_link", "L1", enabled=True)),
        (ExpandedFault("port_shutdown", "H1"),
         ExpandedAction("set_port", "H1", enabled=True)),
    ]
    for fault, repair in cases:
        env = ExpandedNetworkEnv(topology)
        env.reset((fault,), seed=1)
        assert not env.simulator.is_policy_healthy()
        env.step(repair)
        assert env.simulator.is_policy_healthy()


def test_acl_deny_and_exposure_faults_are_observable_and_repairable() -> None:
    topology = TOPOLOGIES["T3"]
    required = next(fault for fault in atomic_faults(topology) if fault.kind == "acl_deny_required")
    forbidden = next(fault for fault in atomic_faults(topology) if fault.kind == "acl_allow_forbidden")
    for fault, allowed in ((required, True), (forbidden, False)):
        env = ExpandedNetworkEnv(topology)
        env.reset((fault,), seed=1)
        assert not env.simulator.is_policy_healthy()
        env.step(ExpandedAction("set_acl", fault.target, fault.secondary, enabled=allowed))
        assert env.simulator.is_policy_healthy()


def test_command_failure_is_noop_and_counted() -> None:
    env = ExpandedNetworkEnv(TOPOLOGIES["T2"], command_failure_probability=1.0)
    fault = ExpandedFault("access_vlan", "H1", vlan=20)
    before, _ = env.reset((fault,), seed=1)
    after, _, _, _, info = env.step(ExpandedAction("set_access_vlan", "H1", vlan=10))
    assert info["action_failed"]
    assert before["access_vlans"] == after["access_vlans"]
    assert env.metrics()["action_failures"] == 1


def test_delayed_actuation_and_telemetry_expose_pending_state() -> None:
    env = ExpandedNetworkEnv(
        TOPOLOGIES["T2"], actuation_delay=2, telemetry_delay=1, stale_probability=0.0
    )
    env.reset((ExpandedFault("access_vlan", "H1", vlan=20),), seed=1)
    queued, _, _, _, info = env.step(ExpandedAction("set_access_vlan", "H1", vlan=10))
    assert info["action_queued"] and queued["pending_commands"] == 1
    assert queued["access_vlans"]["H1"] == 20
    env.step(ExpandedAction("diagnose"))
    observed, _, _, _, info = env.step(ExpandedAction("diagnose"))
    assert info["applied_pending"] == 1
    assert observed["access_vlans"]["H1"] == 20  # one-step delayed telemetry
    next_observed, *_ = env.step(ExpandedAction("diagnose"))
    assert next_observed["access_vlans"]["H1"] == 10


def test_stale_probability_reports_extra_telemetry_age() -> None:
    env = ExpandedNetworkEnv(TOPOLOGIES["T2"], telemetry_delay=1, stale_probability=1.0)
    observation, _ = env.reset(seed=7)
    assert observation["telemetry_age"] == 2
