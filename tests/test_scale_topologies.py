import pytest
import torch

from agents.graph_dqn import GraphActionQNetwork
from baselines.expanded import reference_repairs
from env.expanded.actions import build_action_catalog
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import atomic_faults
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.topologies import build_topology


@pytest.mark.parametrize(
    "name,switches,hosts,vlans,links",
    [("T5-Mesh", 5, 20, 5, 7), ("T6-Mesh", 6, 24, 6, 9)],
)
def test_scale_topology_dimensions_and_nominal_health(name, switches, hosts, vlans, links):
    topology = build_topology(name)
    assert len(topology.switches) == switches
    assert len(topology.hosts) == hosts
    assert len(topology.vlans) == vlans
    assert len(topology.links) == links
    env = ExpandedNetworkEnv(topology)
    env.reset()
    assert env.simulator.is_policy_healthy()


def test_policy_scoped_acl_catalog_controls_quadratic_growth():
    topology = build_topology("T6-Mesh")
    full = build_action_catalog(topology)
    scoped = build_action_catalog(topology, acl_scope="policy")
    assert len(full) > 700
    assert len(scoped) < 500
    assert len(scoped) < len(full) / 2
    assert {candidate.action.kind for candidate in scoped} == {
        "diagnose", "set_access_vlan", "set_trunk_vlan", "set_link",
        "set_port", "set_acl", "declare_done",
    }


def test_existing_graph_network_runs_on_six_switch_topology():
    topology = build_topology("T6-Mesh")
    catalog = build_action_catalog(topology, acl_scope="policy")
    env = ExpandedNetworkEnv(topology)
    observation, _ = env.reset()
    encoded = GraphObservationEncoder(topology, catalog).encode(observation)
    network = GraphActionQNetwork(8, 5, 11, hidden=32)
    values = network(encoded)
    assert values.shape == (len(catalog),)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("name", ["T5-Mesh", "T6-Mesh"])
def test_policy_scoped_catalog_retains_every_atomic_reference_repair(name):
    topology = build_topology(name)
    catalog = build_action_catalog(topology, acl_scope="policy")
    available = {candidate.action for candidate in catalog}
    for fault in atomic_faults(topology):
        env = ExpandedNetworkEnv(topology)
        env.reset((fault,))
        repairs = reference_repairs(env)
        assert repairs, fault.fault_id
        assert set(repairs) <= available, fault.fault_id
