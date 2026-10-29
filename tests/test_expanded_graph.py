import numpy as np
import pytest
import torch

from agents.graph_dqn import GraphActionQNetwork
from env.expanded.actions import build_action_catalog, validity_mask
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.topologies import build_topology


@pytest.mark.parametrize("name", ["T2", "T3", "T4-Line", "T4-Ring"])
def test_graph_encoder_and_one_shared_network_support_variable_topologies(name):
    topology = build_topology(name)
    catalog = build_action_catalog(topology)
    env = ExpandedNetworkEnv(topology)
    observation, _ = env.reset()
    encoded = GraphObservationEncoder(topology, catalog).encode(observation)
    network = GraphActionQNetwork(8, 5, 11)
    values = network(encoded)
    assert values.shape == (len(catalog),)
    assert torch.isfinite(values).all()
    assert encoded.node_features.shape == (len(topology.switches) + len(topology.hosts), 8)
    assert not any("fault" in key for key in observation)


def test_action_catalog_is_stable_and_scales_with_topology():
    catalogs = [build_action_catalog(build_topology(name))
                for name in ("T2", "T3", "T4-Line", "T4-Ring")]
    assert [candidate.action_id for candidate in catalogs[0]] == list(range(len(catalogs[0])))
    assert len(catalogs[0]) < len(catalogs[1]) < len(catalogs[2]) < len(catalogs[3])
    assert catalogs[0] == build_action_catalog(build_topology("T2"))


def test_validity_mask_enables_repair_and_masked_selection():
    topology = build_topology("T3")
    env = ExpandedNetworkEnv(topology)
    observation, _ = env.reset((ExpandedFault("link_down", "L1"),))
    catalog = build_action_catalog(topology)
    mask = validity_mask(env.simulator, catalog)
    repair = next(candidate.action_id for candidate in catalog
                  if candidate.action.kind == "set_link"
                  and candidate.action.target == "L1" and candidate.action.enabled)
    duplicate = next(candidate.action_id for candidate in catalog
                     if candidate.action.kind == "set_link"
                     and candidate.action.target == "L1" and not candidate.action.enabled)
    assert mask[repair] and not mask[duplicate]
    encoded = GraphObservationEncoder(topology, catalog).encode(observation)
    network = GraphActionQNetwork(8, 5, 11)
    assert mask[network.select_action(encoded, mask)]
    assert all(mask[network.select_action(encoded, mask, epsilon=1.0)] for _ in range(20))


def test_graph_features_are_finite_under_delayed_telemetry():
    topology = build_topology("T4-Ring")
    env = ExpandedNetworkEnv(topology, telemetry_delay=2, stale_probability=1.0)
    observation, _ = env.reset(seed=7)
    encoded = GraphObservationEncoder(topology, build_action_catalog(topology)).encode(observation)
    assert np.isfinite(encoded.node_features).all()
    assert np.isfinite(encoded.edge_features).all()
    assert np.isfinite(encoded.action_features).all()


def test_policy_aware_action_features_identify_acl_semantics():
    topology = build_topology("T4-Ring")
    catalog = build_action_catalog(topology, acl_scope="policy")
    env = ExpandedNetworkEnv(topology)
    observation, _ = env.reset()
    encoder = GraphObservationEncoder(topology, catalog, include_policy_features=True)
    encoded = encoder.encode(observation)
    assert encoded.action_features.shape[1] == 13
    acl_rows = [candidate.action_id for candidate in catalog
                if candidate.action.kind == "set_acl"]
    semantics = {tuple(encoded.action_features[row, -2:]) for row in acl_rows}
    assert (1.0, 0.0) in semantics
    assert (0.0, 1.0) in semantics
