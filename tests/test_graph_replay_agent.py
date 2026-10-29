import numpy as np
import torch

from agents.graph_dqn import GraphDQNAgent
from agents.graph_replay_buffer import GraphReplayBuffer
from env.expanded.actions import build_action_catalog, validity_mask
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.topologies import build_topology


def _transition(topology_name: str):
    topology = build_topology(topology_name)
    env = ExpandedNetworkEnv(topology)
    observation, _ = env.reset()
    catalog = build_action_catalog(topology)
    encoder = GraphObservationEncoder(topology, catalog)
    state = encoder.encode(observation)
    mask = validity_mask(env.simulator, catalog)
    action = int(np.flatnonzero(mask)[0])
    next_observation, reward, terminated, truncated, _ = env.step(catalog[action].action)
    return state, action, reward, encoder.encode(next_observation), terminated, truncated, validity_mask(env.simulator, catalog)


def test_graph_replay_copies_variable_size_transitions():
    replay = GraphReplayBuffer(10)
    for name in ("T2", "T4-Ring"):
        replay.push(*_transition(name))
    samples = replay.sample(2)
    assert len(samples) == 2
    assert len({sample.state.node_features.shape[0] for sample in samples}) == 2


def test_graph_agent_updates_on_mixed_topology_batch():
    torch.manual_seed(1)
    replay = GraphReplayBuffer(10)
    for name in ("T2", "T3", "T4-Line", "T4-Ring"):
        replay.push(*_transition(name))
    agent = GraphDQNAgent(8, 5, 11, hidden=32)
    before = [parameter.detach().clone() for parameter in agent.online_network.parameters()]
    loss = agent.update(replay, 4)
    assert np.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.online_network.parameters()))
