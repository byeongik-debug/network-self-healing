from __future__ import annotations

from env.actions import ACTIONS
from env.network_env import NetworkEnv
from env.simulator import NetworkSimulator


def action_id(kind: str, switch: str | None = None, port: str | None = None, vlan: int | None = None, allowed: bool | None = None) -> int:
    for action in ACTIONS:
        if (action.kind, action.switch, action.port, action.vlan, action.allowed) == (kind, switch, port, vlan, allowed):
            return action.id
    raise AssertionError("action not found")


def test_healthy_same_vlan_connectivity() -> None:
    simulator = NetworkSimulator()
    assert simulator.can_reach("H1", "H3")
    assert simulator.can_reach("H2", "H4")


def test_inter_vlan_isolation() -> None:
    simulator = NetworkSimulator()
    assert not simulator.can_reach("H1", "H2")
    assert not simulator.can_reach("H1", "H4")


def test_single_access_fault_occurs() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    observation, _ = env.reset(seed=1)
    assert observation["ports"]["SW1"]["eth1"]["access_vlan"] == 20
    assert not env.check_connectivity()["H1"]["H3"]


def test_single_trunk_fault_occurs() -> None:
    env = NetworkEnv(scenario="trunk_missing_sw1")
    env.reset(seed=1)
    assert not env.check_connectivity()["H1"]["H3"]


def test_combined_fault_occurs() -> None:
    env = NetworkEnv(scenario="combined_train")
    observation, _ = env.reset(seed=1)
    assert observation["ports"]["SW1"]["eth2"]["access_vlan"] == 10
    assert 20 not in observation["ports"]["SW2"]["eth3"]["allowed_vlans"]


def test_recovery_restores_connectivity() -> None:
    env = NetworkEnv(scenario="combined_train")
    env.reset(seed=1)
    env.step(action_id("set_access_vlan", "SW1", "eth2", 20, None))
    _, reward, _, _, info = env.step(action_id("set_trunk_vlan", "SW2", "eth3", 20, True))
    assert reward == 0.0
    assert info["policy_healthy"]
    assert env.check_connectivity()["H2"]["H4"]


def test_invalid_action_is_reported_without_mutation() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    before, _ = env.reset(seed=1)
    after, reward, terminated, truncated, info = env.step(999)
    assert after["ports"] == before["ports"]
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert not info["valid_action"]
    assert env.get_metrics()["invalid_actions"] == 1


def test_same_seed_reproduces_scenario_and_observation() -> None:
    first = NetworkEnv()
    second = NetworkEnv()
    observation_a, _ = first.reset(seed=2026)
    observation_b, _ = second.reset(seed=2026)
    assert observation_a == observation_b


def test_observation_does_not_leak_answer_or_fault_cause() -> None:
    env = NetworkEnv(scenario="combined_train")
    observation, _ = env.reset(seed=1)
    serialized_keys = repr(observation).lower()
    forbidden = ("reference", "expected", "fault", "scenario", "root_cause", "correct")
    assert all(term not in serialized_keys for term in forbidden)


def test_train_and_eval_scenarios_are_separate() -> None:
    env = NetworkEnv(split="eval", scenario="access_vlan_sw2")
    observation, info = env.reset(seed=1)
    assert info == {"split": "eval"}
    assert observation["ports"]["SW2"]["eth1"]["access_vlan"] == 20


def test_declare_done_reports_success_and_terminates() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    env.step(action_id("set_access_vlan", "SW1", "eth1", 10, None))
    _, _, terminated, _, info = env.step(action_id("declare_done"))
    assert terminated
    assert info["recovery_success"]

