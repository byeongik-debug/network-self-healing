from __future__ import annotations

from env.actions import ACTIONS, ActionSpec
from env.network_env import NetworkEnv


def aid(kind: str, switch: str | None = None, port: str | None = None,
        vlan: int | None = None, allowed: bool | None = None) -> int:
    for action in ACTIONS:
        if (action.kind, action.switch, action.port, action.vlan, action.allowed) == (
            kind, switch, port, vlan, allowed
        ):
            return action.id
    raise AssertionError("action not found")


def test_duplicate_setting_is_masked_and_counted() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=7)
    duplicate = aid("set_access_vlan", "SW1", "eth1", 20)
    before = env.simulator.snapshot()
    assert not env.get_action_mask()[duplicate]
    _, _, _, _, info = env.step(duplicate)
    assert not info["valid_action"]
    assert info["reason"] == "setting_already_applied"
    assert env.simulator.snapshot() == before
    assert env.get_metrics()["unnecessary_actions"] == 1


def test_invalid_target_is_rejected_by_mask_validation() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=7)
    invalid = ActionSpec(99, "set_access_vlan", "SW9", "eth99", 10)
    assert env._action_availability(invalid) == (False, "target_does_not_exist")


def test_single_rollback_restores_previous_state() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    initial, _ = env.reset(seed=7)
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    rollback = aid("rollback")
    assert env.get_action_mask()[rollback]
    env.step(rollback)
    assert env.get_observation()["ports"] == initial["ports"]
    assert not env.get_action_mask()[rollback]


def test_multiple_rollbacks_are_lifo() -> None:
    env = NetworkEnv(scenario="combined_train")
    initial, _ = env.reset(seed=7)
    env.step(aid("set_access_vlan", "SW1", "eth2", 20))
    after_first = env.simulator.snapshot()
    env.step(aid("set_trunk_vlan", "SW2", "eth3", 20, True))
    env.step(aid("rollback"))
    assert env.simulator.snapshot() == after_first
    env.step(aid("rollback"))
    assert env.get_observation()["ports"] == initial["ports"]
    assert env.get_metrics()["rollbacks"] == 2


def test_change_count_excludes_duplicate_and_rollback() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=7)
    env.step(aid("set_access_vlan", "SW1", "eth1", 20))
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    env.step(aid("rollback"))
    metrics = env.get_metrics()
    assert metrics["actual_configuration_changes"] == 1
    assert metrics["unnecessary_actions"] == 1
    assert metrics["rollbacks"] == 1


def test_connectivity_is_unknown_until_diagnosed() -> None:
    env = NetworkEnv(scenario="trunk_missing_sw1")
    observation, _ = env.reset(seed=7)
    values = {value for row in observation["connectivity"].values() for value in row.values()}
    assert values == {"unknown"}


def test_diagnosis_updates_observation() -> None:
    env = NetworkEnv(scenario="trunk_missing_sw1")
    env.reset(seed=7)
    observation, *_ = env.step(aid("diagnose"))
    assert observation["connectivity"]["H1"]["H3"] == "failure"
    assert observation["connectivity"]["H2"]["H4"] == "success"
    assert env.get_metrics()["diagnostic_actions"] == 1


def test_change_invalidates_stale_diagnostics() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=7)
    env.step(aid("diagnose"))
    observation, *_ = env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    values = {value for row in observation["connectivity"].values() for value in row.values()}
    assert values == {"unknown"}


def test_harm_to_previously_correct_policy_is_counted() -> None:
    env = NetworkEnv(scenario="trunk_missing_sw1")
    env.reset(seed=7)
    _, _, _, _, info = env.step(aid("set_access_vlan", "SW2", "eth2", 10))
    assert info["healthy_network_impact"]
    assert env.get_metrics()["healthy_network_impacts"] == 1


def test_same_seed_reproduces_mask_and_unknown_observation() -> None:
    first, second = NetworkEnv(), NetworkEnv()
    observation_a, _ = first.reset(seed=2026)
    observation_b, _ = second.reset(seed=2026)
    assert observation_a == observation_b
    assert first.get_action_mask() == second.get_action_mask()


def test_initial_restore_is_distinct_and_clears_history() -> None:
    env = NetworkEnv(scenario="combined_train")
    initial, _ = env.reset(seed=7)
    env.step(aid("set_access_vlan", "SW1", "eth2", 20))
    env.step(aid("set_trunk_vlan", "SW2", "eth3", 20, True))
    assert env.restore_initial_state()
    assert env.get_observation()["ports"] == initial["ports"]
    assert env.get_metrics()["rollback_depth"] == 0


def test_empty_rollback_is_masked() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=7)
    _, _, _, _, info = env.step(aid("rollback"))
    assert not info["valid_action"]
    assert info["reason"] == "no_change_to_rollback"
    assert env.get_metrics()["rollbacks"] == 0
