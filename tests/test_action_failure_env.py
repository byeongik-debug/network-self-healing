from __future__ import annotations

from env.action_failure_env import ActionFailureNetworkEnv
from env.actions import ACTIONS


def action_id(kind: str) -> int:
    return next(action.id for action in ACTIONS if action.kind == kind)


def test_probability_one_makes_configuration_action_a_noop() -> None:
    env = ActionFailureNetworkEnv(
        split="train", scenario="access_vlan_sw1", action_failure_probability=1.0
    )
    before, _ = env.reset(seed=1)
    target = next(
        action.id for action in ACTIONS
        if action.kind == "set_access_vlan"
        and action.switch == "SW1" and action.port == "eth1" and action.vlan == 10
    )
    after, _, terminated, truncated, info = env.step(target)
    assert after["ports"] == before["ports"]
    assert not terminated and not truncated
    assert info["action_failed"]
    assert env.get_metrics()["action_failures"] == 1
    assert env.get_metrics()["actual_configuration_changes"] == 0


def test_control_action_is_not_subject_to_command_failure() -> None:
    env = ActionFailureNetworkEnv(
        split="train", scenario="access_vlan_sw1", action_failure_probability=1.0
    )
    env.reset(seed=1)
    observation, *_ = env.step(action_id("diagnose"))
    assert all(
        value != "unknown"
        for row in observation["connectivity"].values()
        for value in row.values()
    )
    assert env.get_metrics()["action_failures"] == 0
