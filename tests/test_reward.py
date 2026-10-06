from __future__ import annotations

import pytest

from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv, RewardConfig


def aid(
    kind: str,
    switch: str | None = None,
    port: str | None = None,
    vlan: int | None = None,
    allowed: bool | None = None,
) -> int:
    return next(
        action.id
        for action in ACTIONS
        if (action.kind, action.switch, action.port, action.vlan, action.allowed)
        == (kind, switch, port, vlan, allowed)
    )


def test_diagnosis_has_step_and_diagnosis_cost() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    _, reward, _, _, info = env.step(aid("diagnose"))
    assert reward == pytest.approx(-0.15)
    assert info["reward_terms"] == {
        "step": -0.05,
        "state_change": 0.0,
        "diagnosis": -0.10,
        "invalid_action": 0.0,
        "newly_damaged_policy": 0.0,
        "terminal": 0.0,
    }


def test_successful_and_incorrect_declare_done_rewards() -> None:
    failed = NetworkEnv(scenario="access_vlan_sw1")
    failed.reset(seed=1)
    _, failed_reward, terminated, _, _ = failed.step(aid("declare_done"))
    assert terminated
    assert failed_reward == pytest.approx(-10.05)

    recovered = NetworkEnv(scenario="access_vlan_sw1")
    recovered.reset(seed=1)
    recovered.step(aid("set_access_vlan", "SW1", "eth1", 10))
    _, success_reward, terminated, _, _ = recovered.step(aid("declare_done"))
    assert terminated
    assert success_reward == pytest.approx(9.95)


def test_new_damage_penalty_is_step_local_and_pair_based() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))

    _, reward, _, _, info = env.step(
        aid("set_access_vlan", "SW2", "eth2", 10)
    )
    assert info["newly_damaged_policies"] == 3
    assert info["reward_terms"]["newly_damaged_policy"] == pytest.approx(-1.5)
    assert reward == pytest.approx(-1.8)

    # Existing damage is not charged again during a later diagnostic action.
    _, next_reward, _, _, next_info = env.step(aid("diagnose"))
    assert next_info["reward_terms"]["newly_damaged_policy"] == 0.0
    assert next_reward == pytest.approx(-0.15)


def test_rollback_is_a_state_change_and_can_receive_damage_penalty() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    _, reward, _, _, info = env.step(aid("rollback"))
    assert info["state_changed"]
    assert info["newly_damaged_policies"] == 3
    assert reward == pytest.approx(-1.8)


def test_truncation_adds_no_special_penalty() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1", max_steps=1)
    env.reset(seed=1)
    _, reward, terminated, truncated, _ = env.step(aid("diagnose"))
    assert not terminated and truncated
    assert reward == pytest.approx(-0.15)


def test_configuration_only_fault_allows_immediate_successful_done() -> None:
    scenario = next(
        item
        for item in FaultDataset().all()
        if not item.functional_failure and item.fault_count > 0
    )
    env = NetworkEnv(fault_scenario=scenario)
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(aid("declare_done"))
    assert terminated and not truncated
    assert info["recovery_success"]
    assert reward == pytest.approx(9.95)


def test_reward_coefficients_are_configurable_for_ablation() -> None:
    config = RewardConfig(step=0.0, diagnosis=-1.0)
    env = NetworkEnv(scenario="access_vlan_sw1", reward_config=config)
    env.reset(seed=1)
    _, reward, _, _, _ = env.step(aid("diagnose"))
    assert reward == -1.0
