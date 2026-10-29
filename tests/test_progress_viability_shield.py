from __future__ import annotations

import numpy as np

from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from safety.progress_viability_shield import ProgressViabilityShield
from safety.viability_shield import canonical_configuration


def test_progress_actions_preserve_damage_and_reduce_distance() -> None:
    shield = ProgressViabilityShield()
    scenarios = FaultDataset().stratified_split("eval", 2026)
    for scenario in scenarios:
        env = NetworkEnv(split="eval", fault_scenario=scenario)
        env.reset(seed=2026)
        decision = shield.get_decision(env)
        current_key = canonical_configuration(env)
        current_steps = shield.minimum_steps[current_key]
        assert current_steps > 0
        assert not decision.combined_mask[0]  # diagnose cannot self-loop while unhealthy
        assert not decision.combined_mask[18]  # cannot declare before recovery
        for action_id in np.flatnonzero(decision.combined_mask):
            action = ACTIONS[int(action_id)]
            assert action.kind in ("set_access_vlan", "set_trunk_vlan")
            successor, immediate = shield.transition(current_key, int(action_id))
            assert immediate + shield.minimum_damage[successor] == decision.current_cost
            assert shield.minimum_steps[successor] == current_steps - 1


def test_progress_policy_terminates_for_every_eval_scenario() -> None:
    shield = ProgressViabilityShield()
    scenarios = FaultDataset().stratified_split("eval", 2026)
    for scenario in scenarios:
        env = NetworkEnv(split="eval", fault_scenario=scenario, max_steps=30)
        env.reset(seed=2026)
        initial_cost = shield.cost_to_goal(env)
        while True:
            decision = shield.get_decision(env)
            action_id = int(np.flatnonzero(decision.combined_mask)[0])
            _, _, terminated, truncated, info = env.step(action_id)
            assert not truncated
            if terminated:
                assert info["recovery_success"]
                break
        assert env.get_metrics()["new_policy_damage_events"] == initial_cost


def test_healthy_state_requires_one_diagnosis_then_declaration() -> None:
    shield = ProgressViabilityShield()
    env = NetworkEnv(split="train", scenario="access_vlan_sw1")
    env.reset(seed=2026)
    while not env.simulator.is_policy_healthy():
        decision = shield.get_decision(env)
        env.step(int(np.flatnonzero(decision.combined_mask)[0]))

    before = shield.get_decision(env)
    assert np.flatnonzero(before.combined_mask).tolist() == [0]
    env.step(0)
    after = shield.get_decision(env)
    assert np.flatnonzero(after.combined_mask).tolist() == [18]
