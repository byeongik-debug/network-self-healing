from __future__ import annotations

from copy import deepcopy

import numpy as np

from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from safety.viability_shield import (
    MinimumDamageViabilityShield,
    canonical_configuration,
    safety_cost,
    simulator_from_key,
)


def test_canonical_state_is_deterministic_hashable_and_has_256_states() -> None:
    shield = MinimumDamageViabilityShield()
    assert len(shield.states) == len(set(shield.states)) == 256
    simulator = simulator_from_key(shield.states[37])
    assert canonical_configuration(simulator) == shield.states[37]
    assert hash(canonical_configuration(simulator))


def test_functionally_equivalent_nonreference_state_is_a_goal() -> None:
    scenario = next(
        item for item in FaultDataset().all()
        if not item.functional_failure and item.fault_count > 0
    )
    env = NetworkEnv(fault_scenario=scenario)
    env.reset(seed=1)
    shield = MinimumDamageViabilityShield()
    assert env.simulator.is_policy_healthy()
    assert shield.cost_to_goal(env) == 0


def test_safety_cost_matches_environment_step_local_damage() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    env.step(1)  # restore SW1 eth1 to VLAN 10
    before = deepcopy(env.simulator)
    harmful = next(
        action for action in ACTIONS
        if (action.kind, action.switch, action.port, action.vlan)
        == ("set_access_vlan", "SW2", "eth2", 10)
    )
    after = MinimumDamageViabilityShield._apply_configuration(before, harmful)
    previous = env.get_metrics()["new_policy_damage_events"]
    env.step(harmful.id)
    delta = env.get_metrics()["new_policy_damage_events"] - previous
    assert safety_cost(before, after) == delta == 3


def test_decision_does_not_mutate_environment_and_respects_validity() -> None:
    env = NetworkEnv(scenario="combined_train")
    env.reset(seed=5)
    before = {
        "configuration": env.simulator.snapshot(),
        "history": deepcopy(env._change_history),
        "diagnostics": deepcopy(env._observed_diagnostics),
        "metrics": deepcopy(env._metrics),
        "terminated": env._terminated,
        "truncated": env._truncated,
        "rng": env._rng.getstate(),
    }
    decision = MinimumDamageViabilityShield().get_decision(env)
    after = {
        "configuration": env.simulator.snapshot(),
        "history": deepcopy(env._change_history),
        "diagnostics": deepcopy(env._observed_diagnostics),
        "metrics": deepcopy(env._metrics),
        "terminated": env._terminated,
        "truncated": env._truncated,
        "rng": env._rng.getstate(),
    }
    assert after == before
    assert decision.combined_mask.shape == (19,)
    assert decision.combined_mask.dtype == np.bool_
    assert np.all(~decision.combined_mask | decision.validity_mask)


def test_every_admissible_state_change_satisfies_bellman_equality() -> None:
    shield = MinimumDamageViabilityShield()
    for key in shield.states:
        current = shield.minimum_damage[key]
        for action in shield.configuration_actions:
            successor, immediate = shield.transition(key, action.id)
            admissible = immediate + shield.minimum_damage[successor] == current
            if admissible:
                assert immediate + shield.minimum_damage[successor] == current


def test_control_actions_remain_available_and_rollback_is_evaluated() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=1)
    shield = MinimumDamageViabilityShield()
    first = shield.get_decision(env)
    assert first.combined_mask[0]  # diagnose
    assert first.combined_mask[18]  # declare_done
    assert not first.validity_mask[17]  # no rollback history

    env.step(1)
    second = shield.get_decision(env)
    rollback = second.candidates[17]
    assert rollback.valid
    assert rollback.immediate_damage is not None
    assert rollback.successor_cost is not None


def test_exhaustive_feasibility_reproduces_phase_4_1_counts() -> None:
    shield = MinimumDamageViabilityShield()
    dataset = FaultDataset()
    all_costs = []
    for scenario in dataset.all():
        env = NetworkEnv(fault_scenario=scenario)
        env.reset(seed=2026)
        all_costs.append(shield.cost_to_goal(env))
    eval_costs = []
    for scenario in dataset.stratified_split("eval", 2026):
        env = NetworkEnv(split="eval", fault_scenario=scenario)
        env.reset(seed=2026)
        eval_costs.append(shield.cost_to_goal(env))
    assert sum(cost == 0 for cost in all_costs) == 215
    assert sum(cost > 0 for cost in all_costs) == 40
    assert sum(cost == 0 for cost in eval_costs) == 42
    assert sum(cost > 0 for cost in eval_costs) == 9
    assert sum(eval_costs) == 9
