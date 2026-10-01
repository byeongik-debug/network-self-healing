from __future__ import annotations

import inspect

import pytest

from baselines.config_diff import ConfigurationDiffRecovery
from baselines.rule_based import RuleBasedRecovery
from env.network_env import NetworkEnv


@pytest.mark.parametrize("algorithm", [RuleBasedRecovery, ConfigurationDiffRecovery])
@pytest.mark.parametrize("scenario", ["access_vlan_sw1", "trunk_missing_sw1", "combined_train"])
def test_train_scenarios_are_recovered(algorithm: type, scenario: str) -> None:
    env = NetworkEnv(scenario=scenario)
    env.reset(seed=11)
    result = algorithm().recover(env)
    assert result.success
    assert env.get_metrics()["policy_healthy"]
    assert result.steps <= 30


@pytest.mark.parametrize("algorithm", [RuleBasedRecovery, ConfigurationDiffRecovery])
@pytest.mark.parametrize("scenario", ["access_vlan_sw2", "trunk_missing_sw2", "combined_eval"])
def test_eval_scenarios_are_recovered_without_hardcoded_scenario(algorithm: type, scenario: str) -> None:
    env = NetworkEnv(split="eval", scenario=scenario)
    env.reset(seed=11)
    assert algorithm().recover(env).success


@pytest.mark.parametrize("algorithm", [RuleBasedRecovery, ConfigurationDiffRecovery])
def test_healthy_network_avoids_unnecessary_changes(algorithm: type) -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=11)
    env.simulator.reset()  # Test fixture creates a healthy state; algorithm still sees only public observation.
    result = algorithm().recover(env)
    assert result.success
    assert env.get_metrics()["actual_configuration_changes"] == 0
    assert env.get_metrics()["unnecessary_actions"] == 0


def test_rule_based_has_no_reference_or_scenario_imports() -> None:
    source = inspect.getsource(inspect.getmodule(RuleBasedRecovery))
    assert "REFERENCE_PORTS" not in source
    assert "config.topology" not in source
    assert "config.scenarios" not in source
    assert RuleBasedRecovery.information_access == "observation_and_diagnostics_only"
    assert ConfigurationDiffRecovery.information_access == "observation_plus_reference_configuration"


def test_max_step_limit_is_enforced() -> None:
    env = NetworkEnv(scenario="combined_train", max_steps=1)
    env.reset(seed=11)
    result = RuleBasedRecovery().recover(env, max_steps=30)
    assert result.steps == 1
    assert result.truncated
    assert not result.success
