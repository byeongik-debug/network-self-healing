from __future__ import annotations

import csv

from baselines.config_diff import ConfigurationDiffRecovery
from baselines.rule_based import RuleBasedRecovery
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from experiments.evaluation import configuration_recovery, functional_recovery
from experiments.extended_runner import ExtendedExperimentRunner, save_extended_csv


def test_functional_and_configuration_recovery_are_distinct() -> None:
    dataset = FaultDataset()
    scenario = next(item for item in dataset.all() if not item.functional_failure)
    env = NetworkEnv(fault_scenario=scenario)
    env.reset(seed=0)
    assert functional_recovery(env.simulator)
    assert not configuration_recovery(env.simulator.snapshot())


def test_configuration_diff_recovers_every_generated_combination() -> None:
    dataset = FaultDataset()
    for scenario in dataset.all():
        env = NetworkEnv(fault_scenario=scenario)
        env.reset(seed=0)
        result = ConfigurationDiffRecovery().recover(env)
        assert result.success, scenario.scenario_id
        assert functional_recovery(env.simulator)
        assert configuration_recovery(env.simulator.snapshot())


def test_existing_baselines_accept_generated_scenarios() -> None:
    scenarios = FaultDataset().sample("eval", seed=17, count=10)
    for algorithm in (RuleBasedRecovery, ConfigurationDiffRecovery):
        for scenario in scenarios:
            env = NetworkEnv(split="eval", fault_scenario=scenario)
            env.reset(seed=17)
            result = algorithm().recover(env)
            assert result.steps <= 30


def test_extended_results_include_type_severity_and_failure_reason(tmp_path) -> None:
    rows = ExtendedExperimentRunner().run(algorithms=(RuleBasedRecovery,))
    assert {row.row_type for row in rows} == {
        "episode", "fault_type_mean", "severity_mean", "overall_mean"
    }
    path = save_extended_csv(rows, tmp_path / "extended.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved
    assert {"fault_type", "severity", "functional_recovery_rate", "configuration_recovery_rate", "failure_reason"} <= set(saved[0])


def test_configuration_evaluator_is_not_part_of_rule_based_module() -> None:
    assert "configuration_recovery" not in RuleBasedRecovery.recover.__globals__
    assert "REFERENCE_PORTS" not in RuleBasedRecovery.recover.__globals__
