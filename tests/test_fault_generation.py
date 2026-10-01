from __future__ import annotations

from env.faults import FaultDataset, atomic_faults
from env.network_env import NetworkEnv


def test_atomic_fault_space_and_unique_combination_count() -> None:
    dataset = FaultDataset()
    assert len(atomic_faults()) == 8
    assert len(dataset.all()) == 255
    assert len({scenario.scenario_id for scenario in dataset.all()}) == 255


def test_generation_sampling_is_seed_reproducible() -> None:
    dataset = FaultDataset()
    first = dataset.sample("train", seed=2026, count=12)
    second = dataset.sample("train", seed=2026, count=12)
    assert first == second


def test_train_and_eval_combinations_do_not_overlap() -> None:
    dataset = FaultDataset()
    train = {scenario.scenario_id for scenario in dataset.split("train")}
    evaluation = {scenario.scenario_id for scenario in dataset.split("eval")}
    assert train
    assert evaluation
    assert train.isdisjoint(evaluation)
    assert train | evaluation == {scenario.scenario_id for scenario in dataset.all()}


def test_single_and_combined_faults_are_classified() -> None:
    dataset = FaultDataset()
    singles = [scenario for scenario in dataset.all() if scenario.fault_count == 1]
    combined = [scenario for scenario in dataset.all() if scenario.fault_count > 1]
    assert len(singles) == 8
    assert {scenario.fault_type for scenario in singles} == {"single_access", "single_trunk"}
    assert combined
    assert {scenario.severity for scenario in singles} == {"low"}
    assert {scenario.severity for scenario in combined} == {"medium", "high"}


def test_normal_and_configuration_only_cases_are_distinct() -> None:
    dataset = FaultDataset()
    normal = dataset.normal_case()
    configuration_only = [scenario for scenario in dataset.all() if not scenario.functional_failure]
    assert normal.fault_count == 0
    assert normal.scenario_id == "normal"
    assert configuration_only
    assert all(scenario.fault_count > 0 for scenario in configuration_only)


def test_generated_scenario_preserves_partial_observation_contract() -> None:
    scenario = FaultDataset().sample("eval", seed=9, count=1)[0]
    env = NetworkEnv(split="eval", fault_scenario=scenario)
    observation, info = env.reset(seed=9)
    assert info == {"split": "eval"}
    assert {value for row in observation["connectivity"].values() for value in row.values()} == {"unknown"}
    serialized = repr(observation).lower()
    assert all(term not in serialized for term in ("scenario", "fault", "reference", "policy_healthy"))
    diagnosed, *_ = env.step(0)
    assert "unknown" not in {value for row in diagnosed["connectivity"].values() for value in row.values()}

