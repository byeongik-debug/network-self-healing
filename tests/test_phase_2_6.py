from __future__ import annotations

from copy import deepcopy

import numpy as np

from env.actions import ACTIONS
from env.faults import FaultDataset
from env.network_env import NetworkEnv
from env.observation_encoder import ObservationEncoder


def aid(kind: str, switch: str | None = None, port: str | None = None,
        vlan: int | None = None, allowed: bool | None = None) -> int:
    return next(action.id for action in ACTIONS if (
        action.kind, action.switch, action.port, action.vlan, action.allowed
    ) == (kind, switch, port, vlan, allowed))


def test_stratified_split_is_reproducible_disjoint_and_seeded() -> None:
    dataset = FaultDataset()
    first = dataset.stratified_manifest(seed=2026)
    second = dataset.stratified_manifest(seed=2026)
    different = dataset.stratified_manifest(seed=7)
    assert first == second
    assert first.seed == 2026
    assert set(first.train_ids).isdisjoint(first.eval_ids)
    assert len(first.train_ids) + len(first.eval_ids) == 255
    assert first != different


def test_stratified_split_has_each_severity_on_both_sides() -> None:
    dataset = FaultDataset()
    for split in ("train", "eval"):
        scenarios = dataset.stratified_split(split, seed=2026)
        assert {scenario.severity for scenario in scenarios} == {"low", "medium", "high"}
        assert any(scenario.fault_count == 1 for scenario in scenarios)
        assert any(scenario.fault_count > 1 for scenario in scenarios)


def test_observation_encoder_shape_determinism_and_no_mutation() -> None:
    env = NetworkEnv(scenario="combined_train")
    observation, _ = env.reset(seed=3)
    before = deepcopy(observation)
    encoder = ObservationEncoder()
    first = encoder.encode(observation)
    second = encoder.encode(observation)
    assert first.shape == (79,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert observation == before
    assert len(encoder.feature_names()) == 79


def test_diagnostic_one_hot_distinguishes_unknown_success_failure() -> None:
    env = NetworkEnv(scenario="trunk_missing_sw1")
    observation, _ = env.reset(seed=3)
    encoder = ObservationEncoder()
    unknown = encoder.encode(observation)
    names = encoder.feature_names()
    start = names.index("connectivity.H1.H2.unknown")
    assert unknown[start:start + 3].tolist() == [1.0, 0.0, 0.0]
    diagnosed, *_ = env.step(0)
    encoded = encoder.encode(diagnosed)
    assert encoded[start:start + 3].tolist() == [0.0, 0.0, 1.0]
    success = names.index("connectivity.H2.H4.unknown")
    assert encoded[success:success + 3].tolist() == [0.0, 1.0, 0.0]


def test_encoder_has_no_privileged_feature_names() -> None:
    names = " ".join(ObservationEncoder.feature_names()).lower()
    assert all(term not in names for term in (
        "reference", "fault", "scenario", "policy_healthy", "correct_action", "root_cause"
    ))


def test_action_mask_indices_match_action_ids_and_always_allow_control_actions() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=3)
    mask = env.get_action_mask()
    valid_ids = {action["id"] for action in env.get_valid_actions()}
    assert len(mask) == len(ACTIONS) == 19
    assert [action.id for action in ACTIONS] == list(range(19))
    assert valid_ids == {index for index, allowed in enumerate(mask) if allowed}
    assert mask[aid("diagnose")]
    assert mask[aid("declare_done")]
    assert not mask[aid("rollback")]
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    assert env.get_action_mask()[aid("rollback")]


def test_service_damage_tracks_events_unique_pairs_and_final_state() -> None:
    env = NetworkEnv(scenario="access_vlan_sw1")
    env.reset(seed=3)
    assert env.get_metrics()["initial_damaged_policies"] == 3
    env.step(aid("set_access_vlan", "SW1", "eth1", 10))
    harmful = aid("set_access_vlan", "SW2", "eth2", 10)
    env.step(harmful)
    env.step(aid("rollback"))
    env.step(harmful)
    metrics = env.get_metrics()
    assert metrics["new_policy_damage_events"] == 6
    assert metrics["unique_newly_damaged_policies"] == 3
    assert metrics["service_damage_actions"] == 2
    assert metrics["healthy_network_impacts"] == 2
    assert metrics["post_recovery_damage_events"] == 2
    assert metrics["continuously_healthy_policies"] == 1
    assert metrics["final_damaged_policies"] == 3


def test_completion_and_step_limit_remain_distinct() -> None:
    failed = NetworkEnv(scenario="access_vlan_sw1")
    failed.reset(seed=3)
    _, _, terminated, truncated, info = failed.step(aid("declare_done"))
    assert terminated and not truncated and not info["recovery_success"]

    limited = NetworkEnv(scenario="access_vlan_sw1", max_steps=1)
    limited.reset(seed=3)
    _, _, terminated, truncated, _ = limited.step(aid("diagnose"))
    assert not terminated and truncated
