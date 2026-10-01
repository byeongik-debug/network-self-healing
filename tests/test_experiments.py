from __future__ import annotations

import csv

from experiments.runner import ExperimentRunner, save_results_csv


def stable_fields(row: object) -> tuple[object, ...]:
    return tuple(
        value for key, value in vars(row).items()
        if key != "python_simulation_seconds"
    )


def test_experiment_is_reproducible_except_wall_clock_timing() -> None:
    first = ExperimentRunner(seeds=(3, 5)).run()
    second = ExperimentRunner(seeds=(3, 5)).run()
    assert [stable_fields(row) for row in first] == [stable_fields(row) for row in second]


def test_experiment_uses_both_splits_and_all_scenarios() -> None:
    rows = ExperimentRunner(seeds=(7,)).run()
    episodes = [row for row in rows if row.row_type == "episode"]
    assert len(episodes) == 12
    assert {row.split for row in episodes} == {"train", "eval"}
    assert {row.algorithm for row in episodes} == {"rule_based", "configuration_diff"}
    assert all(row.steps <= 30 for row in episodes)


def test_csv_contains_episode_scenario_and_overall_rows(tmp_path) -> None:
    rows = ExperimentRunner(seeds=(7,)).run()
    destination = save_results_csv(rows, tmp_path / "nested" / "results.csv")
    with destination.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved
    assert {row["row_type"] for row in saved} == {"episode", "scenario_mean", "overall_mean"}
    assert "python_simulation_seconds" in saved[0]
    assert {row["scenario"] for row in saved if row["row_type"] == "overall_mean"} == {"__overall__"}
