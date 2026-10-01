"""Run recovery baselines on the same deterministic scenario set."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable

from baselines.common import RecoveryEnvironment, RecoveryResult
from baselines.config_diff import ConfigurationDiffRecovery
from baselines.rule_based import RuleBasedRecovery
from config.scenarios import EVAL_SCENARIOS, TRAIN_SCENARIOS
from env.network_env import NetworkEnv


@dataclass(frozen=True)
class ExperimentRow:
    row_type: str
    algorithm: str
    split: str
    scenario: str
    seed: int | str
    success_rate: float
    actual_configuration_changes: float
    unnecessary_actions: float
    invalid_actions: float
    diagnostic_actions: float
    rollbacks: float
    healthy_network_impacts: float
    steps: float
    python_simulation_seconds: float


AlgorithmFactory = Callable[[], RuleBasedRecovery | ConfigurationDiffRecovery]


class ExperimentRunner:
    """Evaluate algorithms with identical scenarios, seeds, and step limits."""

    def __init__(self, seeds: Iterable[int] = (0,), max_steps: int = 30) -> None:
        self.seeds = tuple(seeds)
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not 1 <= max_steps <= 30:
            raise ValueError("max_steps must be between 1 and 30")
        self.max_steps = max_steps

    def run(
        self,
        algorithms: Iterable[AlgorithmFactory] | None = None,
    ) -> list[ExperimentRow]:
        factories = tuple(algorithms or (RuleBasedRecovery, ConfigurationDiffRecovery))
        episodes: list[ExperimentRow] = []
        for factory in factories:
            for split, catalog in (("train", TRAIN_SCENARIOS), ("eval", EVAL_SCENARIOS)):
                for scenario in sorted(catalog):
                    for seed in self.seeds:
                        episodes.append(self._run_episode(factory, split, scenario, seed))
        return episodes + self._aggregate(episodes, "scenario_mean") + self._aggregate(episodes, "overall_mean")

    def _run_episode(
        self, factory: AlgorithmFactory, split: str, scenario: str, seed: int
    ) -> ExperimentRow:
        env = NetworkEnv(split=split, scenario=scenario, max_steps=self.max_steps)  # type: ignore[arg-type]
        env.reset(seed=seed)
        algorithm = factory()
        started = perf_counter()
        result: RecoveryResult = algorithm.recover(env, max_steps=self.max_steps)
        elapsed = perf_counter() - started
        metrics = env.get_metrics()
        return ExperimentRow(
            "episode", result.algorithm, split, scenario, seed, float(result.success),
            float(metrics["actual_configuration_changes"]),
            float(metrics["unnecessary_actions"]), float(metrics["invalid_actions"]),
            float(metrics["diagnostic_actions"]), float(metrics["rollbacks"]),
            float(metrics["healthy_network_impacts"]), float(result.steps), elapsed,
        )

    @staticmethod
    def _aggregate(rows: list[ExperimentRow], level: str) -> list[ExperimentRow]:
        groups: dict[tuple[str, ...], list[ExperimentRow]] = {}
        for row in rows:
            key = (row.algorithm, row.split, row.scenario) if level == "scenario_mean" else (row.algorithm,)
            groups.setdefault(key, []).append(row)
        aggregated: list[ExperimentRow] = []
        numeric = (
            "success_rate", "actual_configuration_changes", "unnecessary_actions",
            "invalid_actions", "diagnostic_actions", "rollbacks",
            "healthy_network_impacts", "steps", "python_simulation_seconds",
        )
        for key, values in sorted(groups.items()):
            sample = values[0]
            aggregated.append(ExperimentRow(
                level,
                sample.algorithm,
                sample.split if level == "scenario_mean" else "all",
                sample.scenario if level == "scenario_mean" else "__overall__",
                "mean",
                *[mean(getattr(value, field) for value in values) for field in numeric],
            ))
        return aggregated


def save_results_csv(rows: Iterable[ExperimentRow], path: str | Path) -> Path:
    """Write explicit episode and aggregate rows using the standard library."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fieldnames = list(ExperimentRow.__dataclass_fields__)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in materialized)
    return destination


def main() -> None:
    rows = ExperimentRunner().run()
    output = save_results_csv(rows, Path("results") / "baseline_results.csv")
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
