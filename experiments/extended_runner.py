"""Phase 2.6 evaluation over legacy or stratified fault splits."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, Literal

from baselines.config_diff import ConfigurationDiffRecovery
from baselines.rule_based import RuleBasedRecovery
from env.faults import FaultDataset, FaultScenario
from env.network_env import NetworkEnv
from .evaluation import configuration_recovery, failure_reason, functional_recovery

AlgorithmFactory = Callable[[], RuleBasedRecovery | ConfigurationDiffRecovery]
SplitStrategy = Literal["legacy", "stratified"]


@dataclass(frozen=True)
class ExtendedExperimentRow:
    row_type: str
    algorithm: str
    split: str
    split_strategy: str
    split_seed: int
    scenario_id: str
    fault_type: str
    severity: str
    initial_functional_failure_rate: float
    functional_recovery_rate: float
    configuration_recovery_rate: float
    actual_configuration_changes: float
    unnecessary_actions: float
    invalid_actions: float
    diagnostic_actions: float
    rollbacks: float
    healthy_network_impacts: float
    new_policy_damage_events: float
    unique_newly_damaged_policies: float
    continuously_healthy_policies: float
    final_damaged_policies: float
    post_recovery_damage_events: float
    steps: float
    python_simulation_seconds: float
    termination_status: str
    failure_reason: str


class ExtendedExperimentRunner:
    def __init__(
        self,
        max_steps: int = 30,
        dataset: FaultDataset | None = None,
        split_strategy: SplitStrategy = "legacy",
        split_seed: int = 2026,
    ) -> None:
        if not 1 <= max_steps <= 30:
            raise ValueError("max_steps must be between 1 and 30")
        if split_strategy not in ("legacy", "stratified"):
            raise ValueError("split_strategy must be 'legacy' or 'stratified'")
        self.max_steps = max_steps
        self.dataset = dataset or FaultDataset()
        self.split_strategy = split_strategy
        self.split_seed = split_seed

    def run(self, algorithms: Iterable[AlgorithmFactory] | None = None) -> list[ExtendedExperimentRow]:
        factories = tuple(algorithms or (RuleBasedRecovery, ConfigurationDiffRecovery))
        episodes = [
            self._episode(factory, split, scenario)
            for factory in factories
            for split in ("train", "eval")
            for scenario in self._split(split)
        ]
        return (
            episodes
            + self._aggregate(episodes, "fault_type_mean")
            + self._aggregate(episodes, "severity_mean")
            + self._aggregate(episodes, "overall_mean")
        )

    def _split(self, split: str) -> tuple[FaultScenario, ...]:
        if self.split_strategy == "legacy":
            return self.dataset.split(split)  # type: ignore[arg-type]
        return self.dataset.stratified_split(split, self.split_seed)  # type: ignore[arg-type]

    def _episode(self, factory: AlgorithmFactory, split: str, scenario: FaultScenario) -> ExtendedExperimentRow:
        env = NetworkEnv(split=split, max_steps=self.max_steps, fault_scenario=scenario)  # type: ignore[arg-type]
        env.reset(seed=self.split_seed)
        algorithm = factory()
        started = perf_counter()
        result = algorithm.recover(env, max_steps=self.max_steps)
        elapsed = perf_counter() - started
        functional = functional_recovery(env.simulator)
        configuration = configuration_recovery(env.simulator.snapshot())
        metrics = env.get_metrics()
        return ExtendedExperimentRow(
            row_type="episode", algorithm=result.algorithm, split=split,
            split_strategy=self.split_strategy, split_seed=self.split_seed,
            scenario_id=scenario.scenario_id, fault_type=scenario.fault_type,
            severity=scenario.severity,
            initial_functional_failure_rate=float(scenario.functional_failure),
            functional_recovery_rate=float(functional),
            configuration_recovery_rate=float(configuration),
            actual_configuration_changes=float(metrics["actual_configuration_changes"]),
            unnecessary_actions=float(metrics["unnecessary_actions"]),
            invalid_actions=float(metrics["invalid_actions"]),
            diagnostic_actions=float(metrics["diagnostic_actions"]),
            rollbacks=float(metrics["rollbacks"]),
            healthy_network_impacts=float(metrics["healthy_network_impacts"]),
            new_policy_damage_events=float(metrics["new_policy_damage_events"]),
            unique_newly_damaged_policies=float(metrics["unique_newly_damaged_policies"]),
            continuously_healthy_policies=float(metrics["continuously_healthy_policies"]),
            final_damaged_policies=float(metrics["final_damaged_policies"]),
            post_recovery_damage_events=float(metrics["post_recovery_damage_events"]),
            steps=float(result.steps), python_simulation_seconds=elapsed,
            termination_status=termination_status(result.terminated, result.truncated, functional),
            failure_reason=failure_reason(functional, configuration, result.truncated),
        )

    @staticmethod
    def _aggregate(episodes: list[ExtendedExperimentRow], level: str) -> list[ExtendedExperimentRow]:
        groups: dict[tuple[str, ...], list[ExtendedExperimentRow]] = {}
        for row in episodes:
            if level == "fault_type_mean":
                key = (row.algorithm, row.split, row.fault_type)
            elif level == "severity_mean":
                key = (row.algorithm, row.split, row.severity)
            else:
                key = (row.algorithm,)
            groups.setdefault(key, []).append(row)
        numeric = (
            "initial_functional_failure_rate", "functional_recovery_rate",
            "configuration_recovery_rate", "actual_configuration_changes",
            "unnecessary_actions", "invalid_actions", "diagnostic_actions", "rollbacks",
            "healthy_network_impacts", "new_policy_damage_events",
            "unique_newly_damaged_policies", "continuously_healthy_policies",
            "final_damaged_policies", "post_recovery_damage_events", "steps",
            "python_simulation_seconds",
        )
        output: list[ExtendedExperimentRow] = []
        for _, rows in sorted(groups.items()):
            first = rows[0]
            averages = {field: mean(getattr(row, field) for row in rows) for field in numeric}
            output.append(ExtendedExperimentRow(
                row_type=level, algorithm=first.algorithm,
                split=first.split if level != "overall_mean" else "all",
                split_strategy=first.split_strategy, split_seed=first.split_seed,
                scenario_id="__aggregate__",
                fault_type=first.fault_type if level == "fault_type_mean" else "all",
                severity=first.severity if level == "severity_mean" else "all",
                termination_status=summarize(rows, "termination_status"),
                failure_reason=summarize(rows, "failure_reason"),
                **averages,
            ))
        return output


def termination_status(terminated: bool, truncated: bool, functional: bool) -> str:
    if truncated:
        return "max_steps_exceeded"
    if terminated:
        return "functional_completion" if functional else "incorrect_completion"
    return "incomplete"


def summarize(rows: Iterable[ExtendedExperimentRow], field: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(getattr(row, field))
        if value != "none":
            counts[value] = counts.get(value, 0) + 1
    return ";".join(f"{value}:{count}" for value, count in sorted(counts.items())) or "none"


def save_extended_csv(rows: Iterable[ExtendedExperimentRow], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExtendedExperimentRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return destination


def main() -> None:
    dataset = FaultDataset()
    rows = ExtendedExperimentRunner(dataset=dataset, split_strategy="stratified").run()
    dataset.save_stratified_manifest(Path("results") / "stratified_split_manifest.json", seed=2026)
    output = save_extended_csv(rows, Path("results") / "stratified_baseline_results.csv")
    print(f"Saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
