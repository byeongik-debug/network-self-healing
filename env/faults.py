"""Exhaustive, reproducible fault-combination generation for Phase 2.5."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import random
import json
from pathlib import Path
from typing import Literal

from config.topology import REFERENCE_PORTS
from .simulator import NetworkSimulator

FaultKind = Literal["access_vlan", "trunk_remove"]
Severity = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True, order=True)
class FaultMutation:
    kind: FaultKind
    switch: str
    port: str
    vlan: int

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.switch}:{self.port}:{self.vlan}"

    def as_environment_mutation(self) -> dict[str, object]:
        return {"kind": self.kind, "switch": self.switch, "port": self.port, "value": self.vlan}


@dataclass(frozen=True)
class FaultScenario:
    scenario_id: str
    mutations: tuple[FaultMutation, ...]
    fault_type: str
    severity: Severity
    functional_failure: bool

    @property
    def fault_count(self) -> int:
        return len(self.mutations)


@dataclass(frozen=True)
class SplitManifest:
    strategy: str
    seed: int
    train_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "eval_ids": list(self.eval_ids),
        }


def atomic_faults() -> tuple[FaultMutation, ...]:
    """Return the eight independent binary faults supported by the action space."""
    faults: list[FaultMutation] = []
    for switch, ports in sorted(REFERENCE_PORTS.items()):
        for port, settings in sorted(ports.items()):
            if settings["mode"] == "access":
                current = int(settings["access_vlan"])
                faults.append(FaultMutation("access_vlan", switch, port, 20 if current == 10 else 10))
            elif settings["mode"] == "trunk":
                for vlan in settings["allowed_vlans"]:
                    faults.append(FaultMutation("trunk_remove", switch, port, int(vlan)))
    return tuple(sorted(faults))


class FaultDataset:
    """Enumerate unique combinations and split identities independently of sampling seeds."""

    def __init__(self) -> None:
        self._scenarios = self._enumerate()
        if len({scenario.scenario_id for scenario in self._scenarios}) != len(self._scenarios):
            raise RuntimeError("duplicate fault combinations generated")

    @staticmethod
    def _enumerate() -> tuple[FaultScenario, ...]:
        atoms = atomic_faults()
        scenarios: list[FaultScenario] = []
        for count in range(1, len(atoms) + 1):
            for selected in combinations(atoms, count):
                simulator = NetworkSimulator()
                for mutation in selected:
                    simulator.apply_fault(mutation.as_environment_mutation())
                identifier = "+".join(mutation.key for mutation in selected)
                scenarios.append(FaultScenario(
                    identifier,
                    selected,
                    classify_fault_type(selected),
                    classify_severity(count),
                    not simulator.is_policy_healthy(),
                ))
        return tuple(scenarios)

    def all(self) -> tuple[FaultScenario, ...]:
        return self._scenarios

    def split(self, split: Literal["train", "eval"]) -> tuple[FaultScenario, ...]:
        """Use stable identity hashing, not random seeds, for a disjoint 80/20 split."""
        return tuple(
            scenario for scenario in self._scenarios
            if (int(sha256(scenario.scenario_id.encode()).hexdigest(), 16) % 5 == 0)
            == (split == "eval")
        )

    def sample(
        self,
        split: Literal["train", "eval"],
        seed: int,
        count: int = 1,
        severity: Severity | None = None,
        fault_type: str | None = None,
        functional_failure: bool | None = None,
    ) -> tuple[FaultScenario, ...]:
        candidates = [
            scenario for scenario in self.split(split)
            if (severity is None or scenario.severity == severity)
            and (fault_type is None or scenario.fault_type == fault_type)
            and (functional_failure is None or scenario.functional_failure == functional_failure)
        ]
        if count < 0 or count > len(candidates):
            raise ValueError(f"cannot sample {count} from {len(candidates)} matching scenarios")
        return tuple(random.Random(seed).sample(candidates, count))

    def stratified_manifest(self, seed: int = 2026) -> SplitManifest:
        """Split each viable severity/type stratum without duplicating identities."""
        groups: dict[tuple[str, str], list[FaultScenario]] = {}
        for scenario in self._scenarios:
            groups.setdefault((scenario.severity, scenario.fault_type), []).append(scenario)
        rng = random.Random(seed)
        train: list[str] = []
        evaluation: list[str] = []
        for key in sorted(groups):
            members = sorted(groups[key], key=lambda item: item.scenario_id)
            rng.shuffle(members)
            eval_count = 0 if len(members) == 1 else max(1, round(len(members) * 0.2))
            evaluation.extend(item.scenario_id for item in members[:eval_count])
            train.extend(item.scenario_id for item in members[eval_count:])
        return SplitManifest(
            "stratified_severity_and_type",
            seed,
            tuple(sorted(train)),
            tuple(sorted(evaluation)),
        )

    def stratified_split(
        self, split: Literal["train", "eval"], seed: int = 2026
    ) -> tuple[FaultScenario, ...]:
        manifest = self.stratified_manifest(seed)
        selected = set(manifest.train_ids if split == "train" else manifest.eval_ids)
        return tuple(scenario for scenario in self._scenarios if scenario.scenario_id in selected)

    def save_stratified_manifest(self, path: str | Path, seed: int = 2026) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.stratified_manifest(seed).as_dict(), indent=2),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def normal_case() -> FaultScenario:
        return FaultScenario("normal", (), "none", "none", False)


def classify_fault_type(mutations: tuple[FaultMutation, ...]) -> str:
    access_count = sum(mutation.kind == "access_vlan" for mutation in mutations)
    trunk_count = len(mutations) - access_count
    if len(mutations) == 1:
        return "single_access" if access_count else "single_trunk"
    if access_count and trunk_count:
        return "access_trunk_combined"
    return "multi_access" if access_count else "multi_trunk"


def classify_severity(count: int) -> Severity:
    if count == 0:
        return "none"
    if count == 1:
        return "low"
    if count <= 3:
        return "medium"
    return "high"
