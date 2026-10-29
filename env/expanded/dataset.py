"""Deterministic train/validation/test scenarios with topology-level holdout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations

from env.expanded.faults import ExpandedFault, atomic_faults
from env.expanded.simulator import ExpandedNetworkSimulator
from env.expanded.topologies import build_topology


TRAIN_TOPOLOGIES = ("T2", "T3", "T4-Line")
HELD_OUT_TOPOLOGIES = ("T4-Ring",)


@dataclass(frozen=True)
class ExpandedScenario:
    scenario_id: str
    split: str
    topology: str
    faults: tuple[ExpandedFault, ...]
    initially_healthy: bool


def _identifier(topology: str, faults: tuple[ExpandedFault, ...]) -> str:
    payload = topology + "|" + "|".join(fault.fault_id for fault in faults)
    return f"{topology}-{sha256(payload.encode()).hexdigest()[:12]}"


def build_scenarios(max_pairs_per_topology: int = 32) -> tuple[ExpandedScenario, ...]:
    """Create atomic and bounded two-fault cases without topology leakage."""
    scenarios: list[ExpandedScenario] = []
    for topology_name in TRAIN_TOPOLOGIES + HELD_OUT_TOPOLOGIES:
        faults = atomic_faults(build_topology(topology_name))
        cases = [(fault,) for fault in faults]
        pairs = sorted(combinations(faults, 2), key=lambda pair: _identifier(topology_name, pair))
        cases.extend(pairs[:max_pairs_per_topology])
        for case in cases:
            digest = int(sha256(_identifier(topology_name, case).encode()).hexdigest(), 16)
            split = "test" if topology_name in HELD_OUT_TOPOLOGIES else (
                "validation" if digest % 5 == 0 else "train"
            )
            simulator = ExpandedNetworkSimulator(build_topology(topology_name))
            for fault in case:
                fault.apply(simulator)
            scenarios.append(ExpandedScenario(
                _identifier(topology_name, case), split, topology_name, case,
                simulator.is_policy_healthy(),
            ))
    result = tuple(sorted(scenarios, key=lambda scenario: scenario.scenario_id))
    validate_no_topology_leakage(result)
    return result


def validate_no_topology_leakage(scenarios: tuple[ExpandedScenario, ...]) -> None:
    train_seen = {s.topology for s in scenarios if s.split in ("train", "validation")}
    test_seen = {s.topology for s in scenarios if s.split == "test"}
    if train_seen & test_seen:
        raise ValueError(f"topology leakage across held-out test split: {train_seen & test_seen}")
    if test_seen != set(HELD_OUT_TOPOLOGIES):
        raise ValueError("test split must contain exactly the held-out topologies")
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario IDs must be unique")
