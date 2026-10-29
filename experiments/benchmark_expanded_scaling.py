"""Measure action growth and shield/inference cost from two to six switches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

from agents.graph_dqn import GraphActionQNetwork
from env.expanded.actions import build_action_catalog, validity_mask
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import atomic_faults
from env.expanded.graph_observation import GraphObservationEncoder
from env.expanded.shield import environment_safety_mask
from env.expanded.topologies import build_topology


TOPOLOGY_ORDER = ("T2", "T3", "T4-Line", "T4-Ring", "T5-Mesh", "T6-Mesh")


def _milliseconds(operation, repeats: int) -> tuple[float, object]:
    durations, value = [], None
    for _ in range(repeats):
        start = perf_counter()
        value = operation()
        durations.append((perf_counter() - start) * 1000.0)
    return fmean(durations), value


def benchmark(repeats: int = 3) -> list[dict]:
    torch.manual_seed(2026)
    network = GraphActionQNetwork(8, 5, 11, hidden=64).eval()
    rows = []
    for name in TOPOLOGY_ORDER:
        topology = build_topology(name)
        full_catalog = build_action_catalog(topology)
        catalog = build_action_catalog(topology, acl_scope="policy")
        kinds = Counter(candidate.action.kind for candidate in catalog)
        env = ExpandedNetworkEnv(topology)
        observation, _ = env.reset((atomic_faults(topology)[0],), seed=2026)
        encoder = GraphObservationEncoder(topology, catalog)
        encode_ms, encoded = _milliseconds(lambda: encoder.encode(observation), repeats)
        validity_ms, _ = _milliseconds(lambda: validity_mask(env.simulator, catalog), repeats)
        shield_ms, mask = _milliseconds(lambda: environment_safety_mask(env, catalog), repeats)
        with torch.no_grad():
            inference_ms, _ = _milliseconds(lambda: network(encoded), repeats)
        rows.append({
            "topology": name,
            "switches": len(topology.switches),
            "hosts": len(topology.hosts),
            "vlans": len(topology.vlans),
            "links": len(topology.links),
            "full_actions": len(full_catalog),
            "policy_scoped_actions": len(catalog),
            "acl_actions": kinds["set_acl"],
            "trunk_actions": kinds["set_trunk_vlan"],
            "valid_actions_after_shield": int(mask.sum()),
            "encode_ms": encode_ms,
            "validity_mask_ms": validity_ms,
            "shield_mask_ms": shield_ms,
            "q_inference_ms": inference_ms,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/expanded/scaling"))
    args = parser.parse_args()
    rows = benchmark(args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runtime_scaling.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "repeats": args.repeats,
        "device": "cpu",
        "timing_warning": "single-process wall clock; descriptive, not a hardware-neutral benchmark",
        "rows": rows,
    }
    (args.output_dir / "runtime_scaling.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
