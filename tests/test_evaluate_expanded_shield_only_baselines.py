from experiments.evaluate_expanded_shield_only_baselines import (
    ALGORITHMS, _run_random, summarize,
)
from env.expanded.actions import build_action_catalog
from env.expanded.environment import ExpandedNetworkEnv
from env.expanded.faults import ExpandedFault
from env.expanded.topologies import build_topology
import numpy as np


def test_expanded_random_baselines_run_without_new_damage_on_atomic_case():
    topology = build_topology("T2")
    catalog = build_action_catalog(topology)
    for require_progress in (False, True):
        env = ExpandedNetworkEnv(topology, max_steps=30)
        env.reset((ExpandedFault("access_vlan", "H1", vlan=20),), seed=1)
        result = _run_random(
            env, catalog, np.random.default_rng(2026),
            require_progress=require_progress,
        )
        assert result["new_policy_damage_events"] == 0


def test_expanded_baseline_summary_has_all_groups():
    rows = []
    for topology, episodes in (("T4-Ring", 110), ("T6-Mesh", 12)):
        for condition in ("nominal", "combined"):
            for algorithm in ALGORITHMS:
                rows.extend({
                    "topology": topology, "condition": condition,
                    "algorithm": algorithm, "success": True, "steps": 2,
                    "truncated": False, "new_policy_damage_events": 0,
                    "safety_deadlock": False,
                } for _ in range(episodes))
    summary = summarize(rows)
    assert len(summary) == 12
    assert summary["T6-Mesh:combined:progress_shield_random"]["episodes"] == 12
