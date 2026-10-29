"""Hash paper-critical artifacts and record the executable environment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def main() -> None:
    required = [
        ROOT / "EXPERIMENT_EXPANSION_SPEC.md",
        ROOT / "results/expanded/SIX_SWITCH_STUDY.md",
        ROOT / "results/paper_readiness_audit.json",
        ROOT / "results/stratified_split_manifest.json",
        ROOT / "results/multiseed/multiseed_eval_seed_summary.csv",
        ROOT / "results/fresh_progress_comparison.csv",
        ROOT / "results/fresh_progress_statistical_summary.json",
        ROOT / "results/multiseed/safe_seed_2033_liveness_diagnostic.json",
        ROOT / "results/model_mismatch_summary.json",
        ROOT / "results/model_mismatch_scenario_results.csv",
        ROOT / "paper/DRAFT.md",
        ROOT / "paper/CLAIM_EVIDENCE.md",
        ROOT / "paper/EXPANDED_STUDY_PROTOCOL.md",
        ROOT / "paper/ABLATION_LOG.md",
        ROOT / "paper/RELATED_WORK_NOTES.md",
        ROOT / "results/expanded/graph_dqn/model_seed_2026_ep300.pt",
        ROOT / "results/expanded/graph_dqn/training_seed_2026_ep300.csv",
        ROOT / "results/expanded/graph_dqn/eval_300ep_full/heldout_evaluation.csv",
        ROOT / "results/expanded/graph_dqn/eval_300ep_full/heldout_evaluation_summary.json",
        ROOT / "results/expanded/graph_dqn/expanded_paper_summary.json",
        ROOT / "results/expanded/multiseed/seed_results.csv",
        ROOT / "results/expanded/multiseed/summary.json",
        ROOT / "results/expanded/guidance_ablation/paired_seed_results.csv",
        ROOT / "results/expanded/guidance_ablation/summary.json",
        ROOT / "results/expanded/policy_feature_ablation/paired_seed_results.csv",
        ROOT / "results/expanded/policy_feature_ablation/summary.json",
        ROOT / "results/expanded/scaling/runtime_scaling.csv",
        ROOT / "results/expanded/scaling/runtime_scaling.json",
        ROOT / "results/expanded/t6_zero_shot/scenario_results.csv",
        ROOT / "results/expanded/t6_zero_shot/summary.json",
        ROOT / "results/expanded/t6_zero_shot/acl_failure_diagnostic.json",
        ROOT / "paper/figures/expanded_scaling.pdf",
        ROOT / "paper/figures/topology_suite.pdf",
        ROOT / "results/expanded/graph_dqn/model_seed_2026_ep100_policy.pt",
        ROOT / "results/expanded/graph_dqn/training_seed_2026_ep100_policy.csv",
        ROOT / "results/expanded/t6_zero_shot_policy_ep100/scenario_results.csv",
        ROOT / "results/expanded/t6_zero_shot_policy_ep100/summary.json",
        ROOT / "paper/figures/main_comparison.pdf",
        ROOT / "paper/figures/per_seed_profile.pdf",
        ROOT / "paper/figures/seed2033_liveness_failure.pdf",
        ROOT / "paper/figures/model_mismatch.pdf",
    ]
    required.extend(
        ROOT / f"results/multiseed_progress/progress_safe_seed_{seed}.pt"
        for seed in range(2026, 2036)
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing release artifacts: {missing}")
    manifest = {
        "schema_version": 1,
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "worktree_dirty": bool(git_value("status", "--porcelain")),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": [digest(path) for path in required],
        "known_missing_external_evidence": [
            "results/testbed/mininet_replay_seed_2026.json"
        ],
    }
    output = ROOT / "results/artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact_count": len(manifest["artifacts"]),
        "worktree_dirty": manifest["git"]["worktree_dirty"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
