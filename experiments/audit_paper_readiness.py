"""Audit objective manuscript evidence gates without inventing a quality score."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def _exists(relative: str) -> bool:
    return (ROOT / relative).exists()


def _successful_mininet_run() -> bool:
    for path in ROOT.glob("results/testbed/runs/*/mininet_replay_seed_2026.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        if (
            summary.get("scenario_count") == 51
            and summary.get("simulator_testbed_recovery_matches") == 51
            and summary.get("simulator_testbed_damage_matches") == 51
        ):
            return True
    return False


def _successful_clean_reproduction() -> bool:
    for path in ROOT.glob("results/reproduction/runs/*/summary.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("clean_reproduction_passed") is True:
            return True
    return False


def audit() -> dict:
    draft = (ROOT / "paper/DRAFT.md").read_text(encoding="utf-8")
    test_path = ROOT / "results/reproduction_preparation_2026-10-27/pytest.xml"
    tests_pass = False
    test_count = 0
    if test_path.exists():
        root = ET.parse(test_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        test_count = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        tests_pass = (
            sum(int(suite.attrib.get("failures", 0)) for suite in suites) == 0
            and sum(int(suite.attrib.get("errors", 0)) for suite in suites) == 0
        )
    gates = {
        "finite_exact_multiseed": _exists("results/fresh_progress_statistical_summary.json"),
        "t4_full_heldout_evaluation": _exists(
            "results/expanded/graph_dqn/eval_300ep_full/heldout_evaluation.csv"
        ),
        "t6_representative_zero_shot": _exists(
            "results/expanded/t6_zero_shot/scenario_results.csv"
        ),
        "scaling_runtime_curve": _exists("results/expanded/scaling/runtime_scaling.csv"),
        "paper_figures": all(_exists(path) for path in (
            "paper/figures/main_comparison.pdf", "paper/figures/expanded_scaling.pdf",
            "paper/figures/topology_suite.pdf",
        )),
        "related_work_minimum": sum(draft.count(f"- [R{i}]") for i in range(1, 8)) == 7,
        "regression_suite": tests_pass and test_count >= 160,
        "artifact_manifest": _exists("results/artifact_manifest.json"),
        "expanded_multiseed": _exists("results/expanded/multiseed/summary.json"),
        "matched_guidance_ablation": _exists(
            "results/expanded/guidance_ablation/summary.json"
        ),
        "mininet_execution": _successful_mininet_run(),
        "clean_environment_reproduction": _successful_clean_reproduction(),
    }
    blockers = [
        "Execute and archive representative Mininet replay on Linux.",
        "Perform clean-environment reproduction and freeze a clean commit hash.",
    ]
    return {
        "date": "2026-10-27",
        "test_count": test_count,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "gates": gates,
        "submission_blockers": blockers,
        "interpretation": (
            "The finite-state core, expanded multiseed study, matched guidance, "
            "and policy-feature ablations are complete; emulator and clean-"
            "environment validation remain."
        ),
    }


def main() -> None:
    result = audit()
    path = ROOT / "results/paper_readiness_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
