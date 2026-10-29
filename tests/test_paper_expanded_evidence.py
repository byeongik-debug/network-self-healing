import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_t4_manuscript_numbers_match_frozen_summary():
    summary = json.loads((
        ROOT / "results/expanded/graph_dqn/expanded_paper_summary.json"
    ).read_text(encoding="utf-8"))["evaluation"]
    draft = (ROOT / "paper/DRAFT.md").read_text(encoding="utf-8")
    assert summary["nominal:all"]["success_rate"] == 107 / 110
    assert summary["combined:all"]["success_rate"] == 105 / 110
    assert "97.27%" in draft and "95.45%" in draft
    assert summary["nominal:all"]["damage"] == 0
    assert summary["combined:all"]["damage"] == 0


def test_t6_manuscript_numbers_match_frozen_results():
    summary = json.loads((
        ROOT / "results/expanded/t6_zero_shot/summary.json"
    ).read_text(encoding="utf-8"))
    draft = (ROOT / "paper/DRAFT.md").read_text(encoding="utf-8")
    assert summary["nominal"]["success_rate"] == 10 / 12
    assert summary["combined"]["success_rate"] == 10 / 12
    assert summary["nominal"]["impacting_success_rate"] == 6 / 8
    assert summary["combined"]["no_impact_control_success_rate"] == 1.0
    assert summary["nominal"]["damage"] == summary["combined"]["damage"] == 0
    assert "83.33%" in draft


def test_scaling_claims_match_runtime_csv():
    with (ROOT / "results/expanded/scaling/runtime_scaling.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["topology"]: row for row in csv.DictReader(handle)}
    assert int(rows["T2"]["full_actions"]) == 44
    assert int(rows["T6-Mesh"]["full_actions"]) == 1016
    assert int(rows["T6-Mesh"]["policy_scoped_actions"]) == 468
    shield_ms = float(rows["T6-Mesh"]["shield_mask_ms"])
    assert shield_ms < 100
    assert shield_ms > float(rows["T6-Mesh"]["q_inference_ms"])


def test_manuscript_has_no_duplicate_related_work_or_reference_sections():
    draft = (ROOT / "paper/DRAFT.md").read_text(encoding="utf-8")
    assert draft.count("### 1.1 관련 연구") == 1
    assert draft.count("## 참고문헌") == 1
    assert draft.count("## 주요 결과 자료") == 1
    for citation in range(1, 8):
        assert draft.count(f"- [R{citation}]") == 1
