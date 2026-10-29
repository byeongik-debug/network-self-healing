import pytest

from env.faults import FaultDataset
from experiments.evaluate_shield_only_baselines import evaluate, summarize


def test_progress_selectors_recover_small_eval_subset_without_damage():
    scenarios = FaultDataset().stratified_split("eval", 2026)[:3]
    for method in ("progress_random", "progress_first"):
        rows = evaluate(method, scenarios, 2026)
        assert len(rows) == 3
        assert all(row.functional_recovery for row in rows)
        assert all(not row.timeout for row in rows)


def test_summary_uses_damage_only_among_successes():
    scenarios = FaultDataset().stratified_split("eval", 2026)[:4]
    rows = evaluate("valid_random", scenarios, 2026)
    summary = summarize(rows)
    assert summary["episodes"] == 4
    if summary["functional_recoveries"]:
        expected = sum(row.new_policy_damage_events for row in rows
                       if row.functional_recovery) / summary["functional_recoveries"]
        assert summary["damage_per_successful_episode"] == pytest.approx(expected)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        evaluate("unknown", (), 2026)
