from pathlib import Path

import pytest

from experiments.analyze_policy_feature_ablation import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_policy_feature_ablation_is_complete_and_paired():
    rows, summary = analyze(ROOT)
    assert len(rows) == 30
    assert summary["protocol"]["seeds"] == [2026, 2027, 2028, 2029, 2030]
    expected = {
        "T4-Ring:nominal": (0.9963636363636363, 0.06181818181818181),
        "T4-Ring:combined": (0.990909090909091, 0.08545454545454548),
        "T6-Mesh:nominal": (0.9833333333333334, 0.1),
        "T6-Mesh:combined": (0.9666666666666666, 0.1333333333333333),
    }
    for key, (feature_mean, difference) in expected.items():
        result = summary["results"][key]
        assert result["policy_13d_mean"] == pytest.approx(feature_mean)
        assert result["paired_mean_difference"] == pytest.approx(difference)


def test_policy_features_improve_atomic_acl_mean_in_both_conditions():
    _, summary = analyze(ROOT)
    expected = {"nominal": 0.9, "combined": 0.8}
    for condition, feature_mean in expected.items():
        result = summary["results"][f"T6-atomic-ACL:{condition}"]
        assert result["policy_13d_mean"] > result["base_11d_mean"]
        assert result["policy_13d_mean"] == feature_mean
