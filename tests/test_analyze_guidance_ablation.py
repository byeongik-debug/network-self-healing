from pathlib import Path

import pytest

from experiments.analyze_guidance_ablation import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_guidance_ablation_is_paired_and_complete():
    rows, summary = analyze(ROOT)
    assert len(rows) == 20
    assert summary["scope"]["paired"]
    assert summary["scope"]["seeds"] == [2026, 2027, 2028, 2029, 2030]
    expected = {
        "T4-Ring:nominal": (0.9345454545454546, 0.24363636363636365),
        "T4-Ring:combined": (0.9054545454545455, 0.41454545454545455),
        "T6-Mesh:nominal": (0.8833333333333333, 0.2333333333333333),
        "T6-Mesh:combined": (0.8333333333333334, 0.35),
    }
    for key, (unguided, difference) in expected.items():
        result = summary["results"][key]
        assert result["seeds"] == 5
        assert result["no_guidance_mean"] == pytest.approx(unguided)
        assert result["paired_mean_difference"] == pytest.approx(difference)
        assert result["total_guided_damage"] == 0
        assert result["total_no_guidance_damage"] == 0
