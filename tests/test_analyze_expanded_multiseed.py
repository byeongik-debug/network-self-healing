from pathlib import Path

import pytest

from experiments.analyze_expanded_multiseed import collect


ROOT = Path(__file__).resolve().parents[1]


def test_five_seed_expanded_summary_uses_seed_as_statistical_unit():
    rows, summary = collect(ROOT)
    assert len(rows) == 20
    assert summary["scope"]["seeds"] == [2026, 2027, 2028, 2029, 2030]
    assert summary["scope"]["statistical_unit"] == "seed"
    expected = {
        "T4-Ring:nominal": 0.6909090909090909,
        "T4-Ring:combined": 0.4909090909090909,
        "T6-Mesh:nominal": 0.65,
        "T6-Mesh:combined": 0.4833333333333333,
    }
    for key, mean in expected.items():
        result = summary["evaluation"][key]
        assert result["seeds"] == 5
        assert result["success_rate"]["mean"] == pytest.approx(mean)
        assert result["total_damage"] == 0


def test_multiseed_summary_exposes_seed_instability():
    _, summary = collect(ROOT)
    nominal = summary["evaluation"]["T4-Ring:nominal"]["success_rate"]
    assert nominal["min"] == 0.0
    assert nominal["max"] > 0.95
    assert nominal["sample_sd"] > 0.35
