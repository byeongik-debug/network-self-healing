from experiments.benchmark_expanded_scaling import benchmark


def test_scaling_benchmark_reports_ordered_growth_and_timings():
    rows = benchmark(repeats=1)
    assert [row["switches"] for row in rows] == [2, 3, 4, 4, 5, 6]
    assert rows[-1]["policy_scoped_actions"] > rows[0]["policy_scoped_actions"]
    assert rows[-1]["full_actions"] > rows[-1]["policy_scoped_actions"]
    assert all(row["shield_mask_ms"] >= 0 for row in rows)
    assert all(row["q_inference_ms"] >= 0 for row in rows)
