from experiments.evaluate_graph_dqn import EVALUATION_CONDITIONS


def test_graph_evaluation_has_nominal_and_mismatch_conditions():
    assert set(EVALUATION_CONDITIONS) == {"nominal", "combined"}
    assert EVALUATION_CONDITIONS["combined"]["command_failure_probability"] > 0
    assert EVALUATION_CONDITIONS["combined"]["telemetry_delay"] > 0
