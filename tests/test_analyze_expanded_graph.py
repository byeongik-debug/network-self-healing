from experiments.analyze_expanded_graph import evaluation_summary, training_summary


def test_evaluation_summary_separates_impacting_and_controls():
    rows = [
        {"condition": "nominal", "initially_healthy": "False", "success": "True",
         "steps": "3", "new_policy_damage_events": "0"},
        {"condition": "nominal", "initially_healthy": "True", "success": "False",
         "steps": "30", "new_policy_damage_events": "0"},
    ]
    result = evaluation_summary(rows, seed=1)
    assert result["nominal:all"]["success_rate"] == 0.5
    assert result["nominal:impacting"]["success_rate"] == 1.0
    assert result["nominal:no_impact_control"]["success_rate"] == 0.0


def test_training_summary_reports_guidance_fraction():
    rows = [{"steps": "4", "success": "True", "guided_actions": "2",
             "mean_loss": "0.5", "new_policy_damage_events": "0"}]
    result = training_summary(rows, bin_size=50)[0]
    assert result["guided_action_fraction"] == 0.5
    assert result["success_rate"] == 1.0
