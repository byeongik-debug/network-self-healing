from experiments.evaluate_expanded_baseline import summarize


def test_expanded_summary_groups_split_and_condition():
    rows = [
        {"split": "test", "condition": "nominal", "success": True,
         "steps": 2, "action_failures": 0, "new_policy_damage_events": 0},
        {"split": "test", "condition": "nominal", "success": False,
         "steps": 4, "action_failures": 1, "new_policy_damage_events": 0},
    ]
    result = summarize(rows)["test:nominal:unknown"]
    assert result == {"episodes": 2, "success_rate": 0.5, "mean_steps": 3.0,
                      "total_action_failures": 1,
                      "total_new_policy_damage_events": 0,
                      "safety_deadlocks": 0}
