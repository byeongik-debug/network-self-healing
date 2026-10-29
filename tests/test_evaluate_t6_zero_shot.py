from experiments.evaluate_t6_zero_shot import representative_scenarios


def test_t6_representative_manifest_is_deterministic_and_fault_stratified():
    first = representative_scenarios()
    assert first == representative_scenarios()
    assert len(first) == 12
    assert len({scenario.scenario_id for scenario in first}) == 12
    atomic_kinds = {scenario.faults[0].kind for scenario in first if len(scenario.faults) == 1}
    assert atomic_kinds == {
        "access_vlan", "trunk_remove", "link_down", "port_shutdown",
        "acl_deny_required", "acl_allow_forbidden",
    }
