from experiments.audit_paper_readiness import audit


def test_readiness_audit_distinguishes_completed_and_blocked_gates():
    result = audit()
    assert result["gates"]["finite_exact_multiseed"]
    assert result["gates"]["t4_full_heldout_evaluation"]
    assert result["gates"]["t6_representative_zero_shot"]
    assert result["gates"]["expanded_multiseed"]
    assert not result["gates"]["mininet_execution"]
    assert result["passed_gates"] < result["total_gates"]
