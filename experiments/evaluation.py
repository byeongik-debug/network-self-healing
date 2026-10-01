"""Privileged evaluation predicates kept outside recovery algorithms."""

from __future__ import annotations

from typing import Any, Mapping

from config.topology import REFERENCE_PORTS
from env.simulator import NetworkSimulator


def functional_recovery(simulator: NetworkSimulator) -> bool:
    """Require both intended same-VLAN reachability and cross-VLAN isolation."""
    return simulator.is_policy_healthy()


def configuration_recovery(ports: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> bool:
    """Privileged exact comparison used only by evaluation code."""
    return dict(ports) == REFERENCE_PORTS


def failure_reason(functional: bool, configuration: bool, truncated: bool) -> str:
    if truncated:
        return "step_limit"
    if not functional and not configuration:
        return "functional_and_configuration_failure"
    if not functional:
        return "functional_failure"
    if not configuration:
        return "configuration_mismatch_only"
    return "none"

