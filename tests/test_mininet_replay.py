from __future__ import annotations

import pytest

from types import SimpleNamespace

from testbed.mininet_vlan_replay import parse_action, validate_trajectories


def test_parse_control_actions() -> None:
    assert parse_action("diagnose") == {"kind": "diagnose"}
    assert parse_action("rollback") == {"kind": "rollback"}
    assert parse_action("declare_done") == {"kind": "declare_done"}


def test_parse_configuration_actions() -> None:
    assert parse_action("set_access_vlan(SW2,eth1,vlan=10)") == {
        "kind": "set_access_vlan",
        "switch": "SW2",
        "port": "eth1",
        "vlan": 10,
        "operation": None,
    }
    assert parse_action("set_trunk_vlan(SW1,eth3,vlan=20,allow)") == {
        "kind": "set_trunk_vlan",
        "switch": "SW1",
        "port": "eth3",
        "vlan": 20,
        "operation": "allow",
    }


def test_parse_rejects_unknown_or_malformed_action() -> None:
    with pytest.raises(ValueError):
        parse_action("set_trunk_vlan(SW1,eth3,vlan=20)")
    with pytest.raises(ValueError):
        parse_action("shell(rm -rf /)")


def test_validate_trajectories_checks_coverage_and_outcomes() -> None:
    scenario = SimpleNamespace(scenario_id="scenario-1")
    row = {
        "scenario_id": "scenario-1",
        "actions": ["diagnose", "declare_done"],
        "recovery_success": True,
        "new_policy_damage_events": 0,
    }
    assert validate_trajectories([row], [scenario]) == {"scenario-1": row}
    with pytest.raises(ValueError, match="missing trajectories"):
        validate_trajectories([], [scenario])
    with pytest.raises(ValueError, match="expected outcomes"):
        validate_trajectories([{k: v for k, v in row.items() if k != "recovery_success"}], [scenario])
