"""Fixed discrete recovery action catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActionKind = Literal["diagnose", "set_access_vlan", "set_trunk_vlan", "rollback", "declare_done"]


@dataclass(frozen=True)
class ActionSpec:
    id: int
    kind: ActionKind
    switch: str | None = None
    port: str | None = None
    vlan: int | None = None
    allowed: bool | None = None


def build_action_catalog() -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = [ActionSpec(0, "diagnose")]
    for switch in ("SW1", "SW2"):
        for port in ("eth1", "eth2"):
            for vlan in (10, 20):
                actions.append(ActionSpec(len(actions), "set_access_vlan", switch, port, vlan))
    for switch in ("SW1", "SW2"):
        for vlan in (10, 20):
            for allowed in (False, True):
                actions.append(ActionSpec(len(actions), "set_trunk_vlan", switch, "eth3", vlan, allowed))
    actions.append(ActionSpec(len(actions), "rollback"))
    actions.append(ActionSpec(len(actions), "declare_done"))
    return tuple(actions)


ACTIONS = build_action_catalog()

