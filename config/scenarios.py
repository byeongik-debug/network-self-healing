"""Fault catalogs kept separate for training and evaluation."""

from __future__ import annotations

from typing import Final


# A mutation describes what is applied to the healthy reference configuration.
TRAIN_SCENARIOS: Final[dict[str, tuple[dict[str, object], ...]]] = {
    "access_vlan_sw1": (
        {"kind": "access_vlan", "switch": "SW1", "port": "eth1", "value": 20},
    ),
    "trunk_missing_sw1": (
        {"kind": "trunk_remove", "switch": "SW1", "port": "eth3", "value": 10},
    ),
    "combined_train": (
        {"kind": "access_vlan", "switch": "SW1", "port": "eth2", "value": 10},
        {"kind": "trunk_remove", "switch": "SW2", "port": "eth3", "value": 20},
    ),
}

EVAL_SCENARIOS: Final[dict[str, tuple[dict[str, object], ...]]] = {
    "access_vlan_sw2": (
        {"kind": "access_vlan", "switch": "SW2", "port": "eth1", "value": 20},
    ),
    "trunk_missing_sw2": (
        {"kind": "trunk_remove", "switch": "SW2", "port": "eth3", "value": 10},
    ),
    "combined_eval": (
        {"kind": "access_vlan", "switch": "SW2", "port": "eth2", "value": 10},
        {"kind": "trunk_remove", "switch": "SW1", "port": "eth3", "value": 20},
    ),
}

