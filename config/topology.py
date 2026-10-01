"""Reference topology used by the pure-Python simulator."""

from __future__ import annotations

from typing import Final


HOSTS: Final[dict[str, dict[str, object]]] = {
    "H1": {"switch": "SW1", "port": "eth1", "vlan": 10},
    "H2": {"switch": "SW1", "port": "eth2", "vlan": 20},
    "H3": {"switch": "SW2", "port": "eth1", "vlan": 10},
    "H4": {"switch": "SW2", "port": "eth2", "vlan": 20},
}

REFERENCE_PORTS: Final[dict[str, dict[str, dict[str, object]]]] = {
    "SW1": {
        "eth1": {"mode": "access", "access_vlan": 10, "link_up": True},
        "eth2": {"mode": "access", "access_vlan": 20, "link_up": True},
        "eth3": {
            "mode": "trunk",
            "allowed_vlans": [10, 20],
            "link_up": True,
            "peer": "SW2:eth3",
        },
    },
    "SW2": {
        "eth1": {"mode": "access", "access_vlan": 10, "link_up": True},
        "eth2": {"mode": "access", "access_vlan": 20, "link_up": True},
        "eth3": {
            "mode": "trunk",
            "allowed_vlans": [10, 20],
            "link_up": True,
            "peer": "SW1:eth3",
        },
    },
}

