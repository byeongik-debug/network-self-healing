"""Deterministic numeric encoding of the existing public observation only."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray


PORT_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("SW1", "eth1"), ("SW1", "eth2"), ("SW1", "eth3"),
    ("SW2", "eth1"), ("SW2", "eth2"), ("SW2", "eth3"),
)
HOST_ORDER: Final[tuple[str, ...]] = ("H1", "H2", "H3", "H4")
DIAGNOSTIC_ORDER: Final[tuple[str, ...]] = ("unknown", "success", "failure")


class ObservationEncoder:
    """Encode ports, links, diagnostics, and step count into 79 float32 values."""

    PORT_FEATURES: Final[tuple[str, ...]] = (
        "mode_access", "mode_trunk", "access_vlan_10", "access_vlan_20",
        "trunk_allows_10", "trunk_allows_20",
    )
    output_size: Final[int] = 79

    def encode(self, observation: dict[str, Any]) -> NDArray[np.float32]:
        values: list[float] = []
        for switch, port in PORT_ORDER:
            settings = observation["ports"][switch][port]
            mode = settings.get("mode")
            access_vlan = settings.get("access_vlan")
            allowed = settings.get("allowed_vlans", [])
            values.extend((
                float(mode == "access"), float(mode == "trunk"),
                float(access_vlan == 10), float(access_vlan == 20),
                float(10 in allowed), float(20 in allowed),
            ))
        for switch, port in PORT_ORDER:
            values.append(float(bool(observation["link_states"][f"{switch}:{port}"])))
        for source in HOST_ORDER:
            for destination in HOST_ORDER:
                if destination == source:
                    continue
                status = observation["connectivity"][source][destination]
                if status not in DIAGNOSTIC_ORDER:
                    raise ValueError(f"unknown diagnostic status: {status!r}")
                values.extend(float(status == candidate) for candidate in DIAGNOSTIC_ORDER)
        values.append(float(observation["step_count"]))
        encoded = np.asarray(values, dtype=np.float32)
        if encoded.shape != (self.output_size,):
            raise ValueError(f"expected {self.output_size} encoded values, got {encoded.shape}")
        return encoded

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        names = [
            f"port.{switch}.{port}.{feature}"
            for switch, port in PORT_ORDER
            for feature in cls.PORT_FEATURES
        ]
        names.extend(f"link.{switch}.{port}.up" for switch, port in PORT_ORDER)
        names.extend(
            f"connectivity.{source}.{destination}.{status}"
            for source in HOST_ORDER
            for destination in HOST_ORDER
            if source != destination
            for status in DIAGNOSTIC_ORDER
        )
        names.append("step_count")
        return tuple(names)
