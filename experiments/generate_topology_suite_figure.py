"""Render the T2--T6 topology suite for the manuscript."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt

from env.expanded.topologies import build_topology


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("T2", "T3", "T4-Line", "T4-Ring", "T5-Mesh", "T6-Mesh")
ROLES = {
    "T2": "train", "T3": "train", "T4-Line": "train",
    "T4-Ring": "held-out test", "T5-Mesh": "scaling only",
    "T6-Mesh": "zero-shot scaling",
}


def _positions(name: str, count: int) -> dict[str, tuple[float, float]]:
    if name in ("T2", "T4-Line"):
        return {f"SW{i + 1}": (i, 0.0) for i in range(count)}
    return {
        f"SW{i + 1}": (cos(pi / 2 - 2 * pi * i / count),
                       sin(pi / 2 - 2 * pi * i / count))
        for i in range(count)
    }


def main() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.0, 5.6))
    figure.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.08,
                           wspace=0.38, hspace=0.58)
    for axis, name in zip(axes.flat, NAMES):
        topology = build_topology(name)
        positions = _positions(name, len(topology.switches))
        for link in topology.links:
            x = [positions[link.left_switch][0], positions[link.right_switch][0]]
            y = [positions[link.left_switch][1], positions[link.right_switch][1]]
            axis.plot(x, y, color="0.45", linewidth=1.5,
                      linestyle="-" if link.nominal_enabled else "--")
        for switch in topology.switches:
            x, y = positions[switch.name]
            axis.scatter(x, y, s=380, marker="s", color="#2878B5", zorder=3)
            axis.text(x, y, switch.name.replace("SW", "S"), ha="center", va="center",
                      color="white", fontsize=8, fontweight="bold")
        axis.set_title(f"{name} — {ROLES[name]}", fontsize=10)
        axis.text(0.5, -0.10,
                  f"{len(topology.switches)} switches · {len(topology.hosts)} hosts · "
                  f"{len(topology.vlans)} VLANs · {len(topology.links)} links",
                  transform=axis.transAxes, ha="center", va="top", fontsize=8)
        if name in ("T2", "T4-Line"):
            axis.set_aspect("auto")
            axis.set_ylim(-0.55, 0.55)
        else:
            axis.set_aspect("equal")
        axis.axis("off")
        axis.margins(0.25)
    output = ROOT / "paper/figures"
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"topology_suite.{suffix}", dpi=220,
                       bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    print(output / "topology_suite.pdf")


if __name__ == "__main__":
    main()
