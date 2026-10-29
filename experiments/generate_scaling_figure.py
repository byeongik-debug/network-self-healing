"""Generate the paper scaling figure from the frozen runtime CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "results/expanded/scaling/runtime_scaling.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["topology"] for row in rows]
    x = range(len(rows))
    full = [int(row["full_actions"]) for row in rows]
    scoped = [int(row["policy_scoped_actions"]) for row in rows]
    shield = [float(row["shield_mask_ms"]) for row in rows]
    inference = [float(row["q_inference_ms"]) for row in rows]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), constrained_layout=True)
    axes[0].plot(x, full, marker="o", label="Full action catalog")
    axes[0].plot(x, scoped, marker="s", label="Policy-scoped ACL")
    axes[0].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[0].set_ylabel("Number of actions")
    axes[0].set_title("(a) Action-space growth")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(x, shield, marker="o", label="Counterfactual shield")
    axes[1].plot(x, inference, marker="s", label="Graph-Q inference")
    axes[1].set_yscale("log")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right")
    axes[1].set_ylabel("CPU wall time (ms, log scale)")
    axes[1].set_title("(b) Runtime bottleneck")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", which="both", alpha=0.25)
    axes[1].annotate(
        f"{shield[-1]:.1f} ms", (len(rows) - 1, shield[-1]),
        xytext=(-8, -16), textcoords="offset points", ha="right",
    )
    axes[1].annotate(
        f"{inference[-1]:.2f} ms", (len(rows) - 1, inference[-1]),
        xytext=(-8, 7), textcoords="offset points", ha="right",
    )

    output = ROOT / "paper/figures"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"expanded_scaling.{suffix}", dpi=220)
    plt.close(figure)
    print(output / "expanded_scaling.pdf")


if __name__ == "__main__":
    main()
