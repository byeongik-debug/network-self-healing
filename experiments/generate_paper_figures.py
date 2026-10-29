"""Generate deterministic paper figures from frozen result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results")
OUT = Path("paper/figures")
COLORS = {"vanilla": "#4C78A8", "safe": "#F58518", "progress_safe": "#54A24B"}
LABELS = {"vanilla": "Vanilla DQN", "safe": "Damage-only Safe DQN", "progress_safe": "Progress-Safe DQN"}


def load_rows() -> list[dict[str, str]]:
    with (ROOT / "fresh_progress_comparison.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main_comparison(rows: list[dict[str, str]]) -> None:
    algorithms = ("vanilla", "safe", "progress_safe")
    metrics = (
        ("functional_recovery", "Functional recovery", 100.0),
        ("total_damage", "Total collateral damage", 1.0),
        ("timeout", "Timeout rate", 100.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))
    for axis, (metric, title, scale) in zip(axes, metrics):
        values = [
            [float(row[metric]) * scale for row in rows if row["algorithm"] == algorithm]
            for algorithm in algorithms
        ]
        positions = np.arange(len(algorithms))
        means = [np.mean(value) for value in values]
        rng = np.random.default_rng(2026)
        intervals = []
        for samples in values:
            sample_array = np.asarray(samples, dtype=float)
            boot = rng.choice(
                sample_array, size=(20_000, len(sample_array)), replace=True
            ).mean(axis=1)
            intervals.append(np.quantile(boot, (0.025, 0.975)))
        lower = [mean - interval[0] for mean, interval in zip(means, intervals)]
        upper = [interval[1] - mean for mean, interval in zip(means, intervals)]
        axis.bar(
            positions, means, yerr=np.asarray([lower, upper]), capsize=4,
            color=[COLORS[a] for a in algorithms], alpha=0.82,
        )
        for x, samples in zip(positions, values):
            jitter = np.linspace(-0.09, 0.09, len(samples))
            axis.scatter(x + jitter, samples, s=15, color="black", alpha=0.65, zorder=3)
        axis.set_title(title)
        axis.set_xticks(positions, [LABELS[a] for a in algorithms], rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if metric in ("functional_recovery", "timeout"):
            axis.set_ylim(0, 105)
    axes[0].set_ylabel("Percent")
    axes[2].set_ylabel("Percent")
    save(fig, "main_comparison")


def seed_profile(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True)
    for algorithm in ("vanilla", "safe", "progress_safe"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        seeds = [int(row["seed"]) for row in selected]
        axes[0].plot(
            seeds, [100 * float(row["functional_recovery"]) for row in selected],
            marker="o", label=LABELS[algorithm], color=COLORS[algorithm],
        )
        axes[1].plot(
            seeds, [float(row["total_damage"]) for row in selected],
            marker="o", label=LABELS[algorithm], color=COLORS[algorithm],
        )
    axes[0].set_ylabel("Recovery (%)")
    axes[1].set_ylabel("Total damage")
    axes[1].set_xlabel("Training seed")
    axes[0].axvline(2033, color="#E45756", linestyle="--", alpha=0.55)
    axes[0].annotate("liveness failure", (2033, 8), xytext=(2030.4, 30),
                     arrowprops={"arrowstyle": "->", "color": "#E45756"})
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    save(fig, "per_seed_profile")


def liveness_trace() -> None:
    diagnostic = json.loads(
        (ROOT / "multiseed/safe_seed_2033_liveness_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    failures = [episode for episode in diagnostic["episodes"] if not episode["recovery_success"]]
    diagnose_fraction = [episode["diagnose_actions"] / episode["steps"] for episode in failures]
    revisits = [episode["max_configuration_revisits"] for episode in failures]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].hist(diagnose_fraction, bins=np.linspace(0, 1, 11), color=COLORS["safe"], edgecolor="white")
    axes[0].set_xlabel("Fraction of diagnose actions")
    axes[0].set_ylabel("Failed scenarios")
    axes[1].hist(revisits, bins=range(2, 32), color="#E45756", edgecolor="white")
    axes[1].set_xlabel("Maximum visits to one configuration")
    axes[1].set_ylabel("Failed scenarios")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save(fig, "seed2033_liveness_failure")


def model_mismatch() -> None:
    report = json.loads(
        (ROOT / "model_mismatch_summary.json").read_text(encoding="utf-8")
    )
    probabilities = [float(value) for value in report["design"]["failure_probabilities"]]
    aggregate = report["aggregate"]
    means = [float(aggregate[str(value)]["mean_steps"]) for value in probabilities]
    p95 = [float(aggregate[str(value)]["p95_steps"]) for value in probabilities]
    fig, axis = plt.subplots(figsize=(5.4, 3.3))
    x = [100 * value for value in probabilities]
    axis.plot(x, means, marker="o", label="Mean steps", color=COLORS["progress_safe"])
    axis.plot(x, p95, marker="s", label="P95 steps", color="#E45756")
    axis.set_xlabel("Configuration-command failure probability (%)")
    axis.set_ylabel("Recovery steps")
    axis.set_xticks(x)
    axis.set_ylim(0, 30)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.text(29, 26.5, "20,400/20,400 recovered\n0 additional damage",
              ha="right", va="top", fontsize=9)
    save(fig, "model_mismatch")


def main() -> None:
    rows = load_rows()
    main_comparison(rows)
    seed_profile(rows)
    liveness_trace()
    model_mismatch()
    manifest = {
        "figures": [
            "main_comparison.pdf", "per_seed_profile.pdf",
            "seed2033_liveness_failure.pdf", "model_mismatch.pdf",
        ],
        "source_artifacts": [
            "results/fresh_progress_comparison.csv",
            "results/multiseed/safe_seed_2033_liveness_diagnostic.json",
            "results/model_mismatch_summary.json",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
