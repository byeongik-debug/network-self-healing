"""Paired five-seed analysis of reference-guided versus unguided training."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import wilcoxon

from experiments.analyze_expanded_multiseed import ROOT, collect


SEEDS = tuple(range(2026, 2031))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_paired(differences: list[float], draws: int = 20_000) -> list[float]:
    rng = random.Random(2026)
    estimates = sorted(fmean(rng.choices(differences, k=len(differences)))
                       for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[int(.975 * draws)]]


def analyze(root: Path = ROOT) -> tuple[list[dict], dict]:
    guided_rows, _ = collect(root)
    guided = {(row["seed"], row["topology"], row["condition"]): row
              for row in guided_rows}
    rows: list[dict] = []
    for seed in SEEDS:
        for topology, leaf, filename in (
            ("T4-Ring", "t4", "heldout_evaluation_summary.json"),
            ("T6-Mesh", "t6", "summary.json"),
        ):
            unguided = _load(root / f"results/expanded/guidance_ablation/seed_{seed}/{leaf}/{filename}")
            for condition in ("nominal", "combined"):
                baseline = guided[(seed, topology, condition)]
                rows.append({
                    "seed": seed, "topology": topology, "condition": condition,
                    "guided_success_rate": baseline["success_rate"],
                    "no_guidance_success_rate": unguided[condition]["success_rate"],
                    "paired_difference": unguided[condition]["success_rate"] - baseline["success_rate"],
                    "guided_damage": baseline["damage"],
                    "no_guidance_damage": unguided[condition]["damage"],
                })
    groups: dict[str, dict] = {}
    for topology in ("T4-Ring", "T6-Mesh"):
        for condition in ("nominal", "combined"):
            group = [row for row in rows if row["topology"] == topology
                     and row["condition"] == condition]
            differences = [row["paired_difference"] for row in group]
            test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            groups[f"{topology}:{condition}"] = {
                "seeds": len(group),
                "guided_mean": fmean(row["guided_success_rate"] for row in group),
                "no_guidance_mean": fmean(row["no_guidance_success_rate"] for row in group),
                "paired_mean_difference": fmean(differences),
                "paired_difference_sample_sd": stdev(differences),
                "paired_seed_bootstrap_95ci": _bootstrap_paired(differences),
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_two_sided_p": float(test.pvalue),
                "seeds_favoring_no_guidance": sum(value > 0 for value in differences),
                "seeds_favoring_guidance": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "total_guided_damage": sum(row["guided_damage"] for row in group),
                "total_no_guidance_damage": sum(row["no_guidance_damage"] for row in group),
            }
    return rows, {
        "scope": {
            "seeds": list(SEEDS), "paired": True,
            "training_episodes_per_seed": 300,
            "treatment": "reference-repair guidance disabled during training",
            "evaluation_guidance": False, "bootstrap_draws": 20_000,
        },
        "results": groups,
        "interpretation": (
            "No-guidance has a positive mean difference in all four conditions, "
            "but five paired seeds provide limited power; report intervals and "
            "individual seed directions alongside p-values."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/expanded/guidance_ablation")
    args = parser.parse_args()
    rows, summary = analyze(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
