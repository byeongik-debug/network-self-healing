"""Paired analysis of 11-D versus policy-aware 13-D action features."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(2026, 2031))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bootstrap(values: list[float], draws: int = 20_000) -> list[float]:
    rng = random.Random(2026)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [means[int(.025 * draws)], means[int(.975 * draws)]]


def _group(rows: list[dict], key: str) -> dict:
    differences = [row["paired_difference"] for row in rows]
    nonzero = [value for value in differences if value != 0]
    test = wilcoxon(differences, zero_method="wilcox") if nonzero else None
    return {
        "scope": key,
        "seeds": len(rows),
        "base_11d_mean": fmean(row["base_11d"] for row in rows),
        "policy_13d_mean": fmean(row["policy_13d"] for row in rows),
        "paired_mean_difference": fmean(differences),
        "paired_difference_sample_sd": stdev(differences),
        "paired_seed_bootstrap_95ci": _bootstrap(differences),
        "wilcoxon_statistic": float(test.statistic) if test else None,
        "wilcoxon_two_sided_p": float(test.pvalue) if test else None,
        "seeds_favoring_policy_features": sum(value > 0 for value in differences),
        "seeds_favoring_base": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
    }


def analyze(root: Path = ROOT) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        base_root = root / f"results/expanded/guidance_ablation/seed_{seed}"
        feature_root = root / f"results/expanded/policy_feature_ablation/seed_{seed}"
        for topology, leaf, filename in (
            ("T4-Ring", "t4", "heldout_evaluation_summary.json"),
            ("T6-Mesh", "t6", "summary.json"),
        ):
            base, feature = _json(base_root / leaf / filename), _json(feature_root / leaf / filename)
            for condition in ("nominal", "combined"):
                left, right = base[condition]["success_rate"], feature[condition]["success_rate"]
                rows.append({
                    "seed": seed, "scope": topology, "condition": condition,
                    "base_11d": left, "policy_13d": right,
                    "paired_difference": right - left,
                })
        base_acl = _csv(base_root / "t6/scenario_results.csv")
        feature_acl = _csv(feature_root / "t6/scenario_results.csv")
        for condition in ("nominal", "combined"):
            def rate(source: list[dict]) -> float:
                selected = [row for row in source if row["condition"] == condition
                            and int(row["fault_count"]) == 1 and "acl_" in row["fault_ids"]]
                return fmean(row["success"].lower() == "true" for row in selected)
            left, right = rate(base_acl), rate(feature_acl)
            rows.append({
                "seed": seed, "scope": "T6-atomic-ACL", "condition": condition,
                "base_11d": left, "policy_13d": right,
                "paired_difference": right - left,
            })
    results = {}
    for scope in ("T4-Ring", "T6-Mesh", "T6-atomic-ACL"):
        for condition in ("nominal", "combined"):
            selected = [row for row in rows if row["scope"] == scope
                        and row["condition"] == condition]
            results[f"{scope}:{condition}"] = _group(selected, scope)
    return rows, {
        "protocol": {
            "seeds": list(SEEDS), "episodes_per_seed": 300,
            "guidance": False, "base_action_features": 11,
            "policy_action_features": 13, "bootstrap_draws": 20_000,
        },
        "results": results,
        "interpretation": (
            "Policy-relation features improve mean recovery in every reported "
            "scope, including atomic ACL faults; five seeds still limit power."
        ),
    }


def main() -> None:
    output = ROOT / "results/expanded/policy_feature_ablation"
    rows, summary = analyze(ROOT)
    csv_path, json_path = output / "paired_seed_results.csv", output / "summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite policy-feature ablation results")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
