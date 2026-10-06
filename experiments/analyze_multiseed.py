"""Evaluate and statistically analyze the frozen 10-seed experiment."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import numpy as np
from scipy.stats import rankdata, wilcoxon

from experiments.evaluate_dqn import DQNEvaluator, EvaluationConfig, summarize_rows
from experiments.evaluate_shielded_dqn import ShieldedDQNEvaluator


ROOT = Path("results/multiseed")
SEEDS = tuple(range(2026, 2036))


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    training = []
    for algorithm in ("vanilla", "safe"):
        for seed in SEEDS:
            training.append(json.loads((ROOT / f"{algorithm}_seed_{seed}_training_summary.json").read_text()))
    save_csv(ROOT / "multiseed_training_summary.csv", training)

    raw, seed_summaries, trajectories = [], [], {}
    for algorithm in ("vanilla", "safe"):
        for seed in SEEDS:
            checkpoint = ROOT / f"{algorithm}_seed_{seed}.pt"
            config = EvaluationConfig(checkpoint_path=str(checkpoint))
            started = perf_counter()
            if algorithm == "safe":
                evaluated = ShieldedDQNEvaluator(config).evaluate(save=False)
            else:
                evaluated = DQNEvaluator(config).evaluate(save=False)
            runtime = perf_counter() - started
            trajectories[f"{algorithm}_{seed}"] = evaluated["trajectories"]
            rows = evaluated["rows"]
            overall = summarize_rows(rows)
            total_damage = sum(r.new_policy_damage_events for r in rows)
            total_minimum = sum(getattr(r, "theoretical_minimum_damage", 0) for r in rows)
            seed_summaries.append({
                "algorithm": algorithm, "seed": seed,
                "functional_recovery": overall["functional_recovery_rate"],
                "configuration_recovery": overall["configuration_recovery_rate"],
                "recovered_scenarios": sum(r.recovery_success for r in rows),
                "failed_scenarios": sum(not r.recovery_success for r in rows),
                "total_damage": total_damage,
                "mean_damage": overall["mean_new_policy_damage_events"],
                "mean_steps": overall["mean_steps"], "p95_steps": overall["p95_steps"],
                "mean_return": overall["mean_return"],
                "incorrect_declare": overall["incorrect_declare_rate"],
                "timeout": overall["timeout_rate"],
                "theoretical_minimum_damage": total_minimum if algorithm == "safe" else "",
                "safety_optimality_gap": total_damage-total_minimum if algorithm == "safe" else "",
                "evaluation_runtime_seconds": runtime,
            })
            for row in rows:
                item = asdict(row)
                item.update({"algorithm": algorithm, "seed": seed,
                    "minimum_required_damage": getattr(row,"theoretical_minimum_damage", ""),
                    "safety_optimality_gap": (row.new_policy_damage_events-
                        getattr(row,"theoretical_minimum_damage", row.new_policy_damage_events)),
                    "shield_interventions": getattr(row,"shield_interventions", ""),
                    "relaxations": getattr(row,"shield_relaxations", "")})
                raw.append(item)
    fields = ["algorithm","seed","scenario_id","fault_type","severity","fault_count",
        "recovery_success","configuration_recovery","steps","episode_return",
        "incorrect_declare","timeout","new_policy_damage_events",
        "unique_newly_damaged_policies","service_damage_actions",
        "post_recovery_damage_events","minimum_required_damage","safety_optimality_gap",
        "shield_interventions","relaxations","action_sequence"]
    save_csv(ROOT / "multiseed_eval_scenario_results.csv",
             [{key: row.get(key, "") for key in fields} for row in raw])
    save_csv(ROOT / "multiseed_eval_seed_summary.csv", seed_summaries)

    aggregate = {}
    for algorithm in ("vanilla", "safe"):
        selected = [r for r in seed_summaries if r["algorithm"] == algorithm]
        aggregate[algorithm] = {
            key: stats([r[key] for r in selected]) for key in (
                "functional_recovery","total_damage","mean_damage","mean_steps",
                "p95_steps","mean_return","incorrect_declare","timeout",
                "configuration_recovery")}
        aggregate[algorithm]["perfect_recovery_seeds"] = sum(r["recovered_scenarios"] == 51 for r in selected)
        aggregate[algorithm]["worst_seed"] = min(selected, key=lambda r:r["recovered_scenarios"])
    safe_selected = [r for r in seed_summaries if r["algorithm"] == "safe"]
    aggregate["safe"]["safety_optimality_gap"] = stats([r["safety_optimality_gap"] for r in safe_selected])
    aggregate["safe"]["zero_gap_seeds"] = sum(r["safety_optimality_gap"] == 0 for r in safe_selected)

    by_key = {(r["algorithm"],r["seed"]):r for r in seed_summaries}
    differences = [by_key[("vanilla",s)]["total_damage"]-
                   by_key[("safe",s)]["total_damage"] for s in SEEDS]
    test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided",
                    method="auto")
    nonzero = np.asarray([d for d in differences if d != 0], dtype=float)
    ranks = rankdata(abs(nonzero)) if len(nonzero) else np.asarray([])
    rank_biserial = (float(ranks[nonzero>0].sum()-ranks[nonzero<0].sum()) /
                     float(ranks.sum())) if len(ranks) else 0.0

    failures = {}
    for algorithm in ("vanilla", "safe"):
        failed = [r for r in raw if r["algorithm"] == algorithm and not r["recovery_success"]]
        counts = Counter(r["scenario_id"] for r in failed)
        failures[algorithm] = {"total_failed_evaluations":len(failed),
            "scenario_frequency": dict(counts.most_common())}

    scenario_analysis = []
    scenario_ids = sorted({r["scenario_id"] for r in raw})
    for scenario in scenario_ids:
        v = [r for r in raw if r["algorithm"]=="vanilla" and r["scenario_id"]==scenario]
        s = [r for r in raw if r["algorithm"]=="safe" and r["scenario_id"]==scenario]
        vd, sd = fmean(r["new_policy_damage_events"] for r in v), fmean(r["new_policy_damage_events"] for r in s)
        scenario_analysis.append({"scenario_id":scenario,"severity":v[0]["severity"],
            "fault_type":v[0]["fault_type"],"vanilla_mean_damage":vd,
            "safe_mean_damage":sd,"vanilla_minus_safe":vd-sd})

    severity = {}
    for algorithm in ("vanilla","safe"):
        severity[algorithm] = {}
        for level in ("low","medium","high"):
            selected=[r for r in raw if r["algorithm"]==algorithm and r["severity"]==level]
            severity[algorithm][level]={"count":len(selected),
                "recovery":fmean(r["recovery_success"] for r in selected),
                "mean_damage":fmean(r["new_policy_damage_events"] for r in selected),
                "mean_steps":fmean(r["steps"] for r in selected)}

    gap_cases=[]
    for row in raw:
        if row["algorithm"]=="safe" and row["safety_optimality_gap"]>0:
            trajectory=next(t for t in trajectories[f"safe_{row['seed']}"]
                            if t["scenario_id"]==row["scenario_id"])
            gap_cases.append({"seed":row["seed"],"scenario_id":row["scenario_id"],
                "damage":row["new_policy_damage_events"],
                "minimum":row["minimum_required_damage"],"trajectory":trajectory})

    tr_by_alg={a:[r for r in training if r["algorithm"]==a] for a in ("vanilla","safe")}
    summary={"seeds":list(SEEDS),"frozen_transition_budget":35747,
        "aggregate":aggregate,
        "safe_gap_interpretation": {
            "definition": "realized total damage - theoretical recovery minimum (9)",
            "warning": "A negative value is an unrecovered-damage deficit, not super-optimal safety.",
            "recovery_conditioned_zero_gap_seeds": sum(
                r["recovered_scenarios"] == 51 and r["safety_optimality_gap"] == 0
                for r in safe_selected),
            "noncomparable_failed_seeds": [r["seed"] for r in safe_selected
                if r["recovered_scenarios"] < 51],
        },
        "paired_damage":{"definition":"Vanilla total damage - Safe total damage",
            "differences":differences,"wilcoxon_statistic":float(test.statistic),
            "p_value":float(test.pvalue),"rank_biserial_correlation":rank_biserial},
        "failures":failures,"scenario_damage":scenario_analysis,
        "severity":severity,"safe_positive_gap_cases":gap_cases,
        "runtime":{"vanilla_training":stats([r["training_runtime_seconds"] for r in tr_by_alg["vanilla"]]),
                   "safe_training":stats([r["training_runtime_seconds"] for r in tr_by_alg["safe"]]),
                   "shield_overhead_ratio_of_means":fmean(r["training_runtime_seconds"] for r in tr_by_alg["safe"])/fmean(r["training_runtime_seconds"] for r in tr_by_alg["vanilla"]),
                   "vanilla_eval":stats([r["evaluation_runtime_seconds"] for r in seed_summaries if r["algorithm"]=="vanilla"]),
                   "safe_eval":stats([r["evaluation_runtime_seconds"] for r in seed_summaries if r["algorithm"]=="safe"])},
        "integrity":{"all_exact_budget":all(r["global_steps"]==35747 for r in training),
            "all_optimizer_updates_34748":all(r["optimizer_updates"]==34748 for r in training),
            "all_eval_exposure_zero":all(r["eval_training_exposure"]==0 for r in training),
            "all_losses_finite":all(r["finite_loss"] for r in training),
            "all_parameters_finite":all(r["finite_parameters"] for r in training),
            "all_q_finite":all(r["finite_q"] for r in training),
            "safe_viability_violations":sum(r["viability_violations"] for r in tr_by_alg["safe"]),
            "safe_empty_masks":sum(r["empty_masks"] for r in tr_by_alg["safe"])}}
    repro_path = ROOT / "multiseed_reproducibility.json"
    if repro_path.exists():
        summary["reproducibility"] = json.loads(repro_path.read_text(encoding="utf-8"))
    (ROOT/"multiseed_statistical_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"aggregate":aggregate,"paired_damage":summary["paired_damage"],
                      "failures":failures,"integrity":summary["integrity"],
                      "runtime":summary["runtime"]},indent=2))


if __name__ == "__main__": main()
