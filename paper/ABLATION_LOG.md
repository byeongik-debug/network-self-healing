# Expanded-study ablation log

This log records development experiments that are informative but not eligible
for the manuscript's primary result table because they do not use matched
training budgets.

## Policy-relation action features, seed 2026

Date: 2026-10-27

Change: append two action features identifying whether an ACL host pair is a
required or explicitly forbidden policy relation. The original 11-dimensional
action representation is preserved as the default; this variant uses 13
dimensions.

Training budget: 100 episodes and 1,440 transitions, compared with 300 episodes
and 6,305 transitions for the current expanded baseline. Guided exploration and
all other trainer defaults were unchanged.

Training result:

- overall success: 76%;
- final 50-episode success: 56%;
- newly damaged policy events: 0.

Representative T6 zero-shot result:

| Condition | Recovery | Mean steps | New damage |
|---|---:|---:|---:|
| Nominal | 0/12 | 30.0 | 0 |
| Combined mismatch | 0/12 | 30.0 | 0 |

Interpretation: this short-run pilot does not support the hypothesis that two
policy-relation bits alone repair T6 transfer. It also cannot refute that
hypothesis because its transition budget is less than one quarter of the
baseline run and DQN learning was previously non-monotonic. The appropriate
next experiment is a matched-budget, multi-seed 11-versus-13-dimensional
ablation. These results must not be combined with the primary T6 table.

Artifacts:

- `results/expanded/graph_dqn/model_seed_2026_ep100_policy.pt`
- `results/expanded/graph_dqn/training_seed_2026_ep100_policy.csv`
- `results/expanded/t6_zero_shot_policy_ep100/scenario_results.csv`
- `results/expanded/t6_zero_shot_policy_ep100/summary.json`

## Reference-guidance removal, matched budget

Date: 2026-10-27

Change: set both training guidance probabilities to zero. Epsilon exploration,
shielding, topology/scenario pool, architecture, replay settings, and the
300-episode budget are unchanged. Evaluation uses no guidance in both groups.

Completed paired seeds: 2026--2030 (5/5 planned).

| Evaluation | Guided mean | No-guidance mean | Paired difference |
|---|---:|---:|---:|
| T4-Ring nominal | 69.09% | 93.45% | +24.36 pp |
| T4-Ring combined | 49.09% | 90.55% | +41.45 pp |
| T6-Mesh nominal | 65.00% | 88.33% | +23.33 pp |
| T6-Mesh combined | 48.33% | 83.33% | +35.00 pp |

All training and evaluation runs recorded zero new policy-damage events. Most
notably, guided seed 2028 collapsed (T4 nominal 0%; both T6 conditions 0%),
whereas its matched no-guidance run reached 98.18% T4 recovery and
91.67%/83.33% T6 recovery. No-guidance has a positive mean paired difference
in all four conditions, but two-sided Wilcoxon p-values range from 0.1875 to
0.4375 and both T6 seed-bootstrap intervals include zero. Guidance is therefore
not necessary in this sample, but a population-level causal harm is not shown.

Artifacts are under `results/expanded/guidance_ablation/seed_2026/` through
`seed_2030/`, and
the corresponding `_noguidance` checkpoints/training logs are under
`results/expanded/graph_dqn/`.

## Policy-relation action features, matched budget

Date: 2026-10-27

Five paired seeds compare the 11-dimensional action representation with a
13-dimensional variant containing required/forbidden policy-relation bits.
Both groups use zero reference guidance and 300 training episodes.

| Evaluation | 11-D mean | 13-D mean | Paired difference | Bootstrap 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| T4-Ring nominal | 93.45% | 99.64% | +6.18 pp | [2.18, 10.18] pp | 0.125 |
| T4-Ring combined | 90.55% | 99.09% | +8.55 pp | [3.45, 12.91] pp | 0.125 |
| T6-Mesh nominal | 88.33% | 98.33% | +10.00 pp | [-1.67, 23.33] pp | 0.3125 |
| T6-Mesh combined | 83.33% | 96.67% | +13.33 pp | [3.33, 23.33] pp | 0.25 |
| T6 atomic ACL nominal | 50.00% | 90.00% | +40.00 pp | [-10, 80] pp | 0.3125 |
| T6 atomic ACL combined | 40.00% | 80.00% | +40.00 pp | [-10, 90] pp | 0.375 |

All groups record zero new policy damage. Policy features improve the mean in
every scope, but one seed worsens on each atomic-ACL condition and both ACL
intervals include zero. The original representation hypothesis is supported as
a plausible contributor, not established as the sole cause of ACL failure.

Artifacts:

- `results/expanded/policy_feature_ablation/summary.json`
- `results/expanded/policy_feature_ablation/paired_seed_results.csv`
