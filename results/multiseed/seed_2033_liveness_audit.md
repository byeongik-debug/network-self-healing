# Safe DQN seed 2033 liveness audit

Source: `multiseed_eval_scenario_results.csv` (frozen epsilon=0 evaluation)

## Observed failure

- Evaluation scenarios: 51
- Functionally recovered: 3
- Failed: 48
- Timed out at 30 steps: 49 (one timed-out episode was already functionally healthy)
- Actions in the 48 failed episodes: 1,440
- `diagnose` actions in failed episodes: 1,184 (82.22%)
- `rollback` actions in failed episodes: 90
- configuration actions in failed episodes: 166
- failures consisting of 30 consecutive `diagnose` actions: 6
- failures with at least 80% `diagnose` actions: 38/48

## Interpretation

The frozen result strongly supports **diagnose/self-loop Q dominance** as the
primary failure mode. It is not evidence of an empty shield mask: the recorded
training summary reports zero empty masks and finite Q-values/parameters. Some
episodes also exhibit rollback/configuration cycles, so a complete causal claim
requires the step-level Q-value and configuration-revisit diagnostic added in
`experiments/diagnose_liveness.py`.

Safety and liveness must not be conflated. The failed policy accumulates little
damage partly because it does not complete recovery. Damage comparisons must
therefore be reported jointly with recovery and conditioned on successful
recovery where appropriate.

## Reproduction

After repairing the Python environment:

```bash
python -m experiments.diagnose_liveness \
  --checkpoint results/multiseed/safe_seed_2033.pt \
  --output results/multiseed/safe_seed_2033_liveness_diagnostic.json
```

The generated JSON records, at every step, the selected action, top valid
action, all admissible Q-values, shield intervention, minimum damage cost,
successor cost, configuration identity, and visit count.
