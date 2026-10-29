# Six-switch scalability study

Frozen on 2026-10-27. All commands are run from the repository root with the
project virtual environment.

## Scope

- Training topologies: T2, T3, T4-Line.
- Held-out full evaluation: T4-Ring, 110 scenarios per condition.
- Scaling-only topology: T5-Mesh.
- Representative zero-shot topology: T6-Mesh, 12 scenarios per condition.
- Maximum modeled size: 6 switches, 24 hosts, 6 VLANs, 9 inter-switch links.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_expanded_scaling --repeats 3
.\.venv\Scripts\python.exe -m experiments.generate_scaling_figure
.\.venv\Scripts\python.exe -m experiments.generate_topology_suite_figure
.\.venv\Scripts\python.exe -m experiments.evaluate_t6_zero_shot `
  --checkpoint results\expanded\graph_dqn\model_seed_2026_ep300.pt
.\.venv\Scripts\python.exe -m experiments.diagnose_t6_acl_failures `
  --checkpoint results\expanded\graph_dqn\model_seed_2026_ep300.pt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m experiments.build_artifact_manifest
```

## Frozen findings

| Item | T2 | T6-Mesh |
|---|---:|---:|
| Full action catalog | 44 | 1,016 |
| Policy-scoped catalog | 34 | 468 |
| Graph encoding | 0.08 ms | 0.41 ms |
| Graph-Q inference | 1.77 ms | 1.67 ms |
| Uncached shield mask | 0.24 ms | 80.48 ms |

T6 representative zero-shot recovery is 10/12 (83.33%) in both nominal and
combined mismatch conditions, with zero newly damaged policy events. Eight
cases are policy-impacting and four are no-impact redundancy controls; recovery
is 6/8 (75%) and 4/4 (100%), respectively. The two failures are the atomic ACL
cases. Their correct repair action initially ranks 136th among shield-admissible
actions.

The runtime numbers are three-repeat, single-process CPU wall-clock
measurements. They locate the current bottleneck but must not be presented as a
hardware-neutral benchmark. The T6 manifest is representative, not exhaustive.
The shield uses exact action-local dependency pruning and its masks were checked
against the independent clone-per-action definition across T2--T6 and both
safety modes. T6 latency is 90.5% below the original 846.76 ms implementation.

## Inconclusive development ablation

A 100-episode policy-feature variant recovered 0/12 T6 cases in both conditions.
Its training budget was not matched to the 300-episode baseline, so it is logged
in `paper/ABLATION_LOG.md` and excluded from the main result table.
