# Expanded experimental environment specification

Status: design freeze candidate, 2026-10-26. This document covers all six
requested extensions. Implementation proceeds in reviewable stages; no result is
considered complete until its acceptance gate is met.

## 1. Research questions

- **RQ1 — Scale:** Does safety/liveness performance persist as topology, host,
  VLAN, trunk, and action counts increase?
- **RQ2 — Fault breadth:** Does the method recover heterogeneous data-plane and
  policy faults rather than only VLAN configuration mutations?
- **RQ3 — Imperfect execution and observation:** How does it degrade under
  command loss, delayed telemetry, and stale telemetry?
- **RQ4 — Topology generalization:** Can a policy trained without the target
  topology recover faults on an unseen topology?
- **RQ5 — Emulator validity:** Do simulator recovery, collateral damage, and
  action effects agree with Mininet/Open vSwitch measurements?

## 2. Topology suite

Topology identities and train/evaluation membership are immutable once results
begin.

| ID | Switches | Hosts | VLANs | Physical graph | Use |
|---|---:|---:|---:|---|---|
| T2 | 2 | 4 | 2 | one inter-switch link | Legacy compatibility and exact-shield reference |
| T3 | 3 | 8 | 3 | triangle, one link disabled in the nominal spanning tree | Training and in-family evaluation |
| T4-Line | 4 | 16 | 4 | line | Training and in-family evaluation |
| T4-Ring | 4 | 16 | 4 | ring | **Held-out topology generalization** |
| T5-Mesh | 5 | 20 | 5 | ring plus two chords | Scaling runtime only |
| T6-Mesh | 6 | 24 | 6 | ring plus three chords | **Representative zero-shot scaling** |

Host placement is deterministic and balanced as far as integer counts permit.
Each VLAN has hosts on at least two switches, so trunk faults have observable
cross-switch consequences. Switch and host ordering in tensors is canonical and
stored in each topology manifest.

### Generalization split

- Train: T2, T3, and T4-Line fault scenarios.
- Validation: held-out combinations on the training topologies.
- Test: T4-Ring only; no T4-Ring transitions enter replay, normalization, model
  selection, or threshold tuning.
- A second leave-one-topology-out rotation is run if compute permits; otherwise
  the limitation is explicit.
- T5-Mesh and T6-Mesh remain outside replay and model selection. T6 evaluation
  is reported separately as a representative 12-scenario feasibility check,
  not as part of the exhaustive T4 test manifest.

## 3. Fault families

Every injected fault has a stable canonical ID and a reversible reference
operation.

| Family | Mutation | Recovery actions | Safety concern |
|---|---|---|---|
| Access VLAN | wrong access VLAN | assign VLAN | temporary tenant exposure/isolation |
| Trunk membership | missing or excess allowed VLAN | allow/remove VLAN | cross-switch isolation/exposure |
| Link down | inter-switch carrier unavailable | enable link or reroute | loss of reachability |
| Port shutdown | access or trunk admin-down | enable port | host or segment outage |
| ACL error | required allow removed or forbidden pair allowed | add/remove ACL rule | reachability or isolation violation |

Scenario strata are `(topology, family set, fault count, functional severity)`.
The initial study uses one- to three-fault combinations for all families and a
balanced sample of four- to six-fault combinations. Exhaustive enumeration is
retained only where tractable. Train/test scenario IDs must be disjoint.

ACL policy is defined on unordered endpoint groups:

- same-tenant pairs must be reachable;
- different-tenant pairs must be isolated;
- explicit service exceptions may permit selected cross-tenant flows.

Both loss of required reachability and accidental forbidden reachability count
as policy damage.

## 4. Imperfect execution and telemetry

Four independently controlled mismatch dimensions are evaluated:

| Variable | Levels |
|---|---|
| Configuration-command no-op probability | 0, 0.1, 0.2, 0.3 |
| Command actuation delay | 0, 1, 2, 4 environment steps |
| Telemetry delay | 0, 1, 2, 4 steps |
| Stale-observation probability | 0, 0.1, 0.2, 0.3 |

The full Cartesian product is not used. One-factor sweeps establish sensitivity,
followed by a prespecified stress matrix: `(no-op, telemetry delay, stale)` of
`(0.1,1,0.1)`, `(0.2,2,0.2)`, and `(0.3,4,0.3)`. Random streams for scenario
selection, action execution, and telemetry are independent and recorded.

The existing 20,400-run no-op study is the first completed slice. Delayed actions
must be represented as pending operations in the observation; otherwise the
agent is unfairly denied observable system state.

## 5. Scalable state and action representation

The current 79-input/19-output MLP cannot evaluate an unseen topology and will
remain only as the T2 baseline.

The expanded policy uses a graph representation:

- switch nodes: admin state, incident-link summary, per-VLAN membership summary;
- host/access-port nodes: admin state, assigned VLAN, diagnostic status;
- trunk edges: link/admin state and per-VLAN allow bits;
- policy-pair features: required reachability/isolation and latest diagnostic;
- global features: normalized step count, pending-command counts, telemetry age.

A shared message-passing encoder produces node/edge embeddings. A shared action
scorer evaluates parameterized candidates `(operation, target, VLAN/policy)`.
This avoids topology-specific output dimensions and permits T4-Ring inference
without changing network weights. Invalid candidates remain masked.

### Mandatory anti-leakage rule

Node identifiers, scenario IDs, fault labels, injected mutation identities, and
reference-repair actions are excluded from policy inputs. Canonical ordering may
be used for batching but not as an implicit target label.

## 6. Safety and progress mechanisms at scale

The 256-state explicit shield is retained as an exact oracle on T2 but is not
constructed for T3/T4. Expanded experiments compare:

1. Graph policy with validity mask only.
2. Graph policy with one-step invariant shield.
3. Graph policy with bounded-horizon safety lookahead.
4. Graph policy with safety lookahead plus a progress ranking.
5. Privileged reference planner as an upper-bound baseline, not a learned method.

The scalable shield rejects actions that violate a currently healthy required
policy or create a forbidden reachable pair, unless a bounded recovery proof
shows that the damage is unavoidable. Progress is measured by a lexicographic
tuple `(policy violations, unresolved diagnostic components, estimated repair
distance, pending commands)` and must decrease within a finite receding horizon.

This is an empirical bounded guarantee. Only the T2 explicit graph supports the
current exact global guarantee; the paper must keep those claims separate.

## 7. Baselines and ablations

- Rule-based diagnosis/recovery.
- Privileged configuration-difference planner.
- Valid-mask MLP DQN on T2.
- Damage-only exact Safe DQN on T2.
- Exact Progress-Safe DQN on T2.
- Graph policy without shield.
- Graph policy with safety only.
- Graph policy with safety and progress.
- Progress mechanism with learned scores replaced by random admissible choice.
- Progress mechanism with shortest-path planner replacing learning.

The last two ablations determine whether learning adds value beyond the shield.

## 8. Metrics and statistical protocol

Primary metrics:

- functional recovery and successful declaration;
- required-connectivity and forbidden-connectivity violations;
- newly damaged policy events and unique damaged policies;
- timeout and recovery steps;
- action failures, pending-action conflicts, and stale decisions;
- shield interventions and false-positive intervention rate;
- wall-clock decision latency and peak shield memory.

Report macro averages across topology and fault-family strata so large strata do
not dominate. Use paired scenario/seed evaluation, at least 10 training seeds,
bootstrap 95% confidence intervals, paired effect sizes, and family-wise error
control for confirmatory comparisons. Failed recovery is always reported jointly
with damage.

## 9. Mininet/Open vSwitch validation

Mininet implements T2 first and then T3. Representative validation set:

- two scenarios per atomic fault family;
- three mixed two-fault scenarios;
- three high-severity scenarios;
- one command-failure and one delayed-telemetry scenario.

For each scenario, compare simulator and emulator at every action:

- applied configuration;
- required and forbidden connectivity matrix;
- newly damaged policies;
- final recovery and declaration;
- action latency.

Minimum acceptance gate is exact outcome agreement on all representative
scenarios. Any mismatch is retained and analyzed rather than removed.

## 10. Implementation stages and stop points

Each stage ends with a short review before the next begins.

1. **Schema:** topology, policy, fault, and dynamic action data models.
2. **General simulator:** T2 parity, then T3/T4 construction and fault execution.
3. **Imperfect environment:** no-op, pending commands, telemetry delay/staleness.
4. **Graph observation/action API:** dimension-independent encoding and masks.
5. **Baselines:** rule, planner, random-progress, and shortest-progress.
6. **Graph learner:** shared encoder/action scorer and replay/training pipeline.
7. **Scale shield:** one-step, bounded lookahead, and progress variants.
8. **Experiment freeze:** manifests, splits, budgets, seeds, and metrics.
9. **Main runs:** in-topology, held-out topology, mismatch sweeps, ablations.
10. **Mininet:** T2 parity and T3 representative replay.
11. **Paper update:** statistics, plots, claims, limitations, and artifact release.

## 11. Acceptance gates

- Legacy T2 results remain reproducible and all existing tests pass.
- Simulator has unit tests for every topology and fault family.
- T4-Ring is absent from every training artifact and present only in test output.
- T5/T6 are absent from training artifacts; T6 claims say "representative"
  unless the full atomic and combination manifest is later executed.
- No policy input contains fault or reference-repair labels.
- Every reported shield guarantee names whether it is exact or bounded.
- Mismatch runs save the random streams and scenario-level outcomes.
- Mininet claims require an actual timestamped result artifact, not only code.
- Paper tables and figures are regenerated from frozen raw results in one command.
