# Expanded-topology study protocol

Status: preliminary protocol freeze, 2026-10-27. This document separates the
new scalability study from the completed two-switch exact-state experiment.

## Research question

Can a single graph-based action-value model trained on T2, T3, and T4-Line
topologies recover faults on the structurally unseen T4-Ring topology while a
runtime shield prevents newly damaged service policies under command failure,
actuation delay, and stale telemetry?

## Fixed topology split

- Training/validation topology families: T2, T3, T4-Line.
- Held-out test topology: T4-Ring only.
- The test topology must not be used for training, replay warmup, model
  selection, early stopping, or hyperparameter selection.
- Atomic faults and at most 32 deterministic two-fault combinations are
  generated per topology. Scenarios that remain policy-healthy because of a
  redundant route are retained but reported separately as no-impact controls.

## Methods

- Graph DQN with shared message-passing parameters and variable action scoring.
- Privileged configuration-difference oracle as a recovery upper bound.
- One-step counterfactual shield using violation-set inclusion, not merely a
  non-increasing violation count.
- Pending-aware serialization: while an actuation is outstanding, only
  diagnosis is admissible. This prevents individually safe delayed commands
  from becoming harmful in combination.

Guided exploration may be used during training and must be reported as the
fraction of environment actions supplied by the reference repair policy. It is
disabled during every reported evaluation.

## Evaluation conditions

1. Nominal deterministic execution.
2. Combined mismatch: command failure probability 0.20, actuation delay two
   steps, telemetry delay two steps, and stale-observation probability 0.50.

Each checkpoint is evaluated greedily (`epsilon=0`) with a 30-step limit. The
full held-out manifest contains 110 T4-Ring scenarios per condition.

## Outcomes

Primary outcomes:

- functional recovery and successful declaration;
- newly damaged service-policy events;
- timeout rate.

Secondary outcomes:

- recovery steps;
- results split between impacting faults and no-impact controls;
- shield computation time and intervention rate;
- learning stability across 50-episode windows.

Scenario bootstrap confidence intervals describe uncertainty over this fixed
manifest. They are not population guarantees. The final paper requires at least
five independently initialized training seeds; the current seed-2026 result is
preliminary and cannot support a stability claim.

## Claim boundary

The expanded experiment can support empirical generalization to one held-out
topology under the stated simulator and mismatch model. It cannot establish
general network-scale safety, hardware transfer, or a formal liveness guarantee.
Mininet results remain a separate validation requirement.

## Six-switch extension

T5-Mesh and T6-Mesh are scaling-only topologies and are never added to the
seed-2026 training replay. The T6 result uses one atomic case per fault family
plus six deterministic cross-family two-fault cases. It is labeled
representative rather than exhaustive. Action count, uncached shield time,
graph encoding time, and Q-inference time are reported from T2 through T6.
