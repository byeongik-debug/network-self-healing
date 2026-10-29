# 안전하고 진행 가능한 네트워크 자동 복구를 위한 사전식 런타임 보증

> 작업 원고. 모든 수치는 `results/`의 JSON/CSV에 근거한다. Mininet 실제 실행과 깨끗한 환경 재현은 아직 완료되지 않았다.

## 초록

학습 기반 네트워크 복구는 설정 장애로 손상된 서비스를 회복할 수 있지만, 제약 없는 정책은 정상 서비스를 추가로 손상할 수 있다. 반대로 안전성만 고려한 action shield는 서비스를 보호하면서도 복구를 끝내지 못할 수 있다. 본 연구는 255개 VLAN 장애 조합과 서로 겹치지 않는 학습 204개·평가 51개 시나리오로 구성된 유한 환경에서 이러한 안전성–진행성 충돌을 분석한다.

최소 피해 viability shield는 Vanilla DQN보다 부수적인 policy 피해를 줄였지만, 10개 학습 seed 중 하나는 진단 self-loop를 반복해 51개 중 3개만 복구했다. 이를 해결하기 위해 누적 피해의 최솟값을 먼저 보존하고, 그 안에서 남은 설정 전이 수를 엄격히 감소시키는 사전식 progress shield를 제안한다.

비학습 uniform random selector와 Progress-Safe DQN은 모두 10개 seed의 510회 평가를 전부 복구했으며, 51개 시나리오 실행당 총 피해 9와 평균 5.2549 step을 동일하게 기록했다. 따라서 유한 모델의 복구율·피해·경로 길이는 DQN이 아니라 exact progress shield가 결정하고, DQN은 허용된 동등 경로 사이의 tie-breaker 역할만 한다. Vanilla DQN의 실행당 피해는 17.10 ± 8.65였고 paired Wilcoxon 검정은 p=0.0078125, rank-biserial correlation은 0.9556이었다. 다만 shield는 완전한 전이 모델을 알고 Vanilla는 알지 못하므로 이는 정보 조건이 같은 순수 학습 성능 비교가 아니다.

확장 실험에서는 graph DQN을 다섯 seed로 학습해 학습에 없던 T4-Ring과 대표 T6-Mesh 시나리오에서 평가했다. 기존 설정의 평균 복구율은 T4에서 nominal 69.09%, combined 49.09%, T6에서 65.00%, 48.33%였으며 신규 policy 피해는 모두 0이었다. seed별 복구율은 0~100%로 크게 달라 안전성은 유지됐지만 학습된 liveness는 불안정했다. Guidance 제거와 policy-relation feature 추가 ablation은 평균 복구율을 높였으나 표본이 작고 ACL 불안정성을 완전히 제거하지 못했다.

## 1. 서론

네트워크 자동 복구는 두 조건을 동시에 만족해야 한다. 장애를 고치는 동안 정상 서비스를 새로 손상하지 않아야 하고, 안전한 self-loop에 머무르지 않고 복구를 끝내야 한다. 첫 번째 조건만 만족하는 정책은 충분한 self-healing 시스템이 아니다.

초기 시스템은 masked DQN과 exact minimum-damage viability shield를 결합했다. 열 개 seed 중 아홉 개는 모든 평가 장애를 이론적 최소 피해로 복구했지만 seed 2033은 3/51만 복구했다. 실패한 48 episode의 1,440개 행동 가운데 진단이 1,184개(82.22%)였고 모든 실패가 이전 설정 상태를 재방문했다.

원인은 구조적이다. 피해만 최적화하면 피해가 없는 self-loop와 복구를 향해 전진하는 행동의 viability가 같아진다. 본 연구는 누적 피해를 먼저 최소화하고, 같은 최소 피해 경로 중 남은 설정 전이 수를 최소화하는 사전식 목적함수로 이 모호성을 제거한다.

### 주요 기여

1. 피해만 고려한 viability가 liveness를 보장하지 않는 구체적인 진단 self-loop 반례를 제시한다.
2. 유한 설정 그래프에서 최소 피해를 보존하면서 엄격한 진행을 요구하는 exact progress shield를 정의하고 전수 검사한다.
3. 비학습 random·first selector도 DQN과 동일한 주요 결과를 낸다는 사실을 보여 DQN의 역할을 동등 경로 사이의 선택으로 한정한다.
4. 기능 복구, exact configuration recovery, 부수 서비스 피해를 분리해 실패한 실행의 낮은 피해가 우수한 안전성으로 오해되는 것을 막는다.
5. variable-size graph 정책과 pending-aware counterfactual shield를 이용한 미관측 topology 확장 실험을 제공한다.

### 1.1 관련 연구

Shielded reinforcement learning은 환경 abstraction과 안전 명세로부터 런타임 장치를 구성해 위반 행동을 차단한다 [R1]. Runtime assurance와 Simplex 구조도 검증되지 않은 controller 주위에 신뢰 가능한 감시기나 backup controller를 둔다 [R2]. 본 연구는 이를 네트워크 설정 복구에 적용하되, damage-optimal shield가 진행 없는 cycle을 허용할 수 있음을 보이고 사전식 progress certificate로 보완한다.

RL 기반 cellular self-healing [R3, R4]과 intent-based SDN 복구 [R5]는 학습된 복구 행동의 가능성을 보여준다. RouteNet [R6]과 GNN–DRL routing [R7]은 topology가 달라져도 적용할 수 있는 표현을 다룬다. 본 연구의 T4/T6 실험은 traffic engineering이 아니라 장애 복구와 runtime shielding을 대상으로 하며, T6 ACL 실패는 topology 불변 표현만으로 policy 의미를 충분히 나타낼 수 없음을 보여준다.

## 2. 문제 정의

기본 환경은 switch 2대, host 4대, VLAN 2개와 switch 간 trunk 하나로 구성된다. Access VLAN 반전 4개와 trunk membership 제거 4개의 비어 있지 않은 조합으로 255개 장애를 만든다. 행동 공간은 진단, access VLAN 설정 8개, trunk 설정 8개, rollback, 완료 선언의 19개 행동이다.

`N(s,a)`를 행동 `a`로 정상에서 손상으로 바뀐 policy 수, `J(s)`를 기능적으로 정상인 상태까지 필요한 최소 누적 추가 피해라고 하자. 기존 shield는 다음 조건을 만족하는 설정 행동을 허용한다.

```text
N(s,a) + J(T(s,a)) = J(s)
```

이 조건은 불필요한 피해를 막지만 비용이 0인 cycle을 허용한다. 위 등식을 만족하는 경로 중 정상 상태까지 필요한 최소 설정 전이 수를 `L(s)`라고 정의하면 progress shield는 비정상 상태에서 다음 조건까지 요구한다.

```text
N(s,a) + J(T(s,a)) = J(s)
L(T(s,a)) = L(s) - 1
```

`L`은 음이 아닌 정수이고 매 설정 전이에서 감소하므로 결정론적 유한 모델 안에서 종료를 보장한다. 정상 상태에서는 관측 갱신을 위한 진단 한 번과 완료 선언만 허용한다.

## 3. 실험 방법

### 3.1 분할과 정보 경계

Seed 2026 stratified split은 학습 204개와 평가 51개로 구성되며 시나리오 중복이 없다. DQN의 79차원 관측에는 port 설정, link 상태, 진단 결과와 step 수가 포함되지만 시나리오 ID, 주입 장애, reference action과 실제 health는 포함되지 않는다. 반면 exact shield는 simulator의 설정과 전이 모델 전체를 사용한다.

### 3.2 비교 방법

- Rule-based recovery
- reference 설정을 직접 이용하는 Configuration-difference oracle
- valid-action mask만 사용하는 Vanilla DQN
- 기존 Minimum-damage Safe DQN
- 사전식 Progress-Safe DQN
- 학습 없는 progress shield + random 또는 first selector

### 3.3 학습과 평가

유한 모델의 DQN은 `79–128–128–19` 구조, batch 64, replay capacity 10,000, warmup 1,000, discount 0.99, learning rate 0.001을 사용했다. Epsilon은 20,000 transition 동안 1.0에서 0.05로 감소하고 target network는 500 transition마다 동기화했다. Seed 2026–2035의 각 학습 방법은 35,747 environment transition과 34,748 optimizer update를 동일하게 사용했다. 평가는 `epsilon=0`, 최대 30 step이다.

주요 지표는 functional recovery, new policy damage, safety optimality gap, timeout이며 exact configuration recovery는 보조 지표다. 통계 단위는 seed이고 필요한 경우 20,000회 bootstrap과 paired Wilcoxon 검정을 사용한다.

### 3.4 확장 topology

T2, T3, T4-Line은 학습에 사용하고 T4-Ring은 학습·warmup·checkpoint 선택에서 제외했다. 장애 family는 VLAN과 trunk 외에 link down, port shutdown, ACL mutation을 포함한다. Graph Q-network는 variable-size topology에서 parameter를 공유한다.

평가는 nominal과 combined mismatch 두 조건에서 수행한다. Combined 조건은 command failure 0.20, actuation delay 2 step, telemetry delay 2 step, stale telemetry probability 0.50이다. T4-Ring manifest는 110개이며 policy-damaging 60개와 no-impact control 50개로 구성된다. T6-Mesh는 여섯 atomic fault와 여섯 mixed fault로 구성한 대표 12개 시나리오이며 전수 평가가 아니다.

## 4. 결과

### 4.1 유한 모델의 10-seed 비교

| 방법 | 기능 복구 | 완전 복구 seed | 실행당 피해 | 복구 episode당 피해 | Timeout | 평균 step | Exact 설정 복구 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla DQN | 97.25% ± 4.91% | 7/10 | 17.10 ± 8.65 | 0.35 ± 0.18 | 2.75% ± 4.91% | 5.375 ± 1.444 | 38.24% ± 31.53% |
| Minimum-damage Safe DQN | 90.59% ± 29.76% | 9/10 | 8.40 ± 1.90* | 0.16 ± 0.06* | 9.61% ± 30.38% | 6.741 ± 7.800 | 46.86% ± 19.70% |
| **Progress-Safe DQN** | **100% ± 0%** | **10/10** | **9.00 ± 0** | **0.18 ± 0** | **0%** | **5.255 ± 0** | 46.86% ± 14.45% |
| **Progress shield + random** | **100% ± 0%** | **10/10** | **9.00 ± 0** | **0.18 ± 0** | **0%** | **5.255 ± 0** | 50.39% ± 6.06% |
| Minimum-damage shield + random | 16.86% ± 3.94% | 0/10 | 7.30 ± 1.34* | 0.15 ± 0.12* | 0% | 5.843 ± 0.580 | 9.02% ± 1.65% |

별표 값이 낮은 이유는 완전 복구하지 못해 필요한 피해 자체를 지불하지 않았기 때문이다. Progress shield + first도 51/51, 피해 9, 평균 5.2549 step을 기록했다. 따라서 주요 결과는 학습 성과가 아니라 shield의 구조적 결과다.

### 4.2 Liveness 실패와 수정

기존 seed 2033은 3/51만 복구하고 49/51에서 timeout이 발생했다. Frozen checkpoint에 progress shield를 적용하자 복구는 51/51, timeout은 0, 총 피해는 이론적 최솟값 9가 됐다. 신규 학습 10개 seed에서도 같은 주요 결과가 반복됐다.

### 4.3 고전 baseline과 명령 실패

Rule-based baseline은 88.24%, 총 피해 16, 평균 5.92 step이었고 configuration-difference oracle은 100% 복구와 exact 복구를 달성했지만 피해 55를 기록했다. 관측 가능한 no-op command failure를 0%, 10%, 20%, 30%로 적용한 총 20,400회 평가에서는 모두 복구됐다. 평균 step은 5.25에서 6.64로 늘었으나 시나리오당 평균 피해는 0.176으로 유지됐다.

### 4.4 미관측 topology 다중 seed

| Topology·조건 | Seed | Episode | 평균 복구 | Seed SD | Seed 범위 | Seed-bootstrap 95% CI | 피해 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T4-Ring nominal | 5 | 550 | 69.09% | 39.86%p | 0–97.27% | [33.64%, 92.73%] | 0 |
| T4-Ring combined | 5 | 550 | 49.09% | 38.28%p | 0–95.45% | [19.82%, 78.36%] | 0 |
| T6-Mesh nominal | 5 | 60 | 65.00% | 39.26%p | 0–100% | [31.67%, 90.00%] | 0 |
| T6-Mesh combined | 5 | 60 | 48.33% | 44.25%p | 0–83.33% | [15.00%, 81.67%] | 0 |

표본에서는 안전성이 유지됐지만 liveness는 seed에 따라 불안정했다. 따라서 안정적인 cross-topology learning이나 수렴을 주장할 수 없다.

### 4.5 Scaling과 ablation

행동 수는 T2의 44개에서 T6의 1,016개로 증가하며 policy-scoped ACL catalog를 쓰면 T6은 468개다. T6에서 graph encoding은 0.41 ms, Q inference는 1.67 ms, uncached counterfactual mask는 80.48 ms였다. Shield 계산이 가장 큰 runtime 비용이다.

Guidance를 제거한 paired seed 5개는 T4 nominal/combined에서 각각 +24.36%p/+41.45%p, T6에서 +23.33%p/+35.00%p의 평균 차이를 보였다. 그러나 Wilcoxon p-value는 0.1875~0.4375였으므로 guidance가 일반적으로 해롭다고 결론 내리지 않는다.

Policy-relation feature를 추가한 13-D 표현은 11-D 대비 T4 nominal 93.45%→99.64%, combined 90.55%→99.09%, T6 nominal 88.33%→98.33%, combined 83.33%→96.67%로 개선됐다. Atomic ACL은 nominal 50%→90%, combined 40%→80%였지만 신뢰구간이 0을 포함하고 조건별로 악화된 seed도 하나씩 있었다. 모든 비교의 신규 피해는 0이었다.

## 5. 학습 구성요소의 역할

기능 복구·피해·경로 길이의 분산이 0인 것은 DQN 학습 안정성의 근거가 아니라 progress certificate의 성질이다. Exact shield가 주요 결과를 고정하고 Q-function은 사전식 목적에서 동률인 경로를 선택한다. Exact configuration recovery는 seed별 29.41~68.63%로 달라 학습된 tie-breaking이 기능적으로 동등한 도달점에는 영향을 준다. 현재 근거는 학습 selector를 위한 안전하고 진행 가능한 runtime assurance를 지지하지만 우월한 learned planning을 입증하지 않는다.

## 6. 한계와 타당성 위협

- 형식적 보장은 2-switch, 4-host exact-state 모델에 한정된다.
- 확장 실험은 경험적 결과이며 seed 5개로 수렴이나 안정성을 주장할 수 없다.
- Shield는 exact transition model을 사용하므로 Vanilla DQN과 정보가 비대칭이다.
- 지연·stale telemetry는 simulator에서 생성됐으며 실제 controller 측정이 아니다.
- Mininet과 hardware 결과는 아직 없다.
- T6은 대표 12개 사례일 뿐 전수 평가가 아니다.
- Uncached T6 mask 80.48 ms는 통제된 hardware-neutral latency가 아니다.
- 학습 시간은 background load 영향을 받아 성능 주장에 사용하지 않는다.

## 7. 재현성

저장소에는 유한 모델 checkpoint 30개, scenario별 CSV, trajectory, seed 요약과 통계 결과가 있다. 확장 연구도 checkpoint, 학습 trace, held-out 결과와 bootstrap 요약을 보존한다. 회귀 테스트는 현재 196개다. 자세한 절차는 `paper/REPRODUCIBILITY.md`에 있다. Artifact 공개 전에는 동결된 commit의 깨끗한 checkout 재현과 Linux Mininet 실행이 필요하다.

## 8. 결론

최소 피해 안전성만으로는 진행을 보장할 수 없다. 학습 정책은 viability kernel 안에서 무해한 행동을 반복하며 복구에 실패할 수 있다. 누적 피해와 엄격한 진행을 사전식으로 결합한 shield는 최소 피해 보장을 약화하지 않으면서 이 실패를 제거했다. 동시에 비학습 selector도 동일한 주요 결과를 냈으므로 이를 DQN의 성능 향상으로 해석해서는 안 된다. 향후 핵심 과제는 더 큰 상태 공간과 실제 model mismatch에서 approximate progress certificate의 효과를 검증하는 것이다.

## 주요 결과 자료

- 주 통계: `results/fresh_progress_statistical_summary.json`
- 비학습 baseline: `results/shield_only_baselines/summary.json`
- 실패 진단: `results/multiseed/safe_seed_2033_liveness_diagnostic.json`
- 확장 다중 seed: `results/expanded/multiseed/summary.json`
- Guidance ablation: `results/expanded/guidance_ablation/summary.json`
- Policy feature ablation: `results/expanded/policy_feature_ablation/summary.json`
- Scaling: `results/expanded/scaling/runtime_scaling.csv`
- T6 대표 평가: `results/expanded/t6_zero_shot/summary.json`

## 참고문헌

- [R1] Alshiekh et al., “Safe Reinforcement Learning via Shielding,” AAAI 2018.
- [R2] Slagel et al., “A Formal Verification Framework for Runtime Assurance,” NASA Formal Methods 2024.
- [R3] Mismar and Evans, “A Reinforcement Learning Based Solution for Self-Healing in LTE Networks,” VTC Fall 2014.
- [R4] Mismar, Evans, and Alkhateeb, “Deep Q-Learning for Self-Organizing Networks Fault Management and Radio Performance Improvement,” Asilomar 2018.
- [R5] Makiyah, Rasool, and Rawdhan, “A reinforcement learning framework for self-healing fault recovery in intent-based SDNs,” Discover Networks, 2026.
- [R6] Rusek et al., “RouteNet: Leveraging Graph Neural Networks for Network Modeling and Optimization in SDN,” 2019.
- [R7] Ding et al., “GROM: A generalized routing optimization method with graph neural network and deep reinforcement learning,” JNCA 229, 2024.
