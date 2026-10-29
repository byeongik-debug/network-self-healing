# Safe Reinforcement Learning for Network Self-Healing

VLAN·ACL·link 장애가 발생한 네트워크를 진단하고 복구하는 강화학습 연구 프로젝트입니다. 결정론적 Python 시뮬레이터에서 DQN과 safety/progress shield를 평가하고, 더 큰 topology를 위한 Graph DQN 및 imperfect-execution 실험을 제공합니다.

> **연구용 시뮬레이터입니다.** 실제 네트워크 운영에 바로 적용할 수 있는 제품이 아니며, Mininet/Open vSwitch 실행 및 clean-environment 재현은 아직 완료되지 않았습니다.

## 현재 핵심 결과

### 작은 유한 모델: Progress-Safe DQN

T2 topology(스위치 2대, 호스트 4대, VLAN 2개)의 고정 Eval-51에서 Vanilla DQN, damage-only Safe DQN, lexicographic progress shield를 비교했습니다. Vanilla와 Progress-Safe는 동일한 35,747-transition 학습 예산과 10개 seed(2026–2035)를 사용합니다.

| 방법 | 복구율 | 완전 복구 seed | 실행당 신규 피해 | 평균 복구 단계 | Timeout |
|---|---:|---:|---:|---:|---:|
| Vanilla DQN | 97.25% ± 4.91% | 7/10 | 17.10 ± 8.65 | 5.375 | 2.75% |
| Damage-only Safe DQN | 90.59% ± 29.76% | 9/10 | 8.40 ± 1.90 | 6.741 | 9.61% |
| **Progress-Safe DQN** | **100% (510/510)** | **10/10** | **9 (각 seed)** | **5.255** | **0%** |

- Progress-Safe의 10개 seed 모두 Eval-51의 51개 scenario를 복구했고, 이 유한 모델에서 safety/viability violation은 0이었습니다. 이 결과를 모델 밖의 보장으로 일반화하지 않습니다.
- Damage-only shield는 안전성만으로 진행을 보장하지 않습니다. seed 2033에서 diagnose/self-loop에 머무는 liveness 실패가 관찰되어 progress shield를 별도 평가했습니다.
- Vanilla 대비 Progress-Safe의 paired damage 차이는 Wilcoxon `p=0.0078125`, rank-biserial effect size `0.9556`입니다. 피해 수치는 반드시 복구율과 함께 해석해야 합니다.
- Learning-free `progress + random admissible action`도 이 Eval-51에서 10 seeds 모두 51/51 복구, damage 9, 평균 5.255 steps를 달성했습니다. 따라서 이 결과는 DQN의 복구 계획 능력 우위를 입증하지 않습니다. DQN은 shield가 허용한 동등한 경로 사이의 selector로 해석합니다.

### 확장 topology: Graph DQN

T2, T3, T4-Line을 학습 측에 두고 T4-Ring을 held-out topology로 평가합니다. T5-Mesh는 runtime scaling용이며 T6-Mesh 결과는 대표 zero-shot 확인입니다. 확장 study의 5-seed 결과는 초기화에 민감하고 완전한 복구를 보장하지 않습니다.

| 평가 조건 | Topology | 복구율 (5-seed 평균) | 평가 규모 |
|---|---|---:|---:|
| Nominal | T4-Ring | 69.09% | 110 scenario/seed |
| Combined execution/telemetry mismatch | T4-Ring | 49.09% | 110 scenario/seed |
| Nominal | T6-Mesh | 65.00% | 12 대표 scenario/seed |
| Combined execution/telemetry mismatch | T6-Mesh | 48.33% | 12 대표 scenario/seed |

확장 평가에서 새 policy damage는 0이었지만 seed별 복구 편차가 큽니다. T6의 12개 scenario는 exhaustive fault evaluation이 아닙니다. 상세 설계와 결과는 [`paper/EXPANDED_STUDY_PROTOCOL.md`](paper/EXPANDED_STUDY_PROTOCOL.md), [`results/expanded/SIX_SWITCH_STUDY.md`](results/expanded/SIX_SWITCH_STUDY.md), [`paper/CLAIM_EVIDENCE.md`](paper/CLAIM_EVIDENCE.md)를 참고하세요.

### Imperfect execution과 runtime

- T2에서 configuration 명령이 확률적으로 no-op이 되는 조건을 0%, 10%, 20%, 30%로 바꾸어 총 **20,400회** 평가했습니다. 이 설정과 관측 가능한 no-op 모델에서 모두 복구율 100%, timeout 0이었고, 실패율이 높아질수록 복구 단계가 늘었습니다. 지연 실행이나 임의의 실제 장비 오류까지 검증한 결과는 아닙니다.
- CPU 단일 프로세스의 3회 측정에서 T6-Mesh action catalog는 1,016개, graph encoding은 0.41 ms, Q inference는 1.67 ms, uncached shield mask는 80.48 ms였습니다. 이 수치는 해당 환경의 기술적 측정치이며 hardware-neutral benchmark가 아닙니다.

## 환경과 방법

### 기준 T2 환경

- 스위치 2대, 호스트 4대, VLAN 10/20, 스위치 간 trunk
- access VLAN 반전 4개와 trunk VLAN 누락 4개로 이루어진 8개 원자 장애
- 비어 있지 않은 장애 조합 255개, stratified split은 train 204 / Eval 51
- 79차원 관측, 19개 action, 최대 30 step
- Scenario ID, fault 원인, 정답 action, `policy_healthy` 등 privileged 정보는 정책 입력에서 제외

### 확장 환경

| Topology | 스위치 | 호스트 | VLAN | 사용 |
|---|---:|---:|---:|---|
| T2 | 2 | 4 | 2 | 기존 기준 및 exact finite-state shield |
| T3 | 3 | 8 | 3 | Graph DQN 학습/평가 |
| T4-Line | 4 | 16 | 4 | Graph DQN 학습/평가 |
| T4-Ring | 4 | 16 | 4 | held-out topology 일반화 |
| T5-Mesh | 5 | 20 | 5 | runtime scaling |
| T6-Mesh | 6 | 24 | 6 | 대표 zero-shot 평가 |

확장 환경은 access VLAN, trunk membership, link/port 상태, ACL fault를 모델링합니다. T2의 explicit shield와 확장 topology의 runtime shield는 서로 다른 범위의 방법이므로 formal guarantee를 동일하게 해석하지 않습니다.

### Safety와 progress

- **Minimum-damage viability shield:** T2의 256 canonical configuration state에서 functional goal까지의 최소 추가 피해를 계산합니다.
- **Progress shield:** damage 최적성을 유지하면서 복구 진행을 제한하는 lexicographic 기준을 사용해 damage-only shield의 liveness 반례를 다룹니다.
- **Graph safety shield:** 확장 topology에서 policy 위반을 새로 만들지 않도록 admissible action을 제한합니다. 이 확장 방식은 경험적 bounded 검증이며 T2 exact graph의 성질과 구분됩니다.

## 설치, 테스트, 실행

Python 3.10 이상을 권장합니다. Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

현재 전체 테스트 결과: **196 passed**.

### 주요 실험 명령

```powershell
# T2: progress-shield를 사용한 단일 seed 학습 및 Eval-51 평가
python -m experiments.run_progress_multiseed 2026

# Graph DQN 학습 (기본 300 episodes)
python -m experiments.train_graph_dqn

# T6 대표 zero-shot 평가
python -m experiments.evaluate_t6_zero_shot `
  --checkpoint results\expanded\graph_dqn\model_seed_2026_ep300.pt

# Topology별 shield/runtime 측정 및 그림 생성
python -m experiments.benchmark_expanded_scaling --repeats 3
python -m experiments.generate_scaling_figure
python -m experiments.generate_topology_suite_figure
```

`run_progress_multiseed` 명령은 지정한 seed 하나의 실행입니다. 결과 파일은 `results/` 아래에 기록되며 기존 산출물과 같은 경로를 재사용할 때 덮어쓸 수 있으니 중요한 결과는 실행 전에 보관하세요.

## 저장소 구조

```text
agents/       DQN, Graph DQN, replay buffer
baselines/    rule-based, configuration-diff 및 확장 비교군
config/       기준 topology와 scenario
env/          T2 네트워크 환경 및 확장 topology 시뮬레이터
experiments/  학습, 평가, 분석, scaling 및 figure 생성 스크립트
paper/        원고 초안, 프로토콜, 재현 절차, 주장-근거표
results/      평가 CSV/JSON, 요약 및 artifact manifest
safety/       T2 viability/progress shield
testbed/      clean-room 및 Mininet 실행 harness
tests/        회귀, 안전성, 분석 및 확장 환경 테스트
```

## 재현성과 남은 한계

- 결과별 원시 CSV/JSON과 [`results/artifact_manifest.json`](results/artifact_manifest.json)을 보존합니다.
- Linux clean-environment 전체 재현과 Mininet/Open vSwitch 대표 replay는 실행 harness만 준비되어 있으며, 실제 실행 증거는 아직 없습니다. 절차는 [`paper/REPRODUCIBILITY.md`](paper/REPRODUCIBILITY.md)에 있습니다.
- 확장 Graph DQN은 5개 seed에서도 topology·mismatch 조건에 따라 복구율 편차가 큽니다.
- T6 평가는 12개 대표 scenario이지 전수 fault 조합이 아닙니다.
- 시뮬레이터 결과는 실제 네트워크로의 transfer, 운영 안정성, 임의의 장애 모델에 대한 보장을 뜻하지 않습니다.

## 주요 자료

- [확장 실험 설계](EXPERIMENT_EXPANSION_SPEC.md)
- [연구 로드맵](RESEARCH_ROADMAP_2026-11-15.md)
- [논문 초안](paper/DRAFT.md)
- [실험 프로토콜](paper/EXPANDED_STUDY_PROTOCOL.md)
- [주장-근거 대응표](paper/CLAIM_EVIDENCE.md)
- [재현성 절차](paper/REPRODUCIBILITY.md)
- [확장 topology 결과](results/expanded/SIX_SWITCH_STUDY.md)
