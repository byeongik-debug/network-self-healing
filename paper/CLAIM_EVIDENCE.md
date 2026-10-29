# 주장–근거 대응표

상태는 2026-10-27 현재 저장소 자료를 기준으로 한다. “뒷받침됨”은 명시된 범위를 해당 자료가 직접 확인한다는 뜻이며 범위 밖으로 일반화하지 않는다.

| ID | 주장 | 기준 자료 | 상태 |
|---|---|---|---|
| C1 | 피해만 최소화하는 viability는 liveness를 보장하지 않는다 | `results/multiseed/safe_seed_2033_liveness_diagnostic.json` | 유한 simulator에서 뒷받침됨 |
| C2 | seed 2033 실패는 진단/self-loop가 지배한다 | 같은 진단 JSON; `results/multiseed/seed_2033_liveness_audit.md` | 진단 행동 82.22%, 기능 복구 실패 48건 모두 설정 재방문 |
| C3 | progress shield는 최소 피해 불변식을 보존한다 | `tests/test_progress_viability_shield.py`; `results/fresh_progress_statistical_summary.json` | 256상태 결정론적 그래프에서 뒷받침됨 |
| C4 | progress shield는 범위 내 유한 복구를 보장한다 | 전수 종료 검사; seed별 신규 요약 | 모델링한 장애와 결정론적 전이에서 뒷받침됨 |
| C5 | 학습 selector와 무관하게 주요 progress 결과는 seed에 불변이다 | `results/multiseed_progress/`; `results/shield_only_baselines/summary.json` | 두 방법 모두 510/510 복구, 실행당 피해 9. 학습 안정성 근거는 아님 |
| C6 | Progress-Safe는 Vanilla보다 부수 피해가 적다 | `results/fresh_progress_statistical_summary.json` | W=1, p=0.0078125, rank-biserial=0.9556 |
| C7 | 복구 여부 없이 낮은 피해를 우수성으로 해석할 수 없다 | `results/fresh_progress_comparison.csv`의 seed 2033 | 뒷받침됨 |
| C8 | 학습은 기능적으로 동등한 설정 중 도달점을 바꾼다 | `results/fresh_progress_statistical_summary.json` | 기술적으로 뒷받침되나 인과적 이점은 확인되지 않음 |
| C9 | Progress-Safe는 고전 baseline보다 우수하다 | `results/stratified_baseline_results.csv` | 부분적 근거. baseline의 정보 접근 조건이 다름 |
| C10 | simulator 결과는 emulator로 이전된다 | 예정: `results/testbed/runs/<run-id>/mininet_replay_seed_2026.json` | **미완료: harness와 preflight는 있으나 Linux 실행이 없음** |
| C11 | graph 정책은 exact model 밖에서도 동작할 수 있으나 학습 liveness는 초기화에 민감하다 | `results/expanded/multiseed/summary.json`; scaling JSON | T4 평균 69.09%/49.09%, T6 65.00%/48.33%; seed 범위에 0% 포함, 피해 0 |
| C12 | 관측 가능한 no-op 명령 실패 30%까지 견딘다 | `results/model_mismatch_summary.json` | 해당 실패 모델에서만 20,400/20,400 복구 |
| C13 | 결과를 재현할 수 있다 | 요구 패키지, manifest, 테스트 196개, clean runner | 부분적 근거. clean-machine 실행이 남음 |
| C14 | reference guidance는 필수가 아니며 이 표본에서는 평균 liveness 저하와 연관된다 | `results/expanded/guidance_ablation/summary.json` | 네 조건 모두 no-guidance 평균 차이가 양수이나 n=5에서 유의하지 않음 |
| C15 | DQN은 유한 모델의 주요 Progress-Safe 결과를 결정하지 않는다 | `results/shield_only_baselines/summary.json` | progress+random과 progress+first 모두 51/51, 피해 9, 5.2549 steps |
| C16 | policy-relation feature는 복구 평균을 높이지만 ACL 불안정성을 없애지 못한다 | `results/expanded/policy_feature_ablation/summary.json` | 전체 +6.18~+13.33%p, atomic ACL +40%p이나 ACL 구간은 0 포함 |

## 원고에서 허용하는 주장

- “보장한다”는 표현에는 반드시 “유한 결정론적 모델 안에서”라는 제한을 붙인다.
- 모집단 전체의 100% 보장이 아니라 평가 510/510 성공으로 서술한다.
- C10 결과가 생기기 전에는 Mininet을 준비된 검증 harness로만 표현한다.
- DQN은 shield가 허용한 경로 사이의 학습 selector이며 우월한 학습 계획 능력을 주장하지 않는다.
- T4/T6 이전은 가능하지만 초기화에 민감하다고 쓰고 seed 범위와 불확실성을 제시한다.
- T6는 12개 대표 zero-shot 검사이며 전수 평가가 아니다. ACL 두 사례 실패를 명시한다.
- policy-relation feature 부족은 지지된 가설이지 확정된 원인이 아니다.
- guidance 결과는 paired 기술적 연관성으로만 보고하고 구간과 유의하지 않은 p-value를 포함한다.
- 기존 Safe의 피해는 복구율과 함께 보고한다.

## 제출 전 차단 항목

1. Mininet replay를 실행·보관하거나 simulator 한정 연구임을 명시한다.
2. 깨끗한 환경에서 배포 검증기를 실행하고 해시를 동결한다.
3. 투고처에 맞는 관련 연구와 인용을 보완한다.
