# 논문화 로드맵 — 2026-11-15

## 2026-10-27 진행 상황 갱신

핵심 결론은 DQN 우수성이 아니라 damage-only shield의 liveness 반례와
lexicographic progress shield에 의한 수정이다. Learning-free progress+random은
Eval-51에서 10 seeds 모두 51/51 recovery, damage 9, mean 5.2549 steps로
Progress-Safe DQN의 primary outcome을 그대로 재현했다. DQN은 shield가 허용한
동등 경로 사이의 tie-breaker로 제한해 해석한다.

완료:

- [x] finite-state 10-seed 비교와 seed 2033 liveness 진단
- [x] learning-free Eval-51 baseline 네 종류
- [x] 3–6 switch expanded simulator와 T4/T6 holdout 평가
- [x] expanded Graph DQN 5-seed 및 matched guidance ablation
- [x] shield latency 846.76 ms에서 80.48 ms로 최적화
- [x] `shield_view=true|observed` 구현과 10-scenario smoke
- [x] expanded shield-only baseline 세 종류
- [x] 1000-episode timing probe
- [x] 전체 테스트 **196 passed**

준비만 완료하고 실행하지 않은 항목:

- [ ] Full shield-view matrix: 488 episodes, 예상 약 4–5분
- [ ] Guidance-free 10 seeds × 300/1000/3000 episodes: 순차 예상 약
  5시간 44분, 보수 범위 4시간 20분–7시간 10분

남은 제출 blocker:

- [ ] Linux/Mininet representative replay (versioned runner/preflight 준비 완료, 실제 Linux 실행 대기)
- [ ] clean-environment reproduction 및 artifact hash 동결 (clean-checkout runner 준비 완료, release commit 대기)
- [x] policy-feature 5-seed matched ablation
- [ ] 필요 시 observed-shield 전체 평가
- [ ] venue formatting 및 최종 claim-evidence audit

아래 내용은 초기 계획 기록이며, 완료 여부와 수치는 위 갱신을 우선한다.

기준일: 2026-10-26

## 현재 증거 수준

- 범위: 2-switch/4-host, VLAN 10/20의 유한 시뮬레이터
- 데이터: 255개 fault 조합, 고정 stratified split(학습 204 / 평가 51)
- 비교: Vanilla DQN 대 Minimum-Damage Viability Shield를 결합한 Safe DQN
- 공정성: 양쪽 모두 seed 2026–2035와 35,747 transition budget 사용
- 주요 결과: Vanilla damage `17.10 ± 8.65`, Safe damage `8.40 ± 1.90`; paired Wilcoxon `p=0.0078125`, rank-biserial `0.9556`
- 안전성: 복구에 성공한 Safe run은 theoretical minimum damage와 gap 0
- 재현성 자산: 20개 checkpoint, scenario-level CSV, seed-level CSV, 통계 JSON 보존
- 회귀 테스트: README 기준 109개. 현재 머신의 Python 실행기가 깨져 있어 이번 감사에서 재실행하지 못함
- testbed: 비어 있음(`testbed/.gitkeep`만 존재)

## 논문 주장에 남은 핵심 위험

1. **Liveness 반례**: Safe seed 2033은 3/51만 복구했다. 안전하지만 멈추는 정책은 자가복구 논문의 주장을 약화한다.
2. **외적 타당성 부족**: topology와 fault family가 하나이고 실제 네트워크/Mininet 검증이 없다.
3. **비교군 부족**: 현재 핵심 학습 비교는 Vanilla DQN 하나다. oracle/rule-based 결과는 있으나 동일 프로토콜의 강한 baseline과 ablation이 부족하다.
4. **통계 설계 부족**: damage 검정은 있으나 recovery/liveness, confidence interval, multiple metrics 처리와 seed 수 정당화가 부족하다.
5. **시뮬레이터 의존 shield**: 256-state exact graph를 전부 아는 조건이라 확장성과 model mismatch 주장이 아직 없다.
6. **원고 자산 부재**: 논문 구조, figure 생성 파이프라인, artifact manifest가 없다.

## 제출 가능 판정 기준

- Safe 방법이 모든 예정 seed에서 safety violation 0을 유지하고, recovery 하한을 사전 정의된 기준으로 충족
- seed 2033 실패 원인을 trajectory/Q-value/replay 관점에서 설명하고, 수정 전후 ablation으로 효과 분리
- 최소 20 seeds 또는 power/precision 근거가 있는 seed 수, paired protocol, bootstrap 95% CI와 effect size 보고
- 최소 비교군: rule-based, config-diff/oracle, Vanilla DQN, shield-at-eval-only, Safe DQN
- 최소 ablation: shield 제거, liveness 장치 제거, reward 항 제거 또는 mask 적용 위치 변경
- 최소 2개 topology 규모 또는 Mininet testbed에서 대표 fault replay; 불가능하면 논문 범위를 simulator study로 명시하고 scalability 실험 추가
- 모든 표/그림이 원시 CSV/JSON에서 한 명령으로 재생성되고 환경 lock, seed, commit hash가 기록됨
- 초록·방법·실험·한계·재현성 부록을 포함한 원고 초안 완성

## 일정

### 10/26–10/29: 감사와 liveness 원인 규명

- Python 3.13 `.venv` 복구 완료; 신규 검증 포함 **112 tests passed**
- seed 2033 기존 결과 감사 완료: 실패 action의 82.22%가 `diagnose`, 48개 실패 중 38개가 `diagnose` 비율 80% 이상
- action별 Q-value, admissible mask, configuration revisit 진단 실행 완료: 실패 48건 모두 configuration 재방문 확인
- 실패 가설을 `diagnose` Q-dominance, replay imbalance, reward/tie-breaking으로 분리
- 실험 프로토콜과 metric 정의를 동결

### 10/30–11/03: 방법 수정과 ablation

- lexicographic progress shield 구현: damage 최적성을 1순위, 남은 configuration steps를 2순위로 제한
- 전체 Eval-51에서 finite termination과 minimum damage를 검사하는 단위/속성 테스트 추가
- frozen 10-seed checkpoint 평가 완료: 10/10 seeds 51/51 recovery, timeout 0, damage 9, safety gap 0
- fresh 10-seed paired training 완료: 모든 seed exact 35,747 transitions, 51/51 recovery, timeout 0, damage 9, safety gap 0, viability violation 0
- 기존 Safe checkpoint에 progress shield만 적용한 inference-time ablation도 별도 보존
- 결과가 나쁘면 11/03에 범위를 “safety–liveness trade-off 분석”으로 전환

### 11/04–11/08: 본 실험

- seed 확대, paired evaluation, bootstrap CI/effect size 산출
- baseline 5종을 동일 split/budget/evaluator로 실행
- topology/scalability 또는 Mininet 대표 시나리오 검증
- observable command no-op mismatch 20,400회 평가 완료: failure probability 0–30%에서 전부 복구, timeout 0, damage 불변
- wall-clock, shield intervention, recovery, damage, timeout 모두 수집

### 11/09–11/12: 논문 자산

- 주 결과표와 main/per-seed/failure-case figure PDF·PNG 생성 완료
- 정직한 claim boundary와 limitations를 포함한 영문 원고 1차본 `paper/DRAFT.md` 작성 완료
- 주장마다 대응하는 실험/표/그림을 연결한 claim-evidence matrix 작성

### 11/13–11/15: 동결과 내부 리뷰

- clean-environment 재현 실행
- 통계/누수/공정성/과장 주장 점검
- 결과, 코드, 환경, 원고를 release candidate로 동결

## 즉시 다음 실행 순서

1. 로컬 Python 설치/가상환경을 정상화한다.
2. `python -m pytest -q`로 현재 기준선을 확정한다.
3. seed 2033 진단 로그 생성기를 추가하고 기존 checkpoint를 분석한다.
4. liveness 수정은 진단 결과가 나온 뒤 하나만 선택해 구현한다.
5. 수정된 프로토콜을 동결한 뒤 seed 확대를 시작한다.

## 중간 의사결정 게이트

- **10/29**: seed 2033 원인이 재현·설명되지 않으면 대규모 재학습 금지
- **11/03**: safety 0 violation과 recovery 개선을 동시에 못 얻으면 연구 질문을 trade-off 분석으로 변경
- **11/08**: testbed가 준비되지 않으면 실제 배포 주장을 삭제하고 simulator/scalability 논문으로 범위 고정
- **11/12**: 새 실험 추가를 중단하고 원고·재현성·검증에만 집중
