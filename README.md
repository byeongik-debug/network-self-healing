# Network-Self-Healing

Safe Reinforcement Learning 기반 네트워크 설정 장애 자동 진단 및 복구 연구를 위한 순수 Python 실험 환경이다. 현재 Phase 2.6까지 구현되어 있으며 학습 모델과 보상 함수는 포함하지 않는다.

## 현재 범위

- 스위치 2대, 호스트 4대, VLAN 10/20, 스위치 간 trunk
- 고정된 19개 이산 행동과 상태 기반 action mask
- 미진단/성공/실패 연결성 관측과 단계별 rollback
- Rule-Based 및 Configuration Diff baseline
- 고정 시나리오 6개와 생성 가능한 장애 조합 255개
- 기능 복구와 설정 복구를 분리한 평가 및 CSV 저장
- seed manifest를 저장하는 legacy/stratified train-eval split
- DQN 입력 준비용 79차원 NumPy observation encoder
- 복구 과정의 누적·고유·최종 서비스 손상 지표

DQN, Safe DQN, 신경망, Replay Buffer, 학습 루프, 보상 함수와 Mininet 검증은 구현하지 않는다. `NetworkEnv.calculate_reward()`는 항상 `0.0`을 반환한다.

## 설치와 테스트

Python 3.10 이상을 권장한다.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
```

## 장애 조합

Phase 2.5는 다음 8개의 독립적인 원자 장애를 사용한다.

- 4개 access 포트의 VLAN 10/20 반전
- SW1/SW2 trunk에서 VLAN 10 또는 VLAN 20 누락

각 원자 장애는 설정의 서로 다른 한 항목을 변경한다. 비어 있지 않은 모든 부분집합을 생성하므로 고유 장애 조합 수는 `2^8 - 1 = 255`개다. 정상 상태는 별도의 `normal` 사례이며 이 수에 포함하지 않는다.

분류 기준:

- 유형: `single_access`, `single_trunk`, `multi_access`, `multi_trunk`, `access_trunk_combined`
- 심각도: 원자 장애 1개는 `low`, 2~3개는 `medium`, 4개 이상은 `high`
- 영향: 연결성 또는 VLAN 격리 정책을 실제로 깨뜨리면 functional fault, 설정만 다르고 정책이 정상이라면 configuration-only fault

`FaultDataset`은 canonical scenario ID를 SHA-256으로 고정 분할한다. 따라서 train/eval 구분은 sampling seed와 독립적이고 동일 조합이 양쪽에 나타나지 않는다. 현재 분할은 train 202개, eval 53개이며 eval은 학습에서 제외된 조합만 포함한다. seed는 각 split 안에서 재현 가능한 표본을 선택할 때만 사용한다.

현재 분포에서 train은 low 8개, medium 74개, high 120개이고 eval은 medium 10개, high 43개다. 따라서 eval이 의도상 더 어려운 조합으로 구성되지만, 작은 토폴로지에서 파생된 유한 조합이라는 한계가 있다.

Phase 2.6의 stratified split은 severity와 fault type 조합별로 가능한 경우 train/eval 양쪽에 사례를 배치한다. 기본 seed는 2026이며 train 204개, eval 51개다. train 분포는 low 6/medium 67/high 131, eval 분포는 low 2/medium 17/high 32다. 조합 ID와 seed는 `results/stratified_split_manifest.json`에 저장된다. 원소가 하나뿐인 세부 stratum은 중복 없이 양쪽에 배치할 수 없으므로 train에 남긴다.

## 관측과 정보 접근 권한

reset 직후 연결성 관측은 모두 `unknown`이다. 진단 후에만 `success` 또는 `failure`가 공개되고, 설정 변경이나 rollback 뒤에는 기존 결과가 다시 `unknown`으로 무효화된다. observation에는 scenario ID, 장애 원인, 기준 설정, `policy_healthy`가 없다.

Rule-Based는 observation, 진단 결과와 현재 유효 행동만 사용한다. Configuration Diff만 `REFERENCE_PORTS`를 읽는 Oracle 비교군이다. 기능/설정 복구 판정은 `experiments/evaluation.py`에 분리되어 알고리즘 입력으로 사용되지 않는다.

다만 현재 포트 설정 자체는 완전히 관측 가능하며 토폴로지가 매우 작고 대칭적이다. 반복 관찰을 통해 정답 VLAN 패턴을 쉽게 추론하거나 암기할 가능성이 있다. 향후에는 포트 역할 인코딩, 관측 이력, 진단 비용, 제한적/노이즈 진단, topology permutation 중 어떤 부분 관측 모델을 사용할지 연구자가 결정해야 한다. Phase 2.5에서는 관측 구조를 변경하지 않았다.

## Observation Encoder

`env/observation_encoder.py`는 기존 dictionary observation만 읽어 shape `(79,)`, dtype `float32` NumPy 배열을 만든다. 입력 dictionary를 변경하지 않으며 기준 설정, 장애 ID, 실제 원인, 정답 행동과 `policy_healthy`를 추가하지 않는다.

인덱스 구성:

- `0..35`: 고정 포트 순서 `SW1:eth1, eth2, eth3, SW2:eth1, eth2, eth3`; 각 포트마다 `mode_access, mode_trunk, access_vlan_10, access_vlan_20, trunk_allows_10, trunk_allows_20`
- `36..41`: 같은 포트 순서의 link-up 값
- `42..77`: `H1..H4` source/destination 방향성 12개 pair; 각 pair마다 `unknown, success, failure` one-hot
- `78`: 현재 `step_count`

`ObservationEncoder.feature_names()`가 79개 인덱스의 이름을 동일 순서로 반환한다. 현재 observation history는 포함하지 않으므로 과거 진단이나 행동이 필요한 상태는 하나의 벡터만으로 구별할 수 없다.

## Action Mask와 종료

action ID는 항상 `0..18`이고 mask 인덱스와 일치한다. 진단과 완료 선언은 항상 허용되며 rollback은 실제 변경 이력이 있을 때만 허용된다. 따라서 episode 진행 중 유효 행동이 전혀 없는 상태는 없다. mask는 현재 설정 및 rollback 이력만 사용하고 기준 설정을 참조하지 않는다.

완료 선언은 진단을 선행 조건으로 요구하지 않는다. 미복구 상태에서 선언해도 `terminated=True`가 되지만 `recovery_success=False`이고 평가 결과는 `incorrect_completion`이다. 30 step 도달은 `truncated=True`와 `max_steps_exceeded`로 구분된다. 종료 뒤 추가 `step()` 호출은 기존처럼 오류가 발생한다.

## 정상 서비스 손상 지표

기존 `healthy_network_impacts`는 새 손상을 한 번 이상 만든 설정 변경 행동 수로 유지한다. 추가 지표는 다음과 같다.

- `initial_damaged_policies`: 장애 주입 직후 이미 손상된 host-pair 정책 수
- `new_policy_damage_events`: 행동 전 정상에서 행동 후 손상으로 바뀐 pair의 누적 횟수
- `unique_newly_damaged_policies`: 복구 과정에서 새로 손상된 고유 pair 수
- `service_damage_actions`: 하나 이상의 새 손상을 만든 상태 변경 행동 수
- `continuously_healthy_policies`: 초기부터 현재까지 계속 정상인 pair 수
- `final_damaged_policies`: 종료 시점에 손상된 pair 수
- `post_recovery_damage_events`: 전체 기능 복구 상태에서 다시 장애 상태로 전환된 횟수

이 판정은 환경/평가 내부에서만 수행하며 observation에는 포함되지 않는다. 같은 pair가 복구 후 다시 손상되면 누적 이벤트는 증가하지만 고유 pair 수는 증가하지 않는다.

## Baseline

Rule-Based는 양쪽 스위치의 동일 이름 access 포트 VLAN 일관성, 스위치 내부 VLAN 분리, 사용 중인 access VLAN의 trunk 허용 규칙을 적용한다. Configuration Diff는 현재 설정과 기준 설정의 차이만 변경한다. 후자는 정답 설정을 아는 Oracle이므로 향후 DQN과 동등한 정보 조건의 비교군이 아니다.

## 실험 실행

고정 6개 시나리오:

```bash
python -m experiments.runner
```

255개 확장 조합 전체:

```bash
python -m experiments.extended_runner
```

기본 `ExtendedExperimentRunner`는 legacy split을 재현한다. 모듈 실행은 seed 2026 stratified split을 평가하여 `results/stratified_baseline_results.csv`와 manifest를 저장한다. 기존 결과는 `results/extended_baseline_results.csv`에 보존된다. CSV는 episode, 장애 유형 평균, 심각도 평균, 전체 평균을 구분한다. `python_simulation_seconds`는 Python 벽시계 시간이며 실제 네트워크 복구 시간이 아니다.

## 주요 파일

- `config/`: 기준 토폴로지와 기존 고정 시나리오
- `env/faults.py`: 조합 생성, 분류, 고정 train/eval 분리와 seed 표본 추출
- `env/network_env.py`: 19개 행동 환경, 진단, mask, rollback
- `env/observation_encoder.py`: 공개 observation의 79차원 결정적 인코딩
- `baselines/`: observation-only Rule-Based와 Oracle Configuration Diff
- `experiments/evaluation.py`: 기능 및 설정 복구의 privileged 평가
- `experiments/extended_runner.py`: 확장 조합 평가와 집계 CSV
- `tests/`: Phase 1~2.5 회귀 및 기능 테스트
