# 체크포인트 저장 방식 검토

저장 방식 변경은 아직 실행하지 않았다.

## Git LFS 방식

장점:

- Checkpoint paths and versions remain aligned with commits.
- Clone and checkout workflows are convenient during collaboration.
- Individual model versions can be retrieved without unpacking an archive.

단점:

- Hosting quotas and bandwidth limits add maintenance burden.
- Forks and long-term retrieval depend on LFS availability.
- Intermediate checkpoints still accumulate in retained LFS history.

## 별도 압축 자료 방식

장점:

- Keeps Git history small and suits a frozen paper release.
- One archive can contain checksums, metadata, and selected checkpoints.
- Zenodo or an institutional repository can provide a durable DOI.

단점:

- Checkpoints are less convenient during active development.
- Code and model versions can drift without commit and hash metadata.
- Changing one checkpoint requires publishing a new archive.

## 권고안

이 논문 저장소에는 논문 재현에 필요한 체크포인트, 정확한 Git 커밋과 SHA-256
해시만 포함한 DOI 기반 별도 자료 저장소를 권장한다. 여러 개발자가 체크포인트를
빈번하게 갱신하는 방식이 주 작업 흐름이 되는 경우에만 Git LFS를 우선 고려한다.
현재 LFS 설정이나 압축 자료 생성은 수행하지 않았다.
