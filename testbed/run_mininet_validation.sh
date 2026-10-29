#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This validation requires Linux." >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo -E: sudo -E bash testbed/run_mininet_validation.sh" >&2
  exit 2
fi

run_id="${MININET_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="results/testbed/runs/${run_id}"
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing run: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

python_bin="${REPRO_PYTHON:-.venv/bin/python}"
cleanup() { mn -c >/dev/null 2>&1 || true; }
trap cleanup EXIT

"$python_bin" -m pytest tests/test_mininet_replay.py -q \
  --junitxml="$output_dir/preflight_pytest.xml"
"$python_bin" -m testbed.mininet_vlan_replay \
  --trajectories results/multiseed_progress/progress_safe_seed_2026_trajectories.json \
  --output "$output_dir/mininet_replay_seed_2026.json"

"$python_bin" - "$output_dir/mininet_replay_seed_2026.json" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
s = report["summary"]
n = s["scenario_count"]
ok = n == 51 and s["simulator_testbed_recovery_matches"] == n and s["simulator_testbed_damage_matches"] == n
print(json.dumps({"publishable_match": ok, **s}, indent=2))
raise SystemExit(0 if ok else 1)
PY

echo "Validation evidence: $output_dir"
