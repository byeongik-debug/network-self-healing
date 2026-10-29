#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "Clean reproduction requires a committed, clean checkout." >&2
  exit 2
fi

run_id="${REPRO_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="results/reproduction/runs/${run_id}"
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing run: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

venv_dir="$(mktemp -d "${TMPDIR:-/tmp}/safe-dqn-repro.XXXXXX")"
cleanup() { rm -rf -- "$venv_dir"; }
trap cleanup EXIT

python3 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r requirements.txt
"$venv_dir/bin/python" -m pytest -q --junitxml="$output_dir/pytest.xml"
"$venv_dir/bin/python" -m pip freeze > "$output_dir/pip-freeze.txt"
git rev-parse HEAD > "$output_dir/git-commit.txt"

"$venv_dir/bin/python" - "$output_dir" <<'PY'
import json, platform, sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
root = ET.parse(out / "pytest.xml").getroot()
suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
totals = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
          for key in ("tests", "failures", "errors", "skipped")}
report = {
    "schema_version": 1,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": (out / "git-commit.txt").read_text().strip(),
    "python": sys.version,
    "platform": platform.platform(),
    "pytest": totals,
    "clean_reproduction_passed": totals["failures"] == 0 and totals["errors"] == 0,
}
(out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "Clean reproduction evidence: $output_dir"
