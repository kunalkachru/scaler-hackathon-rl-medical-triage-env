#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/artifacts/gates"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/full_browser_test_latest.log"

run_full_browser() {
  python scripts/full_browser_test.py "$@" 2>&1 | tee "$LOG_FILE"
  return "${PIPESTATUS[0]}"
}

is_known_transient_timeout() {
  python - "$LOG_FILE" <<'PY'
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
patterns = [
    "Timeout 30000ms exceeded",
    "TimeoutError",
    "navigation timeout",
]
hit = any(p.lower() in txt.lower() for p in patterns)
sys.exit(0 if hit else 1)
PY
}

echo "[full-browser-retry] Attempt 1: python scripts/full_browser_test.py $*"
if run_full_browser "$@"; then
  echo "[full-browser-retry] PASS on first attempt."
  exit 0
fi

if is_known_transient_timeout; then
  echo "[full-browser-retry] Known transient timeout detected; retrying once."
  echo "[full-browser-retry] Attempt 2: python scripts/full_browser_test.py $*"
  if run_full_browser "$@"; then
    echo "[full-browser-retry] PASS on retry."
    exit 0
  fi
  echo "[full-browser-retry] FAIL after retry."
  exit 1
fi

echo "[full-browser-retry] FAIL (non-transient error; no retry)."
exit 1
