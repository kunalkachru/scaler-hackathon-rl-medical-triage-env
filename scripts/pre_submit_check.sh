#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="medical-triage-env:precheck"
CONTAINER_NAME="medical-triage-env-precheck"
SCRIPT_START_EPOCH="$(date +%s)"
SUMMARY_DIR="${ROOT_DIR}/artifacts/gates"
SUMMARY_FILE="${SUMMARY_DIR}/pre_submit_check_summary.json"

wait_for_health() {
  local url="$1"
  local max_attempts="${2:-30}"
  local delay_sec="${3:-1}"
  local connect_timeout_sec="${4:-3}"

  # First probe can race uvicorn bind; brief settle reduces transient reset noise.
  sleep 2
  for i in $(seq 1 "$max_attempts"); do
    if curl -fsS --connect-timeout "$connect_timeout_sec" "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay_sec"
  done
  return 1
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

write_summary() {
  local exit_code="$1"
  local script_end_epoch
  script_end_epoch="$(date +%s)"
  local duration_sec=$(( script_end_epoch - SCRIPT_START_EPOCH ))
  local status="passed"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
  fi

  mkdir -p "$SUMMARY_DIR"
  python - "$SUMMARY_FILE" "$status" "$duration_sec" "$exit_code" <<'PY'
import datetime
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
duration_sec = int(sys.argv[3])
exit_code = int(sys.argv[4])
payload = {
    "command": "./scripts/pre_submit_check.sh",
    "status": status,
    "duration_sec": duration_sec,
    "exit_code": exit_code,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
out_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

on_exit() {
  local exit_code="$?"
  cleanup
  write_summary "$exit_code"
}
trap on_exit EXIT


echo "[1/5] Running test suite"
pytest tests/ -q

echo "[2/5] Building Docker image"
docker build -t "$IMAGE_NAME" .

echo "[3/5] Starting container and checking health"
docker run -d --name "$CONTAINER_NAME" -p 7860:7860 "$IMAGE_NAME" >/dev/null

if ! wait_for_health "http://localhost:7860/health" 30 1 3; then
  echo "Health check failed on http://localhost:7860/health after 30 attempts"
  exit 1
fi

echo "[4/5] Smoke-check reset endpoint"
curl -fsS -X POST "http://localhost:7860/reset"   -H "content-type: application/json"   -d '{"task_id":"simple_triage","case_index":0,"seed":42}' >/dev/null

echo "[5/5] Running openenv validate"
if ! command -v openenv >/dev/null 2>&1; then
  echo "openenv CLI not found. Install openenv and re-run this script."
  exit 1
fi
openenv validate

echo "All pre-submission checks passed."
