#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="https://kunalkachru23-medical-triage-env.hf.space"
REPO_ID="kunalkachru23/medical-triage-env"
EXPECT_LLM="true"
SKIP_DEPLOY="false"
SKIP_PLAYWRIGHT_INSTALL="false"
RESUME_RUN="false"
ORIGINAL_ARGS=("$@")
SCRIPT_COMMAND="./scripts/final_submission_check.sh ${ORIGINAL_ARGS[*]}"

START_EPOCH="$(date +%s)"
STATE_DIR="${ROOT_DIR}/artifacts/gates"
CHECKPOINT_FILE="${STATE_DIR}/final_submission_check.checkpoint"
SUMMARY_FILE="${STATE_DIR}/final_submission_check_summary.json"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/final_submission_check.sh [options]

Options:
  --base-url <url>              Live base URL (default: production space)
  --repo-id <user/space>        HF Space repo id for deploy step
  --expect-llm <true|false>     Expect llm_used=true in live checks (default: true)
  --skip-deploy                 Skip openenv push in release gate
  --skip-playwright-install     Skip playwright installation in release gate
  --resume                      Resume from checkpoint; otherwise defaults to fresh full run
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --expect-llm) EXPECT_LLM="$2"; shift 2 ;;
    --skip-deploy) SKIP_DEPLOY="true"; shift ;;
    --skip-playwright-install) SKIP_PLAYWRIGHT_INSTALL="true"; shift ;;
    --resume) RESUME_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[final-check] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "$STATE_DIR"

if [[ "$RESUME_RUN" != "true" ]]; then
  rm -f "$CHECKPOINT_FILE"
fi

touch "$CHECKPOINT_FILE"

is_stage_done() {
  local stage_id="$1"
  grep -Fxq "$stage_id" "$CHECKPOINT_FILE"
}

mark_stage_done() {
  local stage_id="$1"
  if ! is_stage_done "$stage_id"; then
    echo "$stage_id" >> "$CHECKPOINT_FILE"
  fi
}

write_summary() {
  local status="$1"
  local failed_stage="${2:-}"
  local exit_code="${3:-0}"
  local end_epoch
  end_epoch="$(date +%s)"
  local duration_sec=$(( end_epoch - START_EPOCH ))
  python - "$SUMMARY_FILE" "$status" "$duration_sec" "$failed_stage" "$exit_code" "$CHECKPOINT_FILE" "$BASE_URL" "$REPO_ID" "$SCRIPT_COMMAND" <<'PY'
import datetime
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
duration_sec = int(sys.argv[3])
failed_stage = sys.argv[4]
exit_code = int(sys.argv[5])
checkpoint_file = pathlib.Path(sys.argv[6])
base_url = sys.argv[7]
repo_id = sys.argv[8]
command = sys.argv[9]

stages = []
if checkpoint_file.exists():
    stages = [ln.strip() for ln in checkpoint_file.read_text().splitlines() if ln.strip()]

payload = {
    "command": command.strip(),
    "status": status,
    "duration_sec": duration_sec,
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_url": base_url,
    "repo_id": repo_id,
    "completed_stages": stages,
}
if failed_stage:
    payload["failed_stage"] = failed_stage
if exit_code:
    payload["exit_code"] = exit_code

summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_stage() {
  local stage_id="$1"
  local stage_title="$2"
  shift 2

  if is_stage_done "$stage_id"; then
    echo "============================================================"
    echo "[final-check] SKIP (already completed): ${stage_title}"
    echo "============================================================"
    return 0
  fi

  echo "============================================================"
  echo "[final-check] ${stage_title}"
  echo "============================================================"

  local attempts=0
  local rc=0
  while (( attempts < 2 )); do
    attempts=$(( attempts + 1 ))
    set +e
    "$@"
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      mark_stage_done "$stage_id"
      return 0
    fi
    # Retry once when process was externally terminated (common wrappers: SIGTERM=143).
    if [[ "$rc" -eq 143 && "$attempts" -lt 2 ]]; then
      echo "[final-check] Stage '${stage_id}' terminated (exit ${rc}); retrying once..."
      continue
    fi
    write_summary "failed" "$stage_id" "$rc"
    echo "[final-check] FAIL at stage '${stage_id}' (exit ${rc})"
    exit "$rc"
  done
}

gate_cmd=(./scripts/full_release_gate.sh --base-url "$BASE_URL" --repo-id "$REPO_ID" --expect-llm "$EXPECT_LLM")
if [[ "$SKIP_DEPLOY" == "true" ]]; then
  gate_cmd+=(--skip-deploy)
fi
if [[ "$SKIP_PLAYWRIGHT_INSTALL" == "true" ]]; then
  gate_cmd+=(--skip-playwright-install)
fi

run_stage "coverage" "1/4 Coverage parity" python scripts/check_coverage.py
echo
run_stage "pre_submit" "2/4 Local pre-submit gate" ./scripts/pre_submit_check.sh
echo
run_stage "full_browser" "3/4 Full browser+API suite (one retry on known flake)" ./scripts/run_full_browser_with_retry.sh --base-url "$BASE_URL"
echo
run_stage "release_gate" "4/4 Release gate (local + live + browser smoke)" "${gate_cmd[@]}"

write_summary "passed" "" "0"

echo
echo "[final-check] PASS — go/no-go result: GO"
echo "[final-check] Artifacts:"
echo "  - artifacts/gates/pre_submit_check_summary.json"
echo "  - artifacts/gates/full_release_gate_summary.json"
echo "  - artifacts/gates/final_submission_check_summary.json"
