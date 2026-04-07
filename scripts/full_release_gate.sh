#!/usr/bin/env bash
set -euo pipefail
set -o errtrace

# Full release gate for hackathon submission.
# Runs:
#   1) local regression + OpenEnv validation
#   2) optional openenv push
#   3) setupCredentials.py
#   4) live API verification
#   5) browser UI smoke verification
#
# Usage:
#   ./scripts/full_release_gate.sh \
#     --base-url "https://<space>.hf.space" \
#     --repo-id "<user>/<space>" \
#     --expect-llm true
#
# Optional flags:
#   --skip-deploy              Skip openenv push step
#   --skip-playwright-install  Skip auto-install of playwright/chromium
#   --expect-llm <true|false>  Whether live_verify should expect llm_used=true

BASE_URL="https://kunalkachru23-medical-triage-env.hf.space"
REPO_ID="kunalkachru23/medical-triage-env"
EXPECT_LLM="true"
SKIP_DEPLOY="false"
SKIP_PLAYWRIGHT_INSTALL="false"
CURRENT_STAGE="initialization"
LAST_CMD=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/full_release_gate.sh [options]

Options:
  --base-url <url>              Live base URL (default: project HF Space)
  --repo-id <user/space>        HF space repo id for openenv push
  --expect-llm <true|false>     Expect llm_used=true in live_verify (default: true)
  --skip-deploy                 Skip openenv push step
  --skip-playwright-install     Skip playwright/chromium installation step
  -h, --help                    Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      [[ $# -ge 2 ]] || { echo "[release-gate] Missing value for --base-url"; exit 2; }
      BASE_URL="$2"
      shift 2
      ;;
    --repo-id)
      [[ $# -ge 2 ]] || { echo "[release-gate] Missing value for --repo-id"; exit 2; }
      REPO_ID="$2"
      shift 2
      ;;
    --expect-llm)
      [[ $# -ge 2 ]] || { echo "[release-gate] Missing value for --expect-llm"; exit 2; }
      EXPECT_LLM="$2"
      shift 2
      ;;
    --skip-deploy)
      SKIP_DEPLOY="true"
      shift
      ;;
    --skip-playwright-install)
      SKIP_PLAYWRIGHT_INSTALL="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[release-gate] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ "$EXPECT_LLM" != "true" && "$EXPECT_LLM" != "false" ]]; then
  echo "[release-gate] --expect-llm must be 'true' or 'false'."
  exit 2
fi

if [[ ! "$BASE_URL" =~ ^https?:// ]]; then
  echo "[release-gate] --base-url must start with http:// or https://"
  echo "[release-gate] Received: $BASE_URL"
  exit 2
fi

if [[ "$REPO_ID" != */* ]]; then
  echo "[release-gate] --repo-id must look like <username>/<space-name>"
  echo "[release-gate] Received: $REPO_ID"
  exit 2
fi

fail_with_guidance() {
  local stage="$1"
  local line="$2"
  local cmd="$3"
  local code="$4"
  echo
  echo "[release-gate] FAILED during stage: ${stage}"
  echo "[release-gate] line: ${line} | exit_code: ${code}"
  echo "[release-gate] command: ${cmd}"
  echo
  case "$stage" in
    "Environment sanity")
      echo "[release-gate] Fix:"
      echo "  - Activate the intended venv first."
      echo "  - Use Python 3.12 or 3.13 (3.14 is not fully supported by pinned deps)."
      ;;
    "Local regression (pytest + validate + pre_submit_check)")
      echo "[release-gate] Fix:"
      echo "  - Reinstall deps: python -m pip install -r server/requirements.txt -r requirements.txt"
      echo "  - Run failing command directly to inspect details:"
      echo "    pytest tests/ -q"
      echo "    openenv validate"
      echo "    ./scripts/pre_submit_check.sh"
      ;;
    "Deploy to HF Space")
      echo "[release-gate] Fix:"
      echo "  - Check HF auth/token and repo id: $REPO_ID"
      echo "  - Re-run deploy manually:"
      echo "    openenv push --repo-id \"$REPO_ID\""
      ;;
    "Sync Space credentials")
      echo "[release-gate] Fix:"
      echo "  - Ensure .env has: HF_TOKEN, SPACE_REPO_ID, INFERENCE_HF_TOKEN"
      echo "  - Re-run: python setupCredentials.py"
      ;;
    "Live API verification")
      echo "[release-gate] Fix:"
      echo "  - Verify Space is healthy and restarted after credential sync."
      echo "  - Re-run:"
      echo "    BASE_URL=\"$BASE_URL\" EXPECT_LLM=\"$EXPECT_LLM\" ./scripts/live_verify.sh"
      ;;
    "Browser smoke setup")
      echo "[release-gate] Fix:"
      echo "  - Install browser deps:"
      echo "    python -m pip install playwright"
      echo "    python -m playwright install chromium"
      ;;
    "Browser UI smoke verification")
      echo "[release-gate] Fix:"
      echo "  - Re-run smoke directly:"
      echo "    python scripts/browser_ui_smoke.py --base-url \"$BASE_URL\""
      echo "  - If browser install was skipped, run playwright install first."
      ;;
    *)
      echo "[release-gate] General fixes:"
      echo "  1) Activate correct venv (Python 3.12/3.13)."
      echo "  2) Re-run: python setupCredentials.py"
      echo "  3) Re-run live verify and browser smoke scripts."
      ;;
  esac
}

on_err() {
  local code="$?"
  fail_with_guidance "$CURRENT_STAGE" "$LINENO" "$LAST_CMD" "$code"
  exit "$code"
}
trap on_err ERR

run_cmd() {
  LAST_CMD="$*"
  "$@"
}

require_cmd() {
  local cmd="$1"
  local install_hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[release-gate] Missing required command: $cmd"
    echo "[release-gate] $install_hint"
    exit 2
  fi
}

step() {
  local title="$1"
  CURRENT_STAGE="$title"
  echo
  echo "============================================================"
  echo "[release-gate] $title"
  echo "============================================================"
}

require_cmd python "Install Python and activate project venv first."
require_cmd pytest "Install test dependencies in your active venv."
require_cmd openenv "Install OpenEnv CLI in your active venv."
require_cmd curl "Install curl."

if [[ "$SKIP_DEPLOY" != "true" ]]; then
  require_cmd git "Install git."
fi

step "Environment sanity"
echo "[release-gate] python: $(command -v python)"
run_cmd python -V
echo "[release-gate] working directory: $(pwd)"

PY_MAJ_MIN="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$PY_MAJ_MIN" != "3.12" && "$PY_MAJ_MIN" != "3.13" ]]; then
  echo "[release-gate] WARNING: detected Python $PY_MAJ_MIN."
  echo "[release-gate] Recommended version is 3.12 or 3.13 for dependency compatibility."
fi

if [[ ! -f "openenv.yaml" ]]; then
  echo "[release-gate] openenv.yaml not found. Run this script from repo root."
  exit 2
fi
[[ -f "./scripts/pre_submit_check.sh" ]] || { echo "[release-gate] Missing ./scripts/pre_submit_check.sh"; exit 2; }
[[ -f "./scripts/live_verify.sh" ]] || { echo "[release-gate] Missing ./scripts/live_verify.sh"; exit 2; }
[[ -f "./scripts/browser_ui_smoke.py" ]] || { echo "[release-gate] Missing ./scripts/browser_ui_smoke.py"; exit 2; }

step "Local regression (pytest + validate + pre_submit_check)"
run_cmd pytest tests/ -q
run_cmd openenv validate
run_cmd ./scripts/pre_submit_check.sh

if [[ "$SKIP_DEPLOY" != "true" ]]; then
  step "Deploy to HF Space"
  echo "[release-gate] openenv push --repo-id $REPO_ID"
  run_cmd openenv push --repo-id "$REPO_ID"
else
  step "Deploy step skipped"
  echo "[release-gate] --skip-deploy enabled."
fi

step "Sync Space credentials"
run_cmd python setupCredentials.py

step "Live API verification"
LAST_CMD="BASE_URL=\"$BASE_URL\" EXPECT_LLM=\"$EXPECT_LLM\" ./scripts/live_verify.sh"
BASE_URL="$BASE_URL" EXPECT_LLM="$EXPECT_LLM" ./scripts/live_verify.sh

step "Browser smoke setup"
if [[ "$SKIP_PLAYWRIGHT_INSTALL" != "true" ]]; then
  run_cmd python -m pip install playwright
  run_cmd python -m playwright install chromium
else
  echo "[release-gate] --skip-playwright-install enabled."
  run_cmd python - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("playwright") is None:
    print("[release-gate] Playwright is not installed but --skip-playwright-install was set.")
    print("[release-gate] Install with:")
    print("  python -m pip install playwright")
    print("  python -m playwright install chromium")
    sys.exit(2)
PY
fi

step "Browser UI smoke verification"
run_cmd python scripts/browser_ui_smoke.py --base-url "$BASE_URL"

step "Release gate PASSED"
echo "[release-gate] All checks succeeded."
