#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="https://kunalkachru23-medical-triage-env.hf.space"
REPO_ID="kunalkachru23/medical-triage-env"
EXPECT_LLM="true"
SKIP_DEPLOY="false"
SKIP_PLAYWRIGHT_INSTALL="false"

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
    -h|--help) usage; exit 0 ;;
    *) echo "[final-check] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

echo "============================================================"
echo "[final-check] 1/4 Coverage parity"
echo "============================================================"
python scripts/check_coverage.py

echo
echo "============================================================"
echo "[final-check] 2/4 Local pre-submit gate"
echo "============================================================"
./scripts/pre_submit_check.sh

echo
echo "============================================================"
echo "[final-check] 3/4 Full browser+API suite (one retry on known flake)"
echo "============================================================"
./scripts/run_full_browser_with_retry.sh --base-url "$BASE_URL"

echo
echo "============================================================"
echo "[final-check] 4/4 Release gate (local + live + browser smoke)"
echo "============================================================"
gate_cmd=(./scripts/full_release_gate.sh --base-url "$BASE_URL" --repo-id "$REPO_ID" --expect-llm "$EXPECT_LLM")
if [[ "$SKIP_DEPLOY" == "true" ]]; then
  gate_cmd+=(--skip-deploy)
fi
if [[ "$SKIP_PLAYWRIGHT_INSTALL" == "true" ]]; then
  gate_cmd+=(--skip-playwright-install)
fi
"${gate_cmd[@]}"

echo
echo "[final-check] PASS — go/no-go result: GO"
echo "[final-check] Artifacts:"
echo "  - artifacts/gates/pre_submit_check_summary.json"
echo "  - artifacts/gates/full_release_gate_summary.json"
