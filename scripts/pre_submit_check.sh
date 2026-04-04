#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="medical-triage-env:precheck"
CONTAINER_NAME="medical-triage-env-precheck"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT


echo "[1/5] Running test suite"
pytest tests/ -q

echo "[2/5] Building Docker image"
docker build -t "$IMAGE_NAME" .

echo "[3/5] Starting container and checking health"
docker run -d --name "$CONTAINER_NAME" -p 7860:7860 "$IMAGE_NAME" >/dev/null

for i in {1..30}; do
  if curl -fsS "http://localhost:7860/health" >/dev/null; then
    break
  fi
  sleep 1
  if [[ "$i" -eq 30 ]]; then
    echo "Health check failed on http://localhost:7860/health"
    exit 1
  fi
done

echo "[4/5] Smoke-check reset endpoint"
curl -fsS -X POST "http://localhost:7860/reset"   -H "content-type: application/json"   -d '{"task_id":"simple_triage","case_index":0,"seed":42}' >/dev/null

echo "[5/5] Running openenv validate"
if ! command -v openenv >/dev/null 2>&1; then
  echo "openenv CLI not found. Install openenv and re-run this script."
  exit 1
fi
openenv validate

echo "All pre-submission checks passed."
