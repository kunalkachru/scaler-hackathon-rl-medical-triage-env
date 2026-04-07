# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical Triage OpenEnv — an RL training environment for clinical triage using the NHS NEWS2 (National Early Warning Score 2) protocol. Deployed on HuggingFace Spaces. The environment exposes a REST API that agents interact with via reset/step/state endpoints.

## Commands

All commands run from `medical_triage_env/` using the local venv.

```bash
# Run all tests (116 tests)
venv/bin/python -m pytest tests/ -v

# Run a single test file
venv/bin/python -m pytest tests/test_graders.py -v

# Run a single test by name
venv/bin/python -m pytest tests/test_graders.py::test_news2_normal_vitals -v

# Start the server locally (note: local uses port 8000, Docker/HF uses port 7860)
venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference baseline (requires API_BASE_URL, MODEL_NAME, HF_TOKEN env vars)
venv/bin/python inference.py

# Run RL training loop
venv/bin/python train.py

# Pre-submit validation (tests → Docker build → health → reset → openenv validate)
bash scripts/pre_submit_check.sh

# Live API verification against deployed HF Space
BASE_URL="https://<space>.hf.space" EXPECT_LLM=true ./scripts/live_verify.sh

# Browser-based headless UI smoke test (requires playwright)
python -m pip install playwright && python -m playwright install chromium
python scripts/browser_ui_smoke.py --base-url "https://<space>.hf.space"

# One-command full release gate (local + deploy + live API + browser UI)
./scripts/full_release_gate.sh \
  --base-url "https://<space>.hf.space" \
  --repo-id "<user>/<space>" \
  --expect-llm true

# Verification-only (skip deploy step)
./scripts/full_release_gate.sh --skip-deploy --base-url "https://<space>.hf.space" --expect-llm true

# Docker build and run
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

## Architecture

The project has five distinct layers:

### 1. HTTP Server (`server/app.py`)
FastAPI server with session management (30-min TTL for concurrent episode isolation). Sessions are UUID-keyed and held in memory — history is lost on server restart. Key endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /reset` | Start episode, returns `info.session_id` — pass this in all subsequent calls |
| `POST /step` | Submit assessment, returns reward + score_breakdown + done |
| `GET /state` | Episode metadata (query param: `?session_id=`) |
| `POST /grade-fairness` | Multi-variant demographic parity score (separate from step rewards) |
| `POST /suggest` | LLM-backed AI fill for web UI; falls back to rule-based if env vars absent |
| `POST /agent-assess` | Full LLM triage assessment for the web UI's "AI Agent" button |
| `GET /metrics` | Score distributions, difficulty gradient, case coverage |

### 2. Environment State Machine (`server/medical_triage_environment.py`)
`MedicalTriageEnvironment` manages episode lifecycle: selects a patient case on `reset()`, calls the appropriate grader on `step()`, returns a `TriageObservation` with score/feedback/done flag.

### 3. Case Bank (`server/cases.py`)
28 patient cases in `CASE_BANK` dict keyed by `task_id`. Five task types with increasing difficulty:
- `simple_triage` (Easy, 4 cases) — basic NEWS2 calculation
- `conflicting_vitals` (Medium, 3 cases) — misleading vital signs, keyword-matched rationale grading
- `masked_deterioration` (Hard, 5 cases) — pharmacological masking (beta-blockers, steroids, uraemia, adrenal)
- `demographic_fairness` (Medium, 12 cases) — 3 scenarios × 4 demographics; parity only via `/grade-fairness`
- `deteriorating_patient` (Hard, multi-turn, 4 cases) — 3-step trajectory each

### 4. Graders (`server/graders.py`)
Deterministic grading with partial-credit scoring. Key functions:
- `compute_news2(vitals)` — implements NHS NEWS2 algorithm
- `news2_to_priority(score)` — maps score to `low/medium/high/critical`
- `priority_distance(predicted, truth)` — asymmetric penalty: under-triage costs 2× over-triage
- Per-task graders with dimension weights (e.g., simple_triage: 0.4 priority + 0.25 NEWS2 + 0.2 critical_sign + 0.15 recommended_action)
- Rationale in `conflicting_vitals` and `critical_clues` in `masked_deterioration` use **keyword matching** — phrasing must match known keywords to score

### 5. Models (`models.py`)
Pydantic v2 models for all I/O. `TriageAction` is what agents submit (fields are task-specific; missing fields score 0); `TriageObservation` is what the environment returns (includes `score_breakdown` per dimension, `hint` when score < 0.4). All fields use `None → ""` validators to be null-safe.

### Client / Training (`client.py`, `inference.py`, `train.py`)
`client.py` wraps the HTTP API. `inference.py` is the mandatory baseline using OpenAI client — **note**: it imports `from server.cases import CASE_BANK` directly (line 265), which requires the `server/` directory to be present locally even when running against a remote HF Space. `train.py` implements reward-conditioned prompting: previous episode scores are injected into the system prompt to improve future actions, using the same fixed case per task to isolate learning signal.

## Browser UI Testing (Playwright MCP)

Playwright MCP is configured — use it to validate the `/web` UI directly without running `scripts/browser_ui_smoke.py`. Start the server first, then use these tools:

```
browser_navigate    → http://localhost:8000/web
browser_screenshot  → verify page rendered correctly
browser_click       → interact with form elements
browser_fill        → fill in triage fields
```

To reconfigure: `claude mcp add playwright npx @playwright/mcp@latest`

## Key Design Decisions

- **Asymmetric penalty**: Under-triage (missing severity) is penalized 2× vs over-triage — intentional and clinically motivated.
- **Demographic fairness parity**: The full parity score requires calling `POST /grade-fairness` with responses from all demographic variants of a case group. The per-step reward in `demographic_fairness` only scores individual case correctness, not parity.
- **Multi-turn deterioration**: Each case spans up to 3 steps (T=0, T=30, T=60 min); T=30 is the critical moment (reward=1.0). The `patient_history` field in the observation is updated with new vitals on each step — agents must re-read it.
- **Session isolation**: Each `POST /reset` creates an independent session; pass the returned `session_id` in all subsequent calls. Omitting it falls back to a default shared session (backward compat only).
- **Keyword matching in graders**: `rationale` (conflicting_vitals) and `critical_clues` (masked_deterioration) are scored by matching against known keyword lists. Correct reasoning phrased differently may score 0.
