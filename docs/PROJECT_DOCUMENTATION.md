# Medical Triage RL Environment — Project Documentation

**Version:** v2.3.0  
**Team:** Team Falcons — Scaler × Meta PyTorch OpenEnv Hackathon 2026  
**Live Space:** https://huggingface.co/spaces/kunalkachru23/medical-triage-env  
**API URL:** https://kunalkachru23-medical-triage-env.hf.space  
**GitHub:** https://github.com/kunalkachru/scaler-hackathon-rl-medical-triage-env  
**Tests (latest local gate):** 345 collected, 331 passed, 14 skipped

> NOTE: This document started as an early-phase runbook. For current implementation truth, prefer `README.md`, `openenv.yaml`, and `docs/TECHNICAL_REPORT.md`.

---

## 1. Real-World Overview and Objective

In a real emergency department, a triage nurse must decide urgency quickly from incomplete or noisy information. This project simulates that exact decision process as an OpenEnv-compatible reinforcement learning environment. The agent reads a realistic patient presentation (history, vitals, medications, trends), then submits a triage decision that deterministic clinical graders evaluate.

**What we achieve:**
- A clinically meaningful RL benchmark grounded in real medical standards (NHS NEWS2)
- Captures both straightforward and failure-prone scenarios — including pharmacological masking and demographic bias
- Transparent, reproducible reward functions with partial credit at every grader dimension
- Dual-purpose: supports both evaluation (benchmark scoring) and training (RL loop via `train.py`)

**Why it matters:**
- Under-triage can delay life-saving care
- Over-triage wastes limited emergency resources
- Clinical AI evaluation requires deterministic, traceable, explainable scoring
- RL training needs dense reward signals, not binary pass/fail labels

---

## 2. Solution Scope

### 2.1 Tasks

| Task | Difficulty | Cases | Episode Type | Research Basis |
|---|---|---|---|---|
| `simple_triage` | Easy | 10 | Single-step | NHS NEWS2 RCP 2017 |
| `conflicting_vitals` | Medium | 8 | Single-step | Clinical reasoning literature |
| `masked_deterioration` | Hard | 10 | Single-step | Pharmacology masking literature |
| `demographic_fairness` | Medium | 12 | Single-step | Nature Medicine 2025, Lancet Digital Health |
| `deteriorating_patient` | Hard | 7 | Multi-turn | MIMIC-III, npj Digital Medicine 2025 |
| `sepsis_bundle` | Hard | 4 | Single-step | Surviving Sepsis Campaign 2021 |
| `paediatric_triage` | Hard | 6 | Single-step | PEWS / RCPCH guidance |
| `medication_reconciliation` | Hard | 6 | Single-step | NPSA/BNF/MHRA medication safety |
| `icu_deterioration` | Hard | 4 | Single-step | SOFA framework / ICU deterioration |
| `sbar_handover` | Medium | 4 | Single-step | NHS SBAR framework |
| `differential_diagnosis` | Hard | 4 | Single-step | Diagnostic safety-net framework |
| **Total** | | **28** | | |

### 2.2 Core Interface

```
POST /reset   → initial observation
POST /step    → scored observation + reward + done
GET  /state   → episode metadata
GET  /health  → 200 {"status":"healthy"}
```

---

## 3. Technical Architecture

### 3.1 Project Structure

```
medical-triage-env/
├── inference.py              ← Mandatory baseline script (OpenAI client, env vars)
├── train.py                  ← RL training loop (reward-conditioned prompting)
├── models.py                 ← Typed Pydantic models: TriageAction, TriageObservation, TriageState
├── client.py                 ← Python HTTP client wrapper
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← python:3.12-slim, port 7860, HEALTHCHECK
├── setupCredentials.py       ← HF Space secrets/variables configuration helper
├── __init__.py
├── server/
│   ├── app.py                ← FastAPI service — all endpoints, session manager, episode history
│   ├── medical_triage_environment.py  ← Environment state machine (reset/step/state)
│   ├── cases.py              ← Patient case bank (28 cases)
│   ├── graders.py            ← Deterministic graders (NEWS2, fairness, deterioration)
│   ├── requirements.txt      ← fastapi, uvicorn, pydantic, openai, requests
│   └── __init__.py
├── tests/
│   ├── test_graders.py       ← 31 grader unit tests
│   ├── test_environment.py   ← 24 environment integration tests
│   ├── test_v2_enhancements.py ← 44 v2 feature tests
│   ├── test_api_contract.py  ← 9 API contract tests
│   ├── test_ui_contract.py   ← 8 UI contract tests
│   └── test_inference_contract.py ← 3 baseline inference contract tests
├── scripts/
│   ├── pre_submit_check.sh   ← Pre-submission validation pipeline
│   ├── validate-submission.sh ← Organizer-style validator parity script (mirrors sample naming/comments/layout)
│   ├── live_verify.sh        ← Live API verification for deployed Space
│   ├── browser_ui_smoke.py   ← Browser smoke validation for deployed UI
│   └── full_release_gate.sh  ← One-command release/evaluator gate
└── docs/
    ├── PROJECT_DOCUMENTATION.md  ← This file
    └── TEST_REPORT.md        ← Exhaustive test narrative
```

### 3.2 Request/Response Flow

```
Client                         Server (FastAPI)           Environment
  │                                │                          │
  ├─ POST /reset ──────────────────►│                          │
  │  {task_id, case_index, seed}    ├─ new_session() ─────────►│
  │                                │◄─ StepResult ────────────┤
  │◄── {observation, session_id} ──┤                          │
  │                                │                          │
  ├─ POST /step ───────────────────►│                          │
  │  {action, session_id}          ├─ step(action) ──────────►│
  │                                │   grader(action, case) ──┤
  │                                │◄─ reward, breakdown ─────┤
  │◄── {observation, reward, done}─┤                          │
  │                                │                          │
  ├─ GET /state?session_id= ───────►│                          │
  │◄── {step_count, cumulative_reward, tasks_completed} ───────┤
```

### 3.3 Session Management

Every `POST /reset` creates a fresh session with its own `MedicalTriageEnvironment` instance. Sessions are identified by a `session_id` returned in `info`. Pass this ID in subsequent `step()` and `state()` calls to ensure correct episode routing when multiple agents run concurrently.

- Default session (`_default`) handles legacy clients that omit `session_id`
- Sessions expire after 30 minutes of inactivity
- Concurrent isolation is fully tested (`test_api_contract.py::test_session_isolation_concurrent_episodes`)

### 3.4 Episode History and Metrics

Every completed episode is recorded in-memory. Three endpoints expose this data:

- `GET /history` — chronological list of all scored episodes (powers training curve)
- `GET /stats` — per-task avg/best/count
- `GET /metrics` — rich evaluator view: score distributions (min/max/p25/p75), case coverage, difficulty gradient verification

---

## 4. Functional Design

### 4.1 Action Space — `TriageAction`

All fields are optional at the model level. Missing fields receive 0 partial credit. `priority: null` is coerced to `""` via a Pydantic validator (null-safe since v2.0.0 — prevents 422 errors from agents that explicitly send null).

| Field | Type | Tasks | Description |
|---|---|---|---|
| `priority` | `Optional[str]` | All | `"low"` \| `"medium"` \| `"high"` \| `"critical"` |
| `news2_score` | `Optional[int]` | 1, 2 | Agent's computed NEWS2 total |
| `critical_sign` | `Optional[str]` | 1, 2, 4 | Most dangerous vital parameter |
| `recommended_action` | `Optional[str]` | 1, 2, 4 | `"emergency_response"` \| `"urgent_review"` \| `"routine_monitoring"` |
| `misleading_signs` | `Optional[list[str]]` | 2 | Signs that appear normal but are deceptive |
| `condition` | `Optional[str]` | 2, 3 | Suspected clinical diagnosis |
| `masking_drug_or_condition` | `Optional[str]` | 3 | Drug or condition suppressing warning signs |
| `masked_sign` | `Optional[str]` | 3 | Pharmacologically suppressed vital parameter |
| `critical_clues` | `Optional[list[str]]` | 3 | Non-vital-sign evidence of true severity |
| `action` | `Optional[str]` | 5 | `"monitor"` \| `"escalate"` \| `"emergency_response"` |
| `rationale` | `Optional[str]` | All | Free-text clinical reasoning |
| `confidence` | `Optional[float]` | All | Confidence [0.0, 1.0] — triggers calibration bonus |

### 4.2 Observation Space — `TriageObservation`

| Field | Type | Description |
|---|---|---|
| `patient_history` | `str` | Full patient case (the agent reads this) |
| `task_id` | `str` | Active task |
| `task_description` | `str` | Task-specific instructions |
| `score` | `Optional[float]` | Step reward — None before first step |
| `score_breakdown` | `Optional[dict]` | Per-dimension reward breakdown |
| `feedback` | `Optional[str]` | Textual explanation of the score |
| `done` | `bool` | True when episode is complete |
| `step_number` | `int` | 0 after reset, increments each step |
| `case_id` | `Optional[str]` | Current case identifier (e.g. "ST001") |
| `available_tasks` | `Optional[list[str]]` | All task IDs |
| `hint` | `Optional[str]` | Clinical hint when score < 0.4 |

### 4.3 State — `TriageState`

Returned by `GET /state`. Tracks episode-level metadata:

| Field | Description |
|---|---|
| `episode_id` | Unique episode identifier |
| `step_count` | Steps taken in this episode |
| `current_task_id` | Active task |
| `current_case_id` | Active case |
| `cumulative_reward` | Sum of all step rewards |
| `tasks_completed` | List of completed task IDs |
| `scores_per_task` | Score achieved per task |
| `is_done` | Whether current episode is complete |

### 4.4 Reward and Grading Design

**Principles:**
- Fully deterministic — same action on same case always gives same score
- Partial credit at every dimension — not binary
- Grader functions compute in **`[0.0, 1.0]`**; **HTTP** `reward` / `observation.score` / **`/grade-fairness`** are remapped to **`(0, 1)`** via `task_score_for_api()` (see `models.py`). **`/reset`** remains **`reward = 0.0`**.
- Asymmetric under/over-triage penalty (under-triage penalised harder)
- Optional confidence calibration bonus

**Task 1 — Simple Triage:**
```
priority .......... 0.40  (exact=1.0, off-by-1=0.5, off-by-2+=0.0)
news2_score ....... 0.25  (Δ=0→1.0, Δ=1→0.7, Δ=2→0.3, Δ≥3→0.0)
critical_sign ..... 0.20  (exact=1.0, wrong=0.2, none=0.0)
recommended_action  0.15  (exact group=1.0, adjacent=0.4, wrong=0.0)
```

**Task 2 — Conflicting Vitals:**
```
priority .......... 0.35
critical_sign ..... 0.25  (fell for trap → 0.0)
misleading_signs .. 0.20  (fraction of true misleading signs identified)
rationale ......... 0.20  (clinical keyword matching against ground truth)
```

**Task 3 — Masked Deterioration:**
```
priority .......... 0.30  (+ 0.5× extra penalty for missing critical)
masking_mechanism . 0.25  (drug/condition name keyword match)
masked_sign ....... 0.25  (vital sign keyword match)
critical_clues .... 0.20  (fraction of non-standard evidence used)
```

**Task 4 — Demographic Fairness (single-variant step scoring):**
```
priority .......... 0.60  (asymmetric distance — under-triage penalised harder)
critical_sign ..... 0.25
recommended_action  0.15
```
Full parity score (consistency across all 4 demographic variants) available via `POST /grade-fairness`.

**Task 5 — Deteriorating Patient (per-step, multi-turn):**
```
T=0  step reward: 0.30 for correct initial disposition
T=30 step reward: 1.00 for correct escalation (CRITICAL MOMENT)
T=60 step reward: 0.40–0.60 for late catch (only reached if T=30 missed)
Signal bonus: up to +0.10 for naming key deterioration signals in rationale
```

**Asymmetric Priority Distance:**  
Under-triage is clinically dangerous; over-triage is wasteful but safe.
```
critical predicted as low   → 0.00  (patient may die)
high predicted as low       → 0.10  (dangerous)
critical predicted as high  → 0.45  (close but risky)
low predicted as critical   → 0.25  (wasteful but safe)
```

**Confidence Calibration Bonus** (up to +0.05):  
Rewards agents that are confident when correct on easy cases, and appropriately uncertain when wrong on hard cases. Overconfidence on wrong answers is penalised.

---

## 5. API Endpoints Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — `{"status":"healthy","version":"2.0.0"}` |
| `POST` | `/reset` | Start episode. Body: `{task_id?, case_index?, seed?, session_id?}`. Returns `info.session_id`. |
| `POST` | `/step` | Submit assessment. Body: `{action: TriageAction, session_id?}`. Returns reward, done, breakdown. |
| `GET` | `/state` | Episode metadata. Query: `?session_id=`. |
| `POST` | `/grade-fairness` | Multi-variant parity score. Body: `{group_id, responses: {case_id: action}}`. |
| `GET` | `/tasks` | All task IDs with case counts and case IDs. |
| `GET` | `/metrics` | Score distributions, difficulty gradient, case coverage. |
| `GET` | `/history` | Chronological episode log. Query: `?limit=100`. |
| `GET` | `/stats` | Per-task avg/best/count. |
| `GET` | `/web` | Interactive web UI. |
| `GET` | `/docs` | Swagger/OpenAPI UI. |
| `GET` | `/redoc` | ReDoc API documentation. |
| `GET` | `/openapi.json` | Raw OpenAPI schema. |

---

## 6. Configuration

### 6.1 Required Environment Variables

Used by both `inference.py` and `train.py`:

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | `hf_xxxx...` |

Optional:

| Variable | Description | Default |
|---|---|---|
| `SERVER_URL` | Environment API URL | `http://localhost:8000` |
| `REPS_PER_TASK` | Reps per task in `train.py` | `3` |
| `TRAIN_EPISODES` | Max episodes in `train.py` | `12` |

### 6.2 HF Space Secret/Variable Setup

Configure Space runtime vars via the `setupCredentials.py` helper:

```bash
export HF_TOKEN="<admin-token>"
export SPACE_REPO_ID="kunalkachru23/medical-triage-env"
export INFERENCE_HF_TOKEN="<inference-token>"
python setupCredentials.py
```

Or set manually in HF Space Settings → Variables and Secrets.

---

## 7. Setup Instructions

### 7.1 Local Python Setup

```bash
git clone https://github.com/kunalkachru/scaler-hackathon-rl-medical-triage-env.git
cd medical-triage-env

python3 -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows

pip install -r server/requirements.txt

uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Access:
- Web UI: http://localhost:8000/web
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 7.2 Docker Setup

```bash
docker build -t medical-triage-env:latest .
docker run -p 7860:7860 medical-triage-env:latest
```

Verify:
```bash
curl http://localhost:7860/health
# → {"status":"healthy","service":"medical-triage-env","version":"2.0.0"}
```

### 7.3 Deployed HF Space

The environment is live at:
- Space: https://huggingface.co/spaces/kunalkachru23/medical-triage-env
- API: https://kunalkachru23-medical-triage-env.hf.space

```bash
curl https://kunalkachru23-medical-triage-env.hf.space/health
curl -X POST https://kunalkachru23-medical-triage-env.hf.space/reset \
  -H "content-type: application/json" \
  -d '{"task_id":"simple_triage","case_index":0,"seed":42}'
```

---

## 8. Running the Scripts

### 8.1 Baseline Inference (`inference.py`)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="<your-token>"

# Local server (auto-started by script)
python inference.py

# Against live HF Space
SERVER_URL="https://kunalkachru23-medical-triage-env.hf.space" python inference.py
```

Runs 2 cases per task (10 total) and emits strict `[START]`, `[STEP]`, `[END]` lines to match organizer parsing rules.

### 8.2 RL Training Loop (`train.py`)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="<your-token>"

python train.py
```

Demonstrates the environment supports training, not just evaluation:
- Repeats the **same case** per task across multiple episodes (isolates learning signal)
- Injects past performance into the system prompt (reward-conditioned prompting)
- Prints per-task score progression curves
- Confirmed: 3/3 tasks improve (conflicting_vitals: 0.265 → 0.467, masked_deterioration: 0.675 → 0.900)

---

## 9. Testing and Validation

### 9.1 Test Suite (current local gate)

```bash
# Full suite
pytest tests/ -q
# Latest local gate: 345 collected, 331 passed, 14 skipped

# By module
pytest tests/test_graders.py -v
pytest tests/test_environment.py -v
pytest tests/test_v2_enhancements.py -v
pytest tests/test_api_contract.py -v
pytest tests/test_ui_contract.py -v
pytest tests/test_inference_contract.py -v
pytest tests/test_app_coverage.py -v
pytest tests/test_client_scripts.py -v
```

**Test coverage:**

| Suite | Scope | What it covers |
|---|---|---|
| `test_graders.py` | unit | NEWS2 boundaries, priority distance, task graders |
| `test_environment.py` | integration | reset/step/state contracts, episode flows, independence |
| `test_v2_enhancements.py` | regression | fairness, deterioration, confidence, synonyms |
| `test_api_contract.py` | API contract | session handling, fairness endpoint, metrics structure |
| `test_ui_contract.py` | UI contract | web UI hooks, task-switch/history guards |
| `test_inference_contract.py` | script contract | inference reproducibility guards |
| `test_app_coverage.py` | endpoint coverage | broad app endpoint and branch coverage |
| `test_client_scripts.py` | script/client | static and live-marked script contract checks |

### 9.2 Pre-Submission Validation Gate

```bash
./scripts/pre_submit_check.sh
```

Pipeline (5 steps):
1. Full test suite must pass (`pytest tests/ -q`)
2. Docker build
3. Container health check on port 7860
4. Reset endpoint smoke check
5. `openenv validate` (static — checks yaml and project structure)

### 9.3 Adversarial Robustness

The `/step` endpoint is hardened against unexpected agent inputs (relevant for Phase 2 agentic evaluation with Nemotron 3 Super or similar):

| Input | Behaviour |
|---|---|
| Empty action `{}` | 200, empty-response path; **HTTP** `reward`/`score` ∈ `(0,1)` at floor (~`1e-4`), not literal `0.0` |
| `priority: null` | 200, coerced to `""`; scored like missing priority (typically low **HTTP** reward, still in `(0,1)` if episode ends) |
| Unknown fields | 200, unknown fields ignored; reward depends on remaining fields (still **HTTP** `(0,1)` when graded) |
| Very long rationale (8000+ chars) | 200, scored normally |
| Unknown priority value | 200, low score; **HTTP** `(0,1)` |
| No `session_id` | 200, uses default session (backward-compat) |
| Nonexistent `session_id` | 400 (not 500) |
| Double step on single-step task | 200 (not 500) |

---

## 10. UI Testing Guide

### 10.1 Accessing the UI

| Environment | URL |
|---|---|
| Live HF Space | https://kunalkachru23-medical-triage-env.hf.space/web |
| Local | http://localhost:8000/web |
| Docker | http://localhost:7860/web |

### 10.2 Core UI Flow (all tasks)

1. Select task from dropdown
2. Click **New Patient Case** → patient narrative appears
3. Fill in assessment fields (or use **Auto-fill with AI**)
4. Click **Submit & Score**
5. Score card appears: reward value, dimension breakdown, feedback text
6. Episode log updates with the result

### 10.3 Task-Specific Manual Tests

**`simple_triage`:**
- Enter correct `priority`, `news2_score`, `critical_sign`, `recommended_action`
- Expect: reward ≈ 0.75–1.0 for clinically correct answers
- Verify: score_breakdown shows 4 dimensions (priority, news2_score, critical_sign, recommended_action)

**`conflicting_vitals`:**
- Identify the truly dangerous sign (not the misleading normals)
- Submit with misleading_signs list and rationale
- Re-run choosing the trap sign as critical → verify score drops on critical_sign dimension

**`masked_deterioration`:**
- Provide `masking_drug_or_condition`, `masked_sign`, and `critical_clues`
- Expect: score_breakdown shows masking_mechanism and critical_clues dimensions
- A low NEWS2 score should not fool the agent — the environment rewards overriding it

**`demographic_fairness`:**
- Run FP001 (white male), then FP001 (black male) — identical vitals
- Expect: same priority and critical_sign for both
- Use `POST /grade-fairness` with both responses to get parity score

**`deteriorating_patient` (multi-turn):**
- Step 1 (T=0): choose monitor/escalate/emergency_response → internal `0.3` if correct (**HTTP** mapped into `(0,1)`)
- Step 2 (T=30): **critical moment** — escalate if vitals deteriorating → internal up to `1.0` (**HTTP** just below `1.0`)
- If T=30 missed: Step 3 (T=60) available with partial credit (internal 0.4–0.6, **HTTP** mapped)
- Episode ends (done=True) when correct action taken or all 3 steps exhausted

**Training Progress tab:**
- Switch to Training tab in UI
- Verify episode history, per-task averages, and best scores update after each submission

### 10.4 AI Auto-fill

Click **Auto-fill with AI** in the UI to request a model-generated suggestion. If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are configured, the app uses the live LLM path; otherwise it falls back to a deterministic rule-based suggestion. The suggestion is pre-populated in the form fields but not submitted — you must click **Submit & Score** to record it.

---

## 11. Deployment

### 11.1 Deploy to HF Space

```bash
cd medical-triage-env
openenv push
```

This validates the project, creates/updates the Space at `kunalkachru23/medical-triage-env`, uploads all files, and confirms deployment. The Space uses `sdk: docker` and `app_port: 7860` as declared in `openenv.yaml`.

### 11.2 Verify Deployed Space

```bash
# Health
curl https://kunalkachru23-medical-triage-env.hf.space/health

# Reset
curl -X POST https://kunalkachru23-medical-triage-env.hf.space/reset \
  -H "content-type: application/json" \
  -d '{"task_id":"simple_triage","case_index":0,"seed":42}'

# openenv validate (static)
openenv validate .
# → [OK] medical_triage: Ready for multi-mode deployment

# full live deployment API pack
./scripts/live_verify.sh

# browser-level smoke validation (headless)
python -m pip install playwright
python -m playwright install chromium
python scripts/browser_ui_smoke.py --base-url "https://<your-space>.hf.space"

# one-command release gate (local + baseline + live checks)
chmod +x ./scripts/full_release_gate.sh
./scripts/full_release_gate.sh \
  --base-url "https://<your-space>.hf.space" \
  --repo-id "<your-username>/<your-space-name>" \
  --expect-llm true
```

Release gate flags:

```text
--base-url <url>               Live target base URL (required for live verification stages)
--repo-id <user/space>         HF repo id used by openenv push (required unless --skip-deploy)
--expect-llm true|false        true = strict LLM-path validation; false = allow fallback mode
--skip-deploy                  Skip deployment step; run validation only
--skip-playwright-install      Skip Playwright install step (script will still fail with guidance if missing)
-h, --help                     Show script usage and options

Baseline env vars (required by automated release gate):
API_BASE_URL, MODEL_NAME, HF_TOKEN (or OPENAI_API_KEY/API_KEY)
```

Recommended evaluator invocations:

```bash
# Full run: local checks + deploy + live API + browser smoke
./scripts/full_release_gate.sh \
  --base-url "https://<your-space>.hf.space" \
  --repo-id "<user>/<space>" \
  --expect-llm true

# Verification-only run after deployment is already complete
./scripts/full_release_gate.sh \
  --skip-deploy \
  --base-url "https://<your-space>.hf.space" \
  --expect-llm true
```

### 11.3 Post-Deploy Configuration

Set Space runtime variables via HF Settings or `setupCredentials.py`:
- `API_BASE_URL` (variable) — LLM endpoint
- `MODEL_NAME` (variable) — model identifier
- `HF_TOKEN` (secret) — Hugging Face API key

If endpoints still run in fallback/mock mode after deploy, apply this one-time refresh:

```bash
# 1) Update local .env
# HF_TOKEN=<admin-token-for-space-settings>
# SPACE_REPO_ID=<your-username>/<your-space-name>
# INFERENCE_HF_TOKEN=<token-used-by-runtime-inference>

# 2) Push variables/secrets to the Space
python setupCredentials.py

# 3) Restart the HF Space once
```

Verify runtime is using the real model path:

```bash
curl -X POST "https://<your-space-subdomain>.hf.space/suggest" \
  -H "content-type: application/json" \
  -d '{"patient_history":"RR=24 SpO2=93 BP=105/70 HR=112 Temp=38.4 Consciousness=Alert","task_id":"simple_triage"}'
```

Expected: response includes `"llm_used": true`.

---

## 12. Multi-Model Benchmark

Empirical scores from `inference.py` (2 cases/task, seed=42, HF Router):

| Task | Llama-3.1-8B | Llama-3.3-70B | Interpretation |
|---|---|---|---|
| `simple_triage` | 0.857 | 0.883 | Both do well — easy task, clear protocol |
| `conflicting_vitals` | 0.270 | 0.281 | Both struggle — resisting misleading normals is hard |
| `masked_deterioration` | 0.475 | 0.588 | Hard — pharmacological masking requires clinical knowledge |
| `demographic_fairness` | 0.810 | 0.810 | Identical — fairness bias is scale-invariant |
| `deteriorating_patient` | 0.750 | 0.750 | Hard — escalation timing requires trajectory reasoning |

The difficulty gradient is real and empirically confirmed: a 38–45% drop from easy to hard tasks. Even the 70B model scores only 0.588 on `masked_deterioration` — the environment genuinely challenges frontier models.

---

## 13. Known Constraints

- Exact baseline scores may vary by model/provider/routing despite deterministic grader logic
- For reproducible demonstrations, always pass `seed=42` and `case_index=<fixed>`
- The confidence calibration bonus is up to +0.05 (`grade_confidence_calibration`); internal total is capped at `1.0`, then remapped to `(0,1)` for HTTP
- Session TTL is 30 minutes; long-running evaluation loops should use explicit `session_id` management

---

## 14. Evaluator Quickstart (10 Minutes)

**Step 1 — Validate project structure:**
```bash
openenv validate .
```

**Step 2 — Run tests:**
```bash
pytest tests/ -q
# Expected: 119 passed
```

**Step 3 — Test the live Space:**
```bash
curl https://kunalkachru23-medical-triage-env.hf.space/health
curl -X POST https://kunalkachru23-medical-triage-env.hf.space/reset \
  -H "content-type: application/json" -d '{"task_id":"simple_triage","case_index":0}'
```

**Step 4 — Run baseline inference:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="<token>"
python inference.py
```

**Step 5 — Run training loop:**
```bash
python train.py
```

**Step 6 — Open the web UI:**
- Live: https://kunalkachru23-medical-triage-env.hf.space/web
- Try `simple_triage` (easy) and `deteriorating_patient` (hard multi-turn)

**Step 7 — Explore the API docs:**
- https://kunalkachru23-medical-triage-env.hf.space/docs
