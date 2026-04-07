# Medical Triage Environment: Low-Level Design

## 1) Purpose

This document maps the codebase at file level and symbol level (classes, methods, interfaces) to explain how each part contributes to the full OpenEnv medical triage system.

Use this as:
- a developer onboarding guide
- an evaluator implementation map
- a debugging/reference document for API, grading, and UI behavior

Note: this is an engineering depth document. It is optional for hackathon evaluation and can be kept local if you prefer a lean submission.

## 2) End-to-End Runtime Flow

1. Client calls `POST /reset` on FastAPI service.
2. `server.app` delegates to `MedicalTriageEnvironment.reset()`.
3. Environment selects case from `server.cases` and returns `StepResult`.
4. Client submits `POST /step` with `TriageAction`.
5. `MedicalTriageEnvironment.step()` routes to:
   - single-step grading via `grade_response()` and optional confidence bonus, or
   - multi-turn grading via `_step_deteriorating()` and `grade_deteriorating_patient_step()`.
6. Response returns typed `StepResult` (`observation`, `reward`, `done`, `info`).
7. UI and analytics endpoints (`/history`, `/stats`) consume recorded episode outcomes.

## 3) Interface Contracts

### 3.1 HTTP Interface (`server/app.py`)

Required OpenEnv endpoints:
- `POST /reset` -> start episode
- `POST /step` -> score action
- `GET /state` -> environment state
- `GET /health` -> deployment health

Additional product/demo endpoints:
- `GET /tasks` -> task catalog
- `GET /history` -> completed episode trail
- `GET /stats` -> aggregate metrics
- `POST /suggest` -> AI/rule-based form suggestions
- `POST /agent-assess` -> LLM-driven assessment helper
- `GET /web`, `GET /` -> interactive browser UI

### 3.2 Typed Model Interface (`models.py`)

Primary data contracts:
- `TriageAction`
- `TriageObservation`
- `TriageState`
- `ResetRequest`
- `StepRequest`
- `StepResult`

All API IO and client calls are typed through these models.

## 4) File-by-File Functional Design

### 4.1 `server/app.py` (HTTP service + UI host)

Role:
- Exposes API contract
- Hosts embedded web UI
- Integrates environment runtime and episode history analytics

Key classes:
- `EpisodeHistory`
  - `record(task_id, case_id, reward, breakdown)`: appends completed episode.
  - `as_list()`: returns episode timeline.
  - `stats()`: computes aggregate metrics by task and overall.
- `SuggestRequest`: request schema for `/suggest`.
- `AgentAssessRequest`: request schema for `/agent-assess`.

Key endpoint handlers:
- `health()`: returns healthy payload for validators/health checks.
- `reset(request: ResetRequest)`: starts episode via environment.
- `step(request: StepRequest)`: scores action, records done episodes.
- `get_history(limit)`: episode timeline for training chart.
- `get_stats()`: aggregate stats for performance dashboard.
- `state()`: raw typed environment state.
- `list_tasks()`: case IDs and counts from case bank.
- `suggest_action(request)`: LLM-first suggestion with deterministic fallback.
- `web_interface()`: serves UI HTML.
- `agent_assess(request)`: LLM assessment helper for UI.
- `main()`: CLI entrypoint used by packaging/OpenEnv validator.

Internal helpers:
- `_rule_based_suggest(patient_history, task_id)`: deterministic parser/scorer for autofill fallback.
- `_mock_agent_response(task_id)`: fallback action when LLM unavailable.

### 4.2 `server/medical_triage_environment.py` (core environment state machine)

Role:
- Implements environment lifecycle (`reset`, `step`, `state`)
- Couples selected case + grading + state transitions

Key class:
- `MedicalTriageEnvironment`
  - `__init__()`: initializes state, RNG, active case pointers.
  - `reset(request)`: task/case selection, seed control, initial observation creation.
  - `step(action)`: validates lifecycle, grades response, updates cumulative state.
  - `_step_deteriorating(action_dict)`: multi-turn timeline logic for deterioration task.
  - `state` property: exposes typed `TriageState`.
  - `_get_hint(task_id, score)`: remediation hints for low scores.

Important behavior:
- Handles both single-step and multi-step episodes.
- Adds confidence calibration bonus where applicable.
- Returns `ground_truth` in terminal step info for transparency.

### 4.3 `server/graders.py` (deterministic scoring engine)

Role:
- Implements all grading functions and reward shaping
- Guarantees bounded deterministic outputs for evaluator reproducibility

Core scoring primitives:
- `compute_news2(vitals)`
- `news2_to_priority(news2_score, individual_scores)`
- `priority_distance(predicted, expected)`
- `asymmetric_priority_distance(predicted, expected)`

Task graders:
- `grade_simple_triage(agent_response, case)`
- `grade_conflicting_vitals(agent_response, case)`
- `grade_masked_deterioration(agent_response, case)`
- `grade_single_fairness_variant(agent_response, case)`
- `grade_demographic_fairness(responses, cases)` (multi-variant parity mode)
- `grade_deteriorating_patient_step(agent_response, timeline_entry, step_index, case)`
- `grade_confidence_calibration(confidence, news2_score, score_was_correct)`

Routing and helpers:
- `grade_response(task_id, agent_response, case)` dispatches to task-specific grader.
- `_canonicalize_action(action)` normalizes action synonyms for timeline tasks.
- `_extract_key_terms(text)` builds rationale keyword set for scoring.

### 4.4 `server/cases.py` (case dataset and task indexing)

Role:
- Stores all scenario definitions and ground-truth labels
- Provides lookups by task and case ID

Main dataset groups:
- `SIMPLE_TRIAGE_CASES`
- `CONFLICTING_VITALS_CASES`
- `MASKED_DETERIORATION_CASES`
- fairness variants and grouped fairness sets
- `DETERIORATION_CASES`

Important symbols/functions:
- `CASE_BANK`: canonical task-to-case mapping.
- `ALL_TASKS`: list of task IDs.
- `_make_fairness_variants(...)`: generator utility for demographic variant sets.
- `get_cases_for_task(task_id)`: returns cases for one task.
- `get_case_by_id(case_id)`: reverse lookup utility.

### 4.5 `models.py` (typed domain models)

Role:
- Defines canonical request/response/state interface contracts
- Drives FastAPI validation and OpenAPI generation

Classes:
- `TriageAction`: full action schema (triage + fairness + masked + multi-turn + confidence). `priority` field is `Optional[str]` with a `field_validator` that coerces explicit JSON `null` → `""` (null-safety fix — prevents 422 errors from agents sending `"priority": null`).
- `TriageObservation`: environment observation + score feedback.
- `TriageState`: runtime metadata for active episode/session.
- `ResetRequest`: reset API input.
- `StepRequest`: step API input wrapper (`action`).
- `StepInfo`: typed metadata alongside every step/reset response (session_id, episode_id, max_steps, step_time, agent_action). Uses `extra="allow"` for forward compatibility.
- `StepResult`: API output envelope.

Functional value:
- Prevents malformed payloads from entering runtime logic.
- Ensures UI/client/server all speak same schema.
- Null-safe priority field prevents 422 errors from agents that explicitly send null.

### 4.6 `client.py` (Python integration client)

Role:
- Lightweight programmatic wrapper over HTTP endpoints for training/evaluation scripts.

Class:
- `MedicalTriageEnv`
  - `reset(task_id=None, case_index=None, seed=None)`
  - `step(action: TriageAction)`
  - `state()`
  - `health()`
  - `list_tasks()`
  - `close()`
  - context manager methods (`__enter__`, `__exit__`)

Functional value:
- Simplifies integration into RL loops and notebooks.

### 4.7 `inference.py` (baseline evaluator executable)

Role:
- Mandatory baseline runner used for reproducible model scoring runs.

Key functions:
- `extract_json(text)`: robust JSON extraction from model output.
- `call_llm(client, patient_history, task_description, task_id)`: model inference step.
- `run_episode(client, task_id, case_index, server_url)`: single or multi-turn episode loop.
- `wait_for_server(url, retries)`: health polling before run.
- `_validate_required_env()`: startup env validation.
- `main()`: orchestrates full benchmark run and summary.

Functional value:
- Demonstrates reference interaction pattern with environment.
- Produces task-wise benchmark signal for evaluators.

### 4.8 `train.py` (reward-conditioned RL training loop)

Role:
- Demonstrates that the environment supports training, not just evaluation.
- Implements reward-conditioned prompting: injects past episode rewards into the LLM system prompt.

Key functions:
- `build_system_prompt(task_id, task_history)`: constructs system prompt with last 3 attempts, scores, breakdowns, and task-specific improvement advice.
- `call_llm(client, system_prompt, patient_history, task_description, task_id)`: model inference with reward-conditioned context.
- `run_episode(client, task_id, case_index, task_history, rep, server_url)`: runs one reset→step episode and returns result dict.
- `print_summary(all_results)`: prints per-task learning curve with trend arrows.
- `main()`: orchestrates 3 tasks × N repetitions with same fixed case per task.

Design decisions:
- Uses the same fixed `case_index` per task across all repetitions — isolates learning signal from case-difficulty variation.
- Task-filtered feedback: agent only sees past performance on the same task type, preventing cross-task confusion.
- `IMPROVEMENT_ADVICE` dict provides grader-aligned hints per task type for targeted guidance.

Env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` (same as `inference.py`). Optional: `SERVER_URL`, `REPS_PER_TASK`.

Functional value:
- Proves dense reward signal enables measurable policy improvement across repetitions.
- Confirms environment suitability for policy gradient and reward-conditioned training.

### 4.9 `setupCredentials.py` (HF runtime configuration utility)

Role:
- Configures Hugging Face Space variables/secrets for hosted inference.

Key functions:
- `load_dotenv_if_present()`: non-destructive `.env` loading.
- `require_env(name, dotenv_found, env_path)`: strict validation with placeholder checks.
- `main()`: calls `huggingface_hub.HfApi` to set variables/secrets.

Functional value:
- Standardizes deployment secret setup and reduces manual errors.

### 4.10 `openenv.yaml` (OpenEnv manifest)

Role:
- Environment metadata and spec-facing configuration.

Defines:
- app port
- exposed endpoints
- task inventory and case counts
- reward range and max episode length

Functional value:
- Used by OpenEnv validator and deployment workflows.

### 4.11 `Dockerfile` (container runtime definition)

Role:
- Packages app into reproducible deployment image.

Functional content:
- installs runtime dependencies
- copies source modules
- exposes `7860`
- defines healthcheck against `/health`
- starts uvicorn service

### 4.12 Validation and test harness files

`scripts/pre_submit_check.sh`:
- runs full pytest (111 tests)
- builds Docker image
- checks container health
- smoke-tests `/reset`
- runs `openenv validate`

`tests/test_graders.py` (30 tests):
- unit-level grading correctness and NEWS2 boundary tests.
- covers all scoring boundary values per NHS RCP 2017 spec.

`tests/test_environment.py` (24 tests):
- integration-level environment lifecycle and state tests.
- reset, step, state, full episode flows.

`tests/test_v2_enhancements.py` (40 tests):
- fairness parity, asymmetric priority distance, confidence calibration bonus, and multi-turn deterioration tests.
- includes 16 adversarial robustness tests (null fields, wrong types, empty body, unknown fields).

`tests/test_api_contract.py` (9 tests):
- API parsing/contract checks for newer action fields.
- requires live server on localhost:8000.

`tests/test_ui_contract.py` (8 tests):
- UI DOM contract, session wiring, and UI regression guards for interactive hooks.
- requires live server on localhost:8000.

**Total: 111 tests — all passing.**

## 5) Cross-File Dependency Map

Primary dependencies:
- `server/app.py` -> `server/medical_triage_environment.py`, `models.py`, `server/cases.py`
- `server/medical_triage_environment.py` -> `server/cases.py`, `server/graders.py`, `models.py`
- `server/graders.py` -> pure logic + case ground truth contract
- `client.py` -> HTTP endpoints + `models.py`
- `inference.py` -> HTTP endpoints + OpenAI client
- `train.py` -> HTTP endpoints + OpenAI client (reward-conditioned loop)
- tests -> app/environment/graders/models interfaces

Data flow contracts:
- `server/cases.py` ground truth schema must match grader expectations.
- `models.py` action fields must stay aligned with UI payload and tests.
- endpoint schemas in `server/app.py` must remain compatible with `client.py` and `inference.py`.

## 6) Extension Guidelines (LLD Perspective)

When adding a new task:
1. Add case set in `server/cases.py`.
2. Implement grader function in `server/graders.py`.
3. Register task in grader dispatch and environment task descriptions.
4. Extend `TriageAction` / observation fields if needed.
5. Expose in UI task selector and form generation logic.
6. Add tests:
   - grader unit tests
   - environment flow tests
   - API/UI contract tests if schema/UI changed.

When changing scoring:
1. Update grader function and breakdown keys.
2. Keep reward bounds in `[0.0, 1.0]`.
3. Update tests and docs to maintain evaluator traceability.

