# Medical Triage RL Environment: Project Documentation

## 1) Real-World Overview and Objective

In a real emergency department, a triage nurse has to decide urgency quickly from incomplete or noisy information. This project simulates that exact decision process as an OpenEnv-compatible reinforcement learning environment. The agent reads a realistic patient presentation (history, vitals, medications, trends), then submits a triage decision that deterministic clinical graders evaluate.

What we are trying to achieve:
- Build a clinically meaningful RL benchmark, not a generic text game.
- Capture both straightforward and failure-prone triage scenarios.
- Measure decisions with transparent, reproducible reward functions.
- Enable safe model comparison, regression testing, and demo-ready deployment.

Real-world examples this environment covers:
- **Clear deterioration (easy):** patient with hypotension, tachycardia, and low oxygen saturation should be escalated quickly. The expected behavior is direct urgent/critical triage.
- **Conflicting signs (medium):** some vitals look normal, but one hidden red flag indicates danger. The agent must avoid being misled by "normal-looking" values.
- **Masked risk (hard):** medications (for example beta-blockers or steroids) can suppress classic warning signs; the model must infer severe risk from indirect clues.
- **Bias-sensitive triage:** clinically equivalent patients with different demographic descriptors should receive the same urgency decision.
- **Progressive deterioration (multi-turn):** a patient worsens over time; escalation at the first critical trend is rewarded more than late escalation.

Why this matters:
- Under-triage can delay life-saving care.
- Over-triage consumes limited emergency resources.
- Clinical AI evaluation requires deterministic graders and traceable score breakdowns.
- RL training needs dense rewards, not binary pass/fail labels.

Primary objective:
- Deliver a production-style OpenEnv environment that evaluators can run, inspect, deploy, and verify end-to-end.

## 2) Solution Scope

The environment includes five tasks:
- `simple_triage` (easy)
- `conflicting_vitals` (medium)
- `masked_deterioration` (hard)
- `demographic_fairness` (medium)
- `deteriorating_patient` (hard, multi-turn)

Total case bank: 28 cases.

Core interface:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

Reference manifest:
- `openenv.yaml`

## 3) Technical Design

### 3.1 Project Structure

- `server/app.py`: FastAPI service, web UI endpoint, helper endpoints.
- `server/medical_triage_environment.py`: environment state machine (`reset`, `step`, `state`).
- `server/graders.py`: deterministic grading and reward logic.
- `server/cases.py`: case bank and task data.
- `models.py`: typed Pydantic models for Action, Observation, State, and request/response wrappers.
- `inference.py`: baseline evaluator script using OpenAI client with env vars.
- `client.py`: Python client wrapper for programmatic use.
- `scripts/pre_submit_check.sh`: one-command validation pipeline.
- `tests/`: unit, integration, API contract, and UI contract tests.

### 3.2 Runtime Architecture

1. Client calls `POST /reset` with optional `task_id`, `case_index`, `seed`.
2. Environment samples/selects case and returns initial observation.
3. Client calls `POST /step` with `TriageAction`.
4. Task-specific deterministic grader computes reward and breakdown.
5. Environment returns `observation`, `reward`, `done`, `info`.
6. `GET /state` returns episode metadata.

### 3.3 State and Episode Model

- Single-step episodes:
  - `simple_triage`, `conflicting_vitals`, `masked_deterioration`, `demographic_fairness`.
- Multi-step episode:
  - `deteriorating_patient` with timeline-based progression and escalation decisions.

State tracks:
- episode ID
- current task/case
- step count
- cumulative reward
- per-task best scores
- done status

## 4) Functional Design

### 4.1 Action and Observation

Action supports:
- core triage fields (`priority`, `news2_score`, `critical_sign`, `recommended_action`)
- reasoning fields (`rationale`, `confidence`)
- task-specific fields (`misleading_signs`, `masking_drug_or_condition`, `masked_sign`, `critical_clues`, `action` for multi-turn)

Observation includes:
- patient history
- task metadata
- score and score breakdown
- done flag
- step number
- optional hint

### 4.2 Reward and Grading Design

Design principles:
- deterministic scoring
- partial credit
- bounded step reward in `[0.0, 1.0]`
- contextual penalties for unsafe behavior

Highlights:
- NEWS2-grounded triage scoring.
- Asymmetric under/over-triage penalty.
- Confidence calibration bonus.
- Fairness scoring support (single-variant in online step flow; full parity function available).
- Multi-turn deterioration grading by timeline step.

## 5) Configuration

### 5.1 Inference Runtime Variables

Used by `inference.py`:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `OPENAI_API_KEY`/`API_KEY` fallback)
- optional `SERVER_URL` (defaults to `http://localhost:8000`)

### 5.2 Space Secret/Variable Setup

Use `setupCredentials.py` to configure Space runtime settings:
- `API_BASE_URL` (Space variable)
- `MODEL_NAME` (Space variable)
- `HF_TOKEN` (Space secret, sourced from local `INFERENCE_HF_TOKEN`)

Local input vars required by `setupCredentials.py`:
- `HF_TOKEN` (admin token for space configuration)
- `SPACE_REPO_ID`
- `INFERENCE_HF_TOKEN`

Behavior:
- Auto-loads `.env` if present.
- Gracefully fails if `.env` is missing or has placeholder values.

## 6) Setup Instructions

### 6.1 Local Python Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open UI:

`http://localhost:8000/web`

### 6.2 Docker Setup

```bash
docker build -t medical-triage-env:latest .
docker run -p 7860:7860 medical-triage-env:latest
```

Containerized health endpoint:

`http://localhost:7860/health`

## 7) Deployment Strategy

### 7.1 OpenEnv Deployment to Hugging Face Spaces

From project root:

```bash
openenv validate
openenv push --repo-id <hf_username>/<space_name>
```

Notes:
- If `--repo-id` is omitted, OpenEnv derives a default target from environment metadata and logged-in user.
- For maintaining multiple versions, use distinct Space names (for example, `...-v2`).

### 7.2 Post-Deploy Runtime Config

Configure Space runtime vars/secrets via:
- HF Space Settings UI, or
- `setupCredentials.py`.

## 8) Testing and Validation

### 8.1 Automated Test Suite

Run all tests:

```bash
pytest tests/ -q
```

Current status: 99 passing tests.

Test coverage includes:
- environment behavior
- grader correctness
- v2 enhancements
- API contracts
- UI contracts

### 8.2 Pre-Submission Gate

```bash
./scripts/pre_submit_check.sh
```

Pipeline includes:
1. test suite
2. Docker build
3. container health check
4. reset endpoint smoke check
5. `openenv validate`

### 8.3 Live Endpoint Verification Checklist

After deployment, verify:
- `GET /health` returns 200 and healthy payload
- `POST /reset` returns initial observation
- `POST /step` returns valid reward/done
- `GET /state` returns structured metadata
- `GET /web` renders UI

### 8.4 UI Test Plan (Step-by-Step)

Use this section to test manually through the browser UI.

Pre-check:
1. Start local app (`http://localhost:8000/web`) or open deployed Space URL.
2. Confirm page loads with task selector, case selector, and "New Patient Case".
3. Click "New Patient Case" once to initialize episode state.

Core UI flow to validate on every task:
1. Select task.
2. Click "New Patient Case".
3. Verify patient narrative appears.
4. Verify the NEWS2 calculator auto-populates (where applicable).
5. Enter/auto-fill response fields.
6. Click "Submit & Score".
7. Verify score card shows reward, breakdown, and feedback.
8. Verify episode log updates.

Task-specific manual tests:

`simple_triage`:
1. Pick any case.
2. Enter `priority`, `news2_score`, `critical_sign`, `recommended_action`.
3. Submit and verify reward is non-zero for clinically coherent answers.
4. Verify score breakdown includes priority/news2/critical sign/action dimensions.

`conflicting_vitals`:
1. Pick a case where some signs look normal.
2. Enter a response that identifies the dangerous sign and misleading signs.
3. Submit and verify rationale contributes to score.
4. Re-run with a trap answer (choose a misleading sign as critical) and verify the score drops.

`masked_deterioration`:
1. Pick a masked case.
2. Provide `masking_drug_or_condition`, `masked_sign`, and `critical_clues`.
3. Submit and verify breakdown includes masking mechanism and critical clues.
4. Verify low-quality answers trigger lower score and optional hints.

`demographic_fairness`:
1. Select fairness task and run multiple demographic variants.
2. Keep triage logic purely clinical (do not use demographics).
3. Verify consistent priorities and actions across variants.
4. Use Training tab (`/history` and `/stats`) to compare variant scores quickly.

`deteriorating_patient` (multi-turn):
1. Start a deterioration case.
2. At first step, choose action (`monitor`/`escalate`/`emergency_response`) and submit.
3. Confirm updated patient history appears for next time step if episode is not done.
4. Continue until done; verify reward behavior changes by timing of escalation.
5. Confirm episode dots and logs reflect progression.

AI assist / autofill checks:
1. Click "Auto-fill with AI".
2. Verify fields are populated and the status indicates LLM/rule-based source.
3. Submit autofilled response and confirm normal scoring pipeline still works.

Training tab checks:
1. Switch to "Training Progress".
2. Verify total episodes, average score, and best score update.
3. Verify charts render and episode table reflects recent submissions.

## 9) UI Design and Usability

The web interface is designed for both evaluators and non-technical demos:
- task selection and case selection
- guided step-based triage workflow
- live NEWS2 calculator
- AI autofill helper
- score card with breakdown, feedback, and hints
- training progress charts (`/history` + `/stats`)

UI contract checks ensure key controls and hooks remain intact.

## 10) Evaluation-Oriented Notes

How this maps to evaluation goals:
- Real-world utility: hospital triage decision simulation.
- Task/grader quality: deterministic, interpretable, varied difficulty.
- Environment design: typed contracts, clean boundaries, state tracking.
- Spec compliance: validated via OpenEnv validator and deployment checks.
- Creativity/novelty: fairness and multi-turn deterioration scenarios.

## 11) Known Constraints and Practical Guidance

- Exact baseline scores may vary by model/provider behavior despite deterministic environment grading.
- Space runtime LLM behavior depends on configured `API_BASE_URL`, `MODEL_NAME`, and token permissions.
- For reproducible demonstrations, use fixed `seed` and `case_index`.

## 12) Quickstart for Evaluators (10-Minute Path)

1. `openenv validate`
2. `./scripts/pre_submit_check.sh`
3. Run baseline:
   - set `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
   - `python inference.py`
4. Open UI:
   - local: `http://localhost:8000/web`
   - space: `https://<space-subdomain>.hf.space/web`
5. Validate one easy and one hard flow:
   - `simple_triage`
   - `deteriorating_patient`

