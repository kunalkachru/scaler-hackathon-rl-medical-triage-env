# TODO: Round 1 enhancement roadmap (merged plan)

**Status:** v2.3.0 — latest local gate: 348 collected, 334 passed, 14 skipped. Staging validation and full release gate are green.

This document merges:

- The hackathon rubric–aligned plan (see also `.cursor/plans/hackathon_win_enhancements_e9354873.plan.md`)
- The clinical / case-bank / API expansion plan (Team Falcons audit)
- Original long-horizon ideas (bottom)

**Git / deploy policy (owner-controlled):** Do not `git push` or deploy to the **judges’ canonical Space** unless you explicitly choose to. AI assistants should not commit or deploy without your approval.

---

## 0. Deployment: staging vs production (read first)

**Goal:** Until Phase 1 results are clear (and while you iterate), **do not overwrite** the deployment judges already evaluated.

| Environment | Suggested HF Space repo id | Purpose |
|-------------|----------------------------|---------|
| **Production (judges)** | `kunalkachru23/medical-triage-env` | Canonical Round 1 submission; leave stable until you decide to update. |
| **Staging / experiments** | `kunalkachru23/medical-triage-env-staging` (or `-dev`, `-preview`) | All `openenv push` trials, risky changes, extra endpoints, case-bank spikes. |

**You must create the staging Space once** on Hugging Face (same account, new Space name). Then:

1. **Manual deploy (first trials):** `openenv push --repo-id kunalkachru23/medical-triage-env-staging` (or your chosen name).
2. **Release gate against staging:**  
   `./scripts/full_release_gate.sh --skip-deploy` and point live checks at the staging URL, **or**  
   `./scripts/full_release_gate.sh --repo-id kunalkachru23/medical-triage-env-staging --base-url https://kunalkachru23-medical-triage-env-staging.hf.space`  
   (adjust `base-url` to match the actual HF Space subdomain after creation.)
3. **`.env` / `setupCredentials.py`:** Set `SPACE_REPO_ID` to the **staging** repo while developing; switch back to production only when you intentionally promote a build.
4. **Scripts with defaults:** [`scripts/full_release_gate.sh`](../scripts/full_release_gate.sh) defaults `REPO_ID` to production — **always pass `--repo-id`** for staging pushes.

When you are ready to **promote** a build for judges / Phase 2 re-eval: merge to `main`, run the full gate against production, then push to `medical-triage-env` **manually**.

---

## 1. Rubric anchor (why this order)

Weights from [`docs/openenv_hackathon_criteria.md`](openenv_hackathon_criteria.md):

| Area | Weight | Focus here |
|------|--------|----------------|
| Real-world utility | 30% | Case bank, sepsis bundle, clinical citations |
| Task & grader quality | 25% | Cases, synonyms, discriminative hard tasks |
| Environment design | 20% | Rewards, fairness output, multi-turn clarity |
| Code quality & spec | 15% | Tests, `openenv validate`, Docker |
| Creativity & novelty | 10% | Sepsis task, optional uncertainty / datasets |

Agentic evaluation cares about **score variance** — avoid flat or trivially exploitable graders.

---

## 2. Tier A — Safest, judge-visible (do early)

| ID | Item | Notes |
|----|------|--------|
| A1 | **Web UI randomness** | [`server/app.py`](../server/app.py) `resetEnv`: stop sending fixed `seed: 42` for “Random case”; omit seed or use per-reset randomness so Training charts don’t cluster artificially. |
| A2 | **Demo + README judge path** | Hero video ([`assets/`](../assets/)), optional [`scripts/record_learning_curve_demo.py`](../scripts/record_learning_curve_demo.py); README: 30s “what to click” on HF. |
| A3 | **Baseline / variance doc** | Re-run [`inference.py`](../inference.py); optional second model; document task-wise spread in `docs/TEST_REPORT.md` or `docs/BASELINE_MATRIX.md`. |
| A4 | **Repo ↔ Space parity** | After any `openenv push`, diff README; align GitHub and HF when you promote. |

---

## 3. Tier B — Case bank + grader acceptance (high impact, additive)

**B1. Expand case bank** (target counts are goals; ship in batches with tests)

| Task | Current | Target (audit) | Examples (audit) |
|------|---------|----------------|------------------|
| simple_triage | 4 | 10 | COPD exacerbation, anaphylaxis, stroke, AKI, PE, hypoglycaemia |
| conflicting_vitals | 3 | 8 | DKA + normal BP, GI bleed + compensated HR, hypothyroid in AF |
| masked_deterioration | 5 | 10 | NSAIDs + peritonitis, antipsychotics + sepsis, ESRD creatinine trap, anticoagulants + bleed |
| demographic_fairness | 12 | (grow as needed) | More parity variants if time |
| deteriorating_patient | 4 | 7 | DKA trajectory, meningitis, hypertensive emergency |

Implementation: [`server/cases.py`](../server/cases.py), graders in [`server/graders.py`](../server/graders.py), tests under [`tests/`](../tests/).

**B2. Synonym / canonicalization in graders** (expand accepted answers only; add tests)

- Map phrases → canonical tokens (e.g. HR / heart rate / tachycardia → `heart_rate`).
- `condition`: sepsis / septicaemia / bacteraemia → agreed canonical.
- `masking_drug_or_condition`: drug classes ↔ exemplar drugs where appropriate.

**Safety:** Never remove previously accepted strings; add normalization layer + golden tests.

---

## 4. Tier C — Sixth task + API (medium lift; use staging first)

**C1. Task 6: Sepsis bundle compliance (novelty)**

- Agent selects Hour-1 bundle elements (cultures, antibiotics, fluids, lactate, vasopressors, etc.).
- New cases, grader, [`openenv.yaml`](../openenv.yaml), env dispatch, UI fields, tests — **touches multiple files**; schedule realistically.

**C2. Additive endpoints** (optional; document each)

| Endpoint | Purpose | Risk |
|----------|---------|------|
| `GET /compute-news2` | Debug / evaluator NEWS2 check | Low if read-only |
| `GET /benchmark` | Aggregate from history/metrics | Avoid duplicating `/metrics` + `/history` logic three ways |
| `POST /batch-step` | Batch actions | **Design:** session semantics, ordering; not core OpenEnv — document clearly |
| `GET /learning-curve` | Server-side series | UI already plots from `/history`; optional for external eval |

**C3. Fairness response enrichment**

- Extend [`/grade-fairness`](../server/app.py) with **additive** fields (e.g. per-demographic notes), not breaking changes.

**C4. Confidence calibration bonus**

- Audit plan: raise cap **0.05 → 0.10**. **Not risk-free:** updates reward distribution — full regression + tests before ship.

**C5. Clinical citations in hints**

- Improve hint strings on wrong answers (masked deterioration, etc.); verify sources.

---

## 5. Tier D — “Training story” without full TRL

- Harden [`train.py`](../train.py): CSV/metrics, before–after table, short section in [`docs/PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).
- Stretch: HF Dataset export `(observation, action, reward)` — only if time.

---

## 6. Tier E — Fairness artifact for humans

- Reproducible markdown table: FP001 variants + one “bad agent” example; link from README.

---

## 7. Tier F — Stretch / post–Round 1 (original backlog)

- **Paediatric NEWS2-P** task.
- **Medication reconciliation** task.
- **Continuous episode scoring** (long-horizon monitoring).
- **Uncertainty-aware** dual-valid-answer graders.
- **Leaderboard** (HF dataset + Space).
- **Structured logging** in `train.py` / `inference.py`.
- **OpenEnv spec** drift tracking.
- **Dockerfile** layer split for faster rebuilds.
- **PPO / GRPO** (TRL / OpenRLHF) — large follow-on.

---

## 8. Suggested calendar (Apr 9–12, adjustable)

| When | Focus | Deploy target |
|------|--------|----------------|
| Early | A1–A3 + B1 start (cases + tests) | **Staging** only |
| Mid | B2 synonyms; start C1 sepsis **or** C2 endpoints (pick depth) | Staging |
| Late | C3–C5 as time allows; D + E | Staging |
| Before final submit | `pytest`, `openenv validate`, `./scripts/validate-submission.sh` on **promoted** URL | **Production** only when you choose |

---

## 9. Pre-flight before any production push

- [ ] All tests green
- [ ] `openenv validate`
- [ ] `./scripts/full_release_gate.sh` (or validate-submission) against the **exact** URL you will submit
- [ ] README / `openenv.yaml` case counts consistent
- [ ] You explicitly approved git push + production `openenv push`

---

## 10. Reference scripts (local / staging)

- Episode seeding UI: [`scripts/browser_run_episodes_per_task.py`](../scripts/browser_run_episodes_per_task.py)
- Side-by-side demo MP4: [`scripts/record_learning_curve_demo.py`](../scripts/record_learning_curve_demo.py) (needs `ffmpeg`)

Replace `DEFAULT_BASE_URL` or pass flags to hit **staging** `.hf.space` URL.

---

## 11. Reliability + presentation polish plan (active)

**Goal:** remove noisy/flaky signals and tighten evaluator narrative for judge-facing confidence.

### Improvement Area 1 — Reliability signal

- [x] **R1. Harden `scripts/pre_submit_check.sh` startup health probe**
  - Add short initial settle (`sleep 2`), connection timeout, silent retries.
  - Keep failure strict when retries exhaust.
  - **Success criteria:** no `curl (56)` noise in normal runs.
  - **Status:** Implemented in script; requires full gate run confirmation.

- [x] **R2. Add explicit readiness wait helper in release path**
  - In `scripts/full_release_gate.sh`, after deploy, poll `/health` and `/reset` with bounded backoff before live checks.
  - **Success criteria:** no race where first live verify sees stale/not-ready revision.
  - **Status:** Implemented in script; requires staging deploy verification.

- [x] **R3. Add controlled retry wrapper for UI-heavy test only**
  - Wrap `scripts/full_browser_test.py` with one retry on known transient navigation timeout.
  - Log retry reason clearly; fail hard on repeated timeout.
  - **Success criteria:** stable CI signal while preserving strictness.
  - **Status:** Implemented as `scripts/run_full_browser_with_retry.sh` (single retry on known timeout signatures).

- [x] **R4. Emit machine-readable run artifact**
  - Output JSON summary for local + staging gates: command, status, duration, timestamp.
  - **Success criteria:** single evidence bundle for judges.
  - **Status:** Implemented in `scripts/pre_submit_check.sh` and `scripts/full_release_gate.sh` (writes under `artifacts/gates/`).

- [x] **R5. Add one-command confidence run**
  - Add `scripts/final_submission_check.sh` chaining canonical checks in strict order.
  - **Success criteria:** one command gives go/no-go.
  - **Status:** Implemented; runs coverage, pre-submit gate, retry-safe full browser suite, then release gate.

### Improvement Area 2 — Presentation clarity

- [x] **P1. Create `docs/EVALUATOR_BRIEF.md`**
  - One-page judge narrative: relevance, why environment is hard/real, task map, run steps, expected outputs.
  - **Status:** Implemented (judge-facing one-page brief added).

- [x] **P2. Create `docs/EVIDENCE_SUMMARY.md`**
  - Single source of truth: latest test totals, local gate, staging/prod verify, endpoint checks, artifact links.
  - **Guardrail:** prevent cross-doc metric drift.
  - **Status:** Implemented; includes restamp instructions after next full confidence run.

- [ ] **P3. Add submission checklist section in `README.md`**
  - Explicit pass table with date and command references (e.g., `openenv validate`, `full_browser_test`).

- [ ] **P4. Standardize counts/claims across docs**
  - Make docs point to canonical metrics in `docs/EVIDENCE_SUMMARY.md`.
  - Label old numbers as historical where needed.

- [ ] **P5. Add known limits + mitigations**
  - Document controlled transient risks and how scripts mitigate them.

### Implementation sequence (recommended)

1. R1 + R2 (largest reliability gains)
2. R3 + R4
3. P1 + P2 (core evaluator packet)
4. P3 + P4 + P5
5. Re-run full gate and stamp evidence docs with fresh results
