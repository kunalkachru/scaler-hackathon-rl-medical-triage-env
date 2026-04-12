# TODO: Round 1 enhancement roadmap (merged plan)

**Status:** v2.0.0 — 119 tests passing; Phase 1 submission cleared. Extended deadline **April 12, 2026** (see organizer comms).

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
