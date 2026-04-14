# Evaluator Brief (Judge-Facing)

Last updated: 2026-04-14

## 1) Why this environment matters

`medical-triage-env` is an OpenEnv-compliant RL environment for clinical safety tasks where wrong decisions can cause real harm. It is designed for agent training and evaluation on realistic hospital triage patterns, not toy tasks.

It targets clinically documented failure modes:
- demographic bias in triage decisions
- masked deterioration (drug/condition effects hiding danger)
- delayed escalation in worsening patients
- must-not-miss diagnostic safety-net failures

## 2) What makes it hard and real

- 11 tasks, 75 cases, easy -> medium -> hard progression
- deterministic graders grounded in NEWS2, PEWS, SOFA, SBAR, and sepsis bundle logic
- dense partial-credit rewards (not binary pass/fail)
- open-interval reward contract on API `(0.0001, 0.9999)` for OpenEnv compatibility
- multi-turn deterioration task (`max_steps_per_episode: 3`)

## 3) Task map (at a glance)

- `simple_triage` (easy)
- `conflicting_vitals` (medium)
- `masked_deterioration` (hard)
- `demographic_fairness` (medium)
- `deteriorating_patient` (hard, multi-turn)
- `sepsis_bundle` (hard)
- `paediatric_triage` (hard)
- `medication_reconciliation` (hard)
- `icu_deterioration` (hard)
- `sbar_handover` (medium)
- `differential_diagnosis` (hard)

## 4) What to run (canonical order)

From project root:

```bash
python scripts/check_coverage.py
./scripts/pre_submit_check.sh
./scripts/run_full_browser_with_retry.sh --base-url "https://<your-space>.hf.space"
./scripts/full_release_gate.sh --base-url "https://<your-space>.hf.space" --repo-id "<user>/<space>" --expect-llm true
```

Single-command go/no-go:

```bash
./scripts/final_submission_check.sh --base-url "https://<your-space>.hf.space" --repo-id "<user>/<space>" --expect-llm true
```

Operational behavior:
- Default behavior is a fresh full run.
- If interrupted, rerun with `--resume` to continue incomplete stages.

## 5) Expected outputs

- `check_coverage.py`: PASS (all tasks represented across required artifacts)
- `pre_submit_check.sh`: tests + docker + reset smoke + `openenv validate` pass
- `run_full_browser_with_retry.sh`: full browser/API suite pass (single controlled retry only on known transient nav timeout)
- `full_release_gate.sh`: end-to-end local + live checks pass
- machine-readable evidence artifacts:
  - `artifacts/gates/pre_submit_check_summary.json`
  - `artifacts/gates/full_release_gate_summary.json`

## 6) Evidence and metrics source of truth

Use `docs/EVIDENCE_SUMMARY.md` as the canonical source for:
- latest test totals
- local/staging/production gate status
- endpoint verification state
- command traceability

## 7) Known limits (transparent)

- hard tasks can show higher variance than easy tasks by design
- transient HF/Playwright timing can occur; mitigated via bounded readiness waits and strict one-retry UI wrapper
- reliability hardening was recently added; re-stamp evidence after the next full gate run
