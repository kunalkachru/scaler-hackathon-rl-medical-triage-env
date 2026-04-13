# Training Results

---

## GRPO Fine-tuning — Run 3 (Qwen2.5-1.5B-Instruct)

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16, α=32)
**Adapter:** `kunalkachru23/grpo-medical-triage-qwen1.5b`
**Config:** 11 tasks × 8 prompts = 88 prompts | 2 epochs | 170 steps | G=4
**Date:** 2026-04-13 | Reward oracle: production HF Space

### Per-task Mean Reward (from training log)

| Task | Mean Reward | Peak | Notes |
|---|---|---|---|
| `demographic_fairness` | ~0.47 | 0.88 | Best performer — frequent 0.67+ hits |
| `paediatric_triage` | ~0.41 | 0.77 | Strong — consistent 0.52-0.77 |
| `sepsis_bundle` | ~0.38 | 0.65 | Good — bundle elements learned |
| `medication_reconciliation` | ~0.32 | 0.62 | Improving — severity/action fields |
| `simple_triage` | ~0.28 | 0.75 | Solid baseline |
| `conflicting_vitals` | ~0.27 | 0.44 | Moderate |
| `icu_deterioration` | ~0.21 | 0.67 | High variance — SOFA scoring sporadic |
| `masked_deterioration` | ~0.18 | 0.30 | Weak — masking drug field rarely correct |
| `sbar_handover` | ~0.19 | 0.97 | Bimodal — either floor or near-perfect |
| `deteriorating_patient` | ~0.12 | 0.9999 | Unstable — multi-turn hard for 1.5B |
| `differential_diagnosis` | ~0.05 | 0.47 | Weakest — must_not_miss reasoning needs larger model |
| **Overall** | **~0.27** | | vs random baseline ~0.29 (random had valid JSON always) |

### Loss trajectory (170 steps)

Steps 1–40: Loss oscillated ±0.2 (exploration phase)
Steps 40–90: Loss stabilised, positive reward signal emerging on simple_triage, paediatric, sepsis
Steps 90–170: Consistent learning — demographic_fairness, sepsis_bundle, medication_reconciliation showing repeated 0.4–0.7 rewards

### Key observations

- **1.5B model ceiling**: `differential_diagnosis` and `sbar_handover` require multi-step clinical reasoning that a 1.5B parameter model cannot reliably perform. Larger model (7B+) needed for these tasks.
- **Format learning confirmed**: Tasks with strict but learnable schemas (paediatric_triage, sepsis_bundle) show clear reward improvement vs floor.
- **Bimodal reward pattern** on new tasks (ICU/SBAR) indicates format is being learned intermittently — more training steps would consolidate this.
- **Run 2 → Run 3 improvement**: Run 2 trained on 8 tasks with corrupted observations (`patient_history` bug). Run 3 fixes both: all 11 tasks with correct observations.

---

## RL Training Loop — Llama-3.3-70B-Instruct (via HF Router)

Model: `meta-llama/Llama-3.3-70B-Instruct`  |  Reps per task: 3


## Before/After Reward Table

| Task | First Score | Last Score | Δ | Trend |
|------|-------------|------------|---|-------|
| `simple_triage` | 0.925 | 1.000 | +0.075 | ▲ IMPROVED |
| `conflicting_vitals` | 0.265 | 0.467 | +0.202 | ▲ IMPROVED |
| `masked_deterioration` | 0.900 | 0.900 | +0.000 | ≈ STABLE |
| `demographic_fairness` | 0.820 | 0.410 | -0.410 | ▼ REGRESSED |
| `deteriorating_patient` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `sepsis_bundle` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `paediatric_triage` | 0.425 | 0.425 | +0.000 | ≈ STABLE |
| `medication_reconciliation` | 0.450 | 0.537 | +0.087 | ▲ IMPROVED |
| `icu_deterioration` | 0.520 | 0.770 | +0.250 | ▲ IMPROVED |
| `sbar_handover` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `differential_diagnosis` | 1.000 | 1.000 | +0.000 | ≈ STABLE |

## Attempt-by-Attempt Scores

### simple_triage

| Attempt | Score |
|---------|-------|
| 1 | 0.925 |
| 2 | 1.000 |
| 3 | 1.000 |

### conflicting_vitals

| Attempt | Score |
|---------|-------|
| 1 | 0.265 |
| 2 | 0.507 |
| 3 | 0.467 |

### masked_deterioration

| Attempt | Score |
|---------|-------|
| 1 | 0.900 |
| 2 | 0.900 |
| 3 | 0.900 |

### demographic_fairness

| Attempt | Score |
|---------|-------|
| 1 | 0.820 |
| 2 | 0.410 |
| 3 | 0.410 |

### deteriorating_patient

| Attempt | Score |
|---------|-------|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

### sepsis_bundle

| Attempt | Score |
|---------|-------|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

### paediatric_triage

| Attempt | Score |
|---------|-------|
| 1 | 0.425 |
| 2 | 0.425 |
| 3 | 0.425 |

### medication_reconciliation

| Attempt | Score |
|---------|-------|
| 1 | 0.450 |
| 2 | 0.737 |
| 3 | 0.537 |

### icu_deterioration

| Attempt | Score |
|---------|-------|
| 1 | 0.520 |
| 2 | 0.830 |
| 3 | 0.770 |

### sbar_handover

| Attempt | Score |
|---------|-------|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

### differential_diagnosis

| Attempt | Score |
|---------|-------|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

