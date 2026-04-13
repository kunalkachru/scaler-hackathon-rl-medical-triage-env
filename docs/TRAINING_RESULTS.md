# Training Results

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

