# Training Results

Model: `meta-llama/Llama-3.3-70B-Instruct`  |  Reps per task: 3


## Before/After Reward Table

| Task | First Score | Last Score | Δ | Trend |
|------|-------------|------------|---|-------|
| `simple_triage` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `conflicting_vitals` | 0.265 | 0.480 | +0.215 | ▲ IMPROVED |
| `masked_deterioration` | 0.900 | 0.900 | +0.000 | ≈ STABLE |
| `demographic_fairness` | 0.820 | 0.410 | -0.410 | ▼ REGRESSED |
| `deteriorating_patient` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `sepsis_bundle` | 1.000 | 1.000 | +0.000 | ≈ STABLE |
| `paediatric_triage` | 0.425 | 0.425 | +0.000 | ≈ STABLE |
| `medication_reconciliation` | 0.250 | 0.717 | +0.467 | ▲ IMPROVED |

## Attempt-by-Attempt Scores

### simple_triage

| Attempt | Score |
|---------|-------|
| 1 | 1.000 |
| 2 | 1.000 |
| 3 | 1.000 |

### conflicting_vitals

| Attempt | Score |
|---------|-------|
| 1 | 0.265 |
| 2 | 0.480 |
| 3 | 0.480 |

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
| 1 | 0.250 |
| 2 | 0.537 |
| 3 | 0.717 |

