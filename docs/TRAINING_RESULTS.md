# Training Results

Model: `meta-llama/Llama-3.3-70B-Instruct`  |  Reps per task: 3


## Before/After Reward Table

| Task | First Score | Last Score | Δ | Trend |
|------|-------------|------------|---|-------|
| `simple_triage` | 0.765 | 0.925 | +0.160 | ▲ IMPROVED |
| `conflicting_vitals` | 0.265 | 0.480 | +0.215 | ▲ IMPROVED |
| `masked_deterioration` | 0.675 | 0.900 | +0.225 | ▲ IMPROVED |

## Attempt-by-Attempt Scores

### simple_triage

| Attempt | Score |
|---------|-------|
| 1 | 0.765 |
| 2 | 1.000 |
| 3 | 0.925 |

### conflicting_vitals

| Attempt | Score |
|---------|-------|
| 1 | 0.265 |
| 2 | 0.480 |
| 3 | 0.480 |

### masked_deterioration

| Attempt | Score |
|---------|-------|
| 1 | 0.675 |
| 2 | 0.900 |
| 3 | 0.900 |

