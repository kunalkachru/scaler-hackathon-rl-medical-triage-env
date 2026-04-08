# TODO: Future Enhancements

Current status: v2.0.0 — 119 tests passing, all hackathon criteria met.

The items below are post-submission ideas ordered by impact.

---

## High Impact

### 1. Additional Task Types

- **Paediatric triage**: separate NEWS2-P scoring table, age-appropriate normal ranges.
- **Sepsis bundle task**: agent must identify and recommend correct bundle elements (blood cultures, lactate, antibiotics timing).
- **Medication reconciliation**: agent reviews drug list for interactions that mask deterioration.

Each new task requires: case set in `server/cases.py`, grader in `server/graders.py`, dispatch registration, action field extension, and tests.

### 2. Larger Case Bank

Current case counts per task are small (3–5 cases). Expanding to 20+ cases per task would:
- reduce overfitting risk in training loops
- enable proper train/val/test splits
- support more diverse demographic variants for fairness evaluation

### 3. Real Model RLHF Integration

`train.py` currently demonstrates reward-conditioned prompting (in-context learning from rewards). A natural extension is:
- PPO or GRPO fine-tuning using episode rewards as the training signal
- Requires exporting reward signals as a dataset or integrating with a training framework (TRL, OpenRLHF)

---

## Medium Impact

### 4. Continuous Episode Scoring

Currently each episode ends after one step (or 3 steps for deteriorating patient). A continuous scoring mode where the agent monitors a patient across many timesteps would better simulate real ICU monitoring.

### 5. Uncertainty-Aware Cases

Add cases where ground truth is ambiguous (two valid priorities). Grader would accept a range and reward calibrated uncertainty rather than a single answer.

### 6. Leaderboard Integration

The `/stats` endpoint already tracks per-task aggregate scores. A public leaderboard (HF Spaces + dataset) would enable community benchmarking.

---

## Low Impact / Maintenance

### 7. Structured Logging

Replace `print()` statements in `train.py` and `inference.py` with structured JSON logs for easier parsing by training pipelines.

### 8. OpenEnv v2 Spec Compliance

Track any updates to the OpenEnv spec (metadata, schema, MCP endpoints) and update `openenv.yaml` and `server/app.py` accordingly.

### 9. Dockerfile Layer Optimisation

Current Dockerfile copies all source files in one layer. Split into dependency layer (requirements.txt) + source layer to improve rebuild times during iterative development.
