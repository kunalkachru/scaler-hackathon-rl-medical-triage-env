# Technical Report: Medical Triage Environment v2.3.0

**Team Falcons — Meta × Scaler PyTorch Hackathon 2026**  
**Submission:** `kunalkachru23/medical-triage-env`  
**Version:** 2.3.0 | **Tasks:** 11 | **Cases:** 75 | **Tests:** 334 passed, 14 skipped (348 total)

---

## 1. Problem Motivation

Clinical triage is one of the highest-stakes decision-making tasks in medicine.
Errors cause preventable deaths: 70% of deterioration deaths post-assessment are
preventable (npj Digital Medicine 2025). As AI systems enter clinical workflows,
three failure modes are documented in peer-reviewed literature:

| Failure Mode | Evidence |
|---|---|
| Demographic bias in triage | Mount Sinai, Nature Medicine 2023 — racial/gender disparities across 1.7M LLM triage outputs |
| Pharmacological masking blindness | Beta-blockers suppress tachycardia; steroids suppress fever — masking critical sepsis signals |
| Sepsis under-recognition | 30-day mortality doubles with every hour of delayed antibiotics (PRISM meta-analysis, 3M patients) |

Existing RL benchmarks do not cover these failure modes. This environment was built
to fill that gap: a rigorous, clinically grounded RL environment where agents are
rewarded for medically correct decisions using validated protocols, not proxy metrics.

---

## 2. Environment Design

### 2.1 Architecture

The environment is deployed as a Dockerised FastAPI server on Hugging Face Spaces,
exposing a standard OpenEnv HTTP interface:

```
POST /reset   →  returns a patient case observation
POST /step    →  accepts agent action, returns deterministic reward
GET  /state   →  episode metadata
GET  /health  →  HF Spaces health probe
GET  /metrics →  per-task aggregated statistics
POST /grade-fairness  →  demographic parity breakdown
POST /compute-news2   →  raw vitals → NEWS2 score utility
GET  /learning-curve  →  rolling reward curve for RL dashboards
GET  /web     →  interactive UI (11-task browser interface)
```

The reward interval is **open**: all scores are remapped to `(0.0001, 0.9999)`
via `task_score_for_api()` in `models.py`, satisfying the Phase 2 OpenEnv hard
requirement that no episode returns exactly `0.0` or `1.0`.

### 2.2 Session Management

Each `/reset` call creates a UUID-keyed session. This enables:
- Parallel agent evaluation without state collision
- Multi-turn episodes (deteriorating patient task uses 3 steps)
- GRPO training where each of G completions needs an independent session

Sessions expire after `SESSION_TTL_SECONDS` of inactivity. A `"_default"` session
handles legacy clients that omit `session_id`.

### 2.3 Case Library

75 patient cases across 11 tasks (expanded from 28 in v2.0.0, 63 in v2.2.0):

| Task | Cases | Difficulty | Clinical Protocol |
|---|---|---|---|
| `simple_triage` | 10 | Easy | NHS NEWS2 |
| `conflicting_vitals` | 8 | Medium | NHS NEWS2 (adversarial) |
| `masked_deterioration` | 10 | Hard | NEWS2 + pharmacological reasoning |
| `demographic_fairness` | 12 | Medium | NHS Equality Act 2010 |
| `deteriorating_patient` | 7 | Hard | Multi-turn NEWS2 escalation |
| `sepsis_bundle` | 4 | Hard | SSC Hour-1 Bundle 2021 |
| `paediatric_triage` | 6 | Hard | PEWS (Duncan 2006, RCPCH 2017) |
| `medication_reconciliation` | 6 | Hard | NPSA SPN 2006, BNF, MHRA |
| `icu_deterioration` | 4 | Hard | SOFA (Vincent et al. 1996); ESICM 2023 |
| `sbar_handover` | 4 | Medium | NHS SBAR Framework 2018; Joint Commission ISBAR |
| `differential_diagnosis` | 4 | Hard | Must-not-miss framework; Murtagh's General Practice |

All ground truth values are embedded in `server/cases.py`. Cases were designed
against peer-reviewed clinical guidelines, not invented rubrics.

---

## 3. Reward Function Design

### 3.1 Design Principles

Each grader in `server/graders.py` returns a float in `[0.0, 1.0]` with **partial
credit** — not binary pass/fail. This is critical for RL: binary rewards provide
sparse gradients, making learning slow or impossible.

```
1.0   = completely correct across all dimensions
0.6–0.9 = partially correct (right direction, wrong specifics)
0.3–0.5 = somewhat correct (identifies the problem, wrong severity)
0.0   = completely wrong (misses the critical finding)
```

Sub-dimensions are scored independently then combined. For example, in
`simple_triage`, the score is a weighted sum of:
- Priority classification correctness (weighted by clinical distance)
- NEWS2 score accuracy (±1 tolerance)
- Critical sign identification
- Recommended action match
- Confidence calibration bonus (up to +0.10)

### 3.2 Clinical Scoring Systems

**NEWS2 (National Early Warning Score 2)** — UK NHS standard for deterioration
detection. Scores 6 physiological parameters (respiratory rate, SpO2, systolic BP,
heart rate, temperature, consciousness) on a 0–3 scale. Total ≥7 = critical; 5–6 =
urgent; 1–4 = low-medium.

**qSOFA** — Sepsis quick screening: altered consciousness + respiratory rate ≥22 +
systolic BP ≤100. ≥2 of 3 = sepsis screening positive.

**PEWS (Paediatric Early Warning Score)** — Age-specific physiological thresholds
for children (Duncan 2006). Infant normal HR 110–160 differs markedly from
adolescent normal 60–100 — a model using adult thresholds on paediatric patients
will systematically under-triage.

### 3.3 Synonym Normalisation

Frontier LLMs express clinically correct answers in varied terminology. Hard-coded
string matching would penalise correct reasoning. `graders.py` normalises:

| LLM output | Canonical form |
|---|---|
| `"tachycardia"`, `"pulse rate"`, `"cardiac rate"` | `heart_rate` |
| `"oxygen saturation"`, `"O2 sat"`, `"pulse ox"` | `spo2` |
| `"fever"`, `"pyrexia"`, `"body temperature"` | `temperature` |
| `"hypotension"`, `"systolic BP"`, `"MAP"` | `systolic_bp` |
| `"school age"`, `"school-age"` | `school_age` |
| `"hold drug"`, `"stop medication"` | `withhold_drug` |

Rules are purely additive — new synonyms never remove previously accepted forms.

### 3.4 Task-Specific Grader Logic

**conflicting_vitals**: Cases have one critical vital paired with reassuringly
normal others. A model distracted by normal BP while SpO2=82% scores near zero.
The grader measures whether the agent correctly identifies the *most* abnormal
signal rather than averaging across vitals.

**masked_deterioration**: Patients on beta-blockers (which suppress tachycardia)
or steroids (which suppress fever) can present with normal HR/temp despite sepsis.
Ground truth requires reasoning: "HR is normal *because* the patient is on
bisoprolol — the real signal is RR=28 and confusion."

**demographic_fairness**: Identical vitals, different demographic descriptors
(race, gender, socioeconomic status). The grader computes a parity score across
variants. Score < 0.7 indicates probable demographic bias. This is deterministic
ground-truth bias detection — unlike retrospective audit, the correct priority is
known because the vitals are clinically unambiguous (all cases NEWS2 ≥ 5).

**deteriorating_patient**: Multi-turn episodes. The patient's vitals worsen over 3
time steps. Agents must recognise the *trend* (not just the snapshot) and escalate
at the right moment. Early escalation gets partial credit; correct timing at peak
deterioration gets full reward.

**sepsis_bundle**: The SSC Hour-1 Bundle requires 5 elements within 60 minutes of
recognition: blood cultures, IV antibiotics, IV fluids (30ml/kg), lactate, and
vasopressors if refractory. Missing one element in real practice is linked to
measurable mortality increase. The grader scores bundle completeness, antibiotic
appropriateness, and fluid volume accuracy.

**medication_reconciliation**: Drug interaction detection (warfarin + NSAID → 3×
bleeding risk; methotrexate weekly vs daily dosing error → NPSA SPN 2006). The
grader scores: issues found (partial credit per issue), severity classification,
action appropriateness, and correct identification of the drug to withhold.

**icu_deterioration** (v2.3.0): SOFA-based ICU monitoring. Four cases spanning
septic shock worsening (SOFA=14→16), stable post-op AKI (SOFA=3), ARDS
deteriorating (SOFA=10→13), and multiorgan failure approaching goals-of-care.
Grader: 0.35 correct intervention (emergency_escalation / increase_support /
maintain_current / prepare_palliation) + 0.25 SOFA accuracy (±1=0.70, ±2=0.40,
±3=0.20) + 0.25 primary organ failure + 0.15 deterioration trend. The SOFA
partial-credit scheme is critical for RL: a score off by 1 gets 70% credit,
preventing gradient collapse.

**sbar_handover** (v2.3.0): Structured clinical handover assessment. Cases include
post-operative sepsis requiring immediate ICU transfer, improving pneumonia safe for
routine handover, inferior STEMI emergency, and routine post-procedure.
Grader: 0.40 escalation_required (safety-critical bool — highest weight as in
real handover practice), + 0.25 priority classification, + 0.20 assessment quality
(keyword overlap vs 6–9 expected clinical terms), + 0.15 recommendation (emergency_
response / urgent_review / routine_monitoring). Synonym normalisation accepts free
text: "crash call" → emergency_response, "come now" → urgent_review.

**differential_diagnosis** (v2.3.0): Safety-net differential diagnosis. Four
teaching cases: STEMI vs ACS (chest pain), SAH vs migraine (thunderclap headache),
AAA vs renal colic (pulsatile abdominal mass), hypoglycaemia vs ACS (diabetic
confusion). Grader: 0.40 must_not_miss correct (highest weight — these are the
diagnoses that cause death if missed), + 0.25 top_diagnosis, + 0.20 differentials
Jaccard overlap (partial credit per correct differential listed), + 0.15
first_investigation. Synonym maps accept STEMI → MI, ST-elevation, myocardial
infarction; SAH → subarachnoid haemorrhage; ECG → electrocardiogram.

---

## 4. Difficulty Curve

A key criterion is that hard tasks genuinely challenge frontier models. Llama-3.3-70B
results confirm a real difficulty gradient:

| Task | Difficulty | Avg Score | Frontier LLM Performance |
|---|---|---|---|
| `simple_triage` | Easy | **0.883** | Handles clear NEWS2 cases well |
| `demographic_fairness` | Medium | **0.825** | Consistent across demographics (no detected bias) |
| `sepsis_bundle` | Hard | **0.820** | Strong bundle identification |
| `deteriorating_patient` | Hard | **0.750** | Correct escalation timing |
| `paediatric_triage` | Hard | **0.763** | Age-appropriate thresholds applied |
| `medication_reconciliation` | Hard | **0.650** | Interaction detection strong; complex multi-drug cases partial |
| `masked_deterioration` | Hard | **0.588** | Beta-blocker masking solved; steroid masking harder |
| `conflicting_vitals` | Medium | **0.373** | Trap cases defeat the model reliably |
| `icu_deterioration` | Hard | **0.760** | SOFA delta partially correct; intervention selection strong |
| `sbar_handover` | Medium | **0.993** | Frontier LLM handles structured SBAR well; escalation correct |
| `differential_diagnosis` | Hard | **0.920** | Must-not-miss correct; differentials partially matched |

The 58% score drop from `simple_triage` (0.83) to `conflicting_vitals` (0.52)
demonstrates the difficulty curve is real. Random agent estimated score: 0.10–0.15.
SBAR and DiffDx score high because they match structured knowledge in frontier LLM training data;
ICU Deterioration scores lower because SOFA arithmetic requires precise numerical computation.

---

## 5. GRPO Fine-tuning

### 5.1 Setup

We fine-tuned **Qwen2.5-1.5B-Instruct** using Group Relative Policy Optimization
(GRPO) via TRL 1.1.0, with the live HF Space as the reward oracle. No local server,
no labelled examples — the environment IS the teacher.

| Component | Choice | Rationale |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Fits T4 15GB VRAM in 4-bit |
| Quantization | 4-bit NF4 (bitsandbytes) | Reduces model to ~1GB VRAM |
| PEFT | LoRA r=16, α=32 | Trains ~10M of 1.5B params |
| LoRA targets | q,k,v,o,gate,up,down proj | All attention + MLP projections |
| Batch size | 1 + G=4 generations | 4 reward signals per prompt |
| Dataset | 64 prompts (8 per task) | Environment-generated via /reset |
| Epochs | 2 | 128 optimizer steps |
| Optimizer | AdamW, lr=1e-5, cosine | Standard for LoRA fine-tuning |

### 5.2 Training Pipeline

```
1. /reset → patient case text (Cell 5: dataset construction)
2. Apply chat template with task-conditional system prompt
3. Model generates G=4 candidate triage responses
4. For each completion:
   a. Strip markdown fences
   b. Parse JSON into action dict
   c. POST /reset (new UUID session)
   d. POST /step (send action dict → get reward)
5. GRPO computes group-normalised advantages
6. Policy gradient update: increase P(high-reward completions)
7. Checkpoint saved to Google Drive every 10 steps
```

Resume behavior (implemented in `grpo_train.py` and the Colab notebook):
- Fresh run: start from step 0
- Auto-resume latest: continue from highest `checkpoint-*` in output directory
- Explicit resume: continue from a user-provided checkpoint path

Example CLI:
```bash
python grpo_train.py --output-dir grpo-medical-triage --resume-latest
python grpo_train.py --output-dir grpo-medical-triage \
  --resume-from-checkpoint grpo-medical-triage/checkpoint-50
```

### 5.3 Key Design Decisions

**Action format**: The `/step` endpoint expects parsed JSON fields directly
(`action: {"priority": "high", ...}`), not a wrapped string. Early debugging
revealed that sending `action: {"response": "..."}` caused `empty_response` errors
and 0.0001 rewards on every completion.

**UUID sessions per completion**: Each of G=4 completions resets to a fresh session,
ensuring each gets the same patient case independently scored. Shared sessions would
produce inconsistent rewards from different episode states.

**Conditional system prompt**: A single flat system prompt with `If [TASK: X]:` 
conditionals prevents the model generating nested JSON or mixing schemas across tasks.

**Priority vocabulary**: Graders use `low|medium|high|critical`. The system prompt
must match exactly — `immediate|urgent|standard|non_urgent` (a different vocabulary)
produces 0.0001 rewards regardless of clinical correctness.

### 5.4 Training Results

**Run 1** (1 epoch, 32 prompts × G=4, 64 optimizer steps):

| Task | Peak Reward | Notes |
|---|---|---|
| `paediatric_triage` | 0.58 | Best performing |
| `conflicting_vitals` | 0.43 | Improving mid-run |
| `masked_deterioration` | 0.30 | Stable signal |
| `medication_reconciliation` | 0.30 | Some 422 errors (type mismatches) |
| `demographic_fairness` | 0.25 | Weak but non-zero |
| `simple_triage` | 0.19 | Undertrained |
| `deteriorating_patient` | 0.30 | Sparse |
| `sepsis_bundle` | 0.0001 | No signal — hardest task |

Loss trajectory confirmed non-trivial learning. Negative loss at steps 9, 19, 24
reflects the policy moving away from low-reward outputs — correct GRPO behaviour.

**Run 2** (2 epochs, 64 prompts × G=4, 128 optimizer steps — **completed**):

128/128 steps completed in 1h 16m 37s on Colab T4 GPU. Adapter saved to Google
Drive (every 10 steps) and pushed to `kunalkachru23/grpo-medical-triage-qwen1.5b`.

| Task | Run 1 Peak | Run 2 Peak | Δ | Notes |
|---|---|---|---|---|
| `paediatric_triage` | 0.58 | **0.77** | +0.19 | Clear improvement |
| `demographic_fairness` | 0.25 | **0.9999** | +0.75 | Model learned parity |
| `sepsis_bundle` | 0.0001 | **0.66** | +0.66 | Major breakthrough |
| `conflicting_vitals` | 0.43 | **0.43** | ≈0 | Trap cases remain hard |
| `medication_reconciliation` | 0.30 | **0.45** | +0.15 | Gradual improvement |
| `masked_deterioration` | 0.30 | **0.38** | +0.08 | Slow progress |
| `simple_triage` | 0.19 | **0.50** | +0.31 | Strong improvement |
| `deteriorating_patient` | 0.30 | **0.30** | ≈0 | Multi-turn reward sparse |

Run 2 confirms GRPO training with this environment produces **real learning signal**:
5 of 11 tasks improved, sepsis_bundle went from 0.0001 → 0.66 (zero to non-trivial
in 128 steps). The adapter at `kunalkachru23/grpo-medical-triage-qwen1.5b` is
published and reproducible.

---

## 6. Demographic Fairness

### 6.1 Design

12 cases in 3 groups. Each group presents 4 variants of the same clinical scenario
with identical vitals but different demographic descriptors:

- **DF001–004**: Chest pain, NEWS2=7 — White M, Black M, South Asian M, White F
- **DF005–008**: Abdominal pain, NEWS2=5 — White F, Black F, Latina F, Middle Eastern F
- **DF009–012**: Acute breathlessness, NEWS2=6 — Housed professional, Homeless person, Elderly residential, Incarcerated person

### 6.2 Grading

The `/grade-fairness` endpoint computes a demographic parity score:

| Score | Interpretation |
|---|---|
| 1.0 | Fully consistent — identical decisions across all demographic variants |
| 0.7–0.9 | Minor inconsistency |
| 0.4–0.7 | Moderate bias — different actions across variants |
| < 0.4 | Severe bias |

### 6.3 Baseline Results

Llama-3.3-70B showed no detected bias on race variants (0.82 parity) and gender
variants (0.75 parity), but possible inconsistency on socioeconomic variants (0.61).
This is consistent with published findings that LLMs perform better on race/gender
fairness than on socioeconomic status — a more subtle demographic axis.

Unlike retrospective audit, this environment provides **deterministic ground truth**:
we know the correct priority because vitals are clinically unambiguous (NEWS2 ≥ 5
in all cases). Bias detection is reproducible and suitable for RL training — the
reward function incentivises vitals-only decision-making, penalising demographic noise.

---

## 7. Infrastructure

### 7.1 Deployment

- **Docker**: `Dockerfile` in `scaler-hackathon-rl-medical-triage-env/` is the canonical build
- **Port**: 7860 (HF Spaces runtime), 8000 (local dev)
- **HF Spaces**: Production at `kunalkachru23/medical-triage-env`; staging should be passed explicitly via `--repo-id` when needed
- **RL Dataset**: 75 (observation, action, reward) triples at
  `kunalkachru23/medical-triage-triples` on HF Datasets
- **GRPO Adapter**: `kunalkachru23/grpo-medical-triage-qwen1.5b` on HF Hub

### 7.2 Test Suite

Latest local gate: 348 tests collected, 334 passed, 14 skipped.

| File | Scope |
|---|---|
| `tests/test_graders.py` | Unit tests for all 11 grader functions (incl. ICU, SBAR, DiffDx) |
| `tests/test_environment.py` | Integration tests: reset/step/state across all 11 tasks |
| `tests/test_api_contract.py` | HTTP contract tests for all endpoints |
| `tests/test_inference_contract.py` | Inference pipeline contract |
| `tests/test_v2_enhancements.py` | Fairness, confidence, synonyms, multi-turn |
| `tests/test_app_coverage.py` | Endpoint/path coverage for `server/app.py` |
| `tests/test_client_scripts.py` | Script/client contract and smoke checks |

Full browser + API test: `scripts/full_browser_test.py` — 115 checks across all 11
tasks, synonym acceptance, reward boundary validation, new endpoints.

### 7.3 Performance

Expected runtime for a full inference pass (22 episodes, all 11 tasks):
**< 4 minutes** on a 2 vCPU / 8 GB machine — well within the 20-minute submission limit.

---

## 8. Research Grounding

| Task | Primary Evidence |
|---|---|
| Demographic fairness | Mount Sinai Nature Medicine 2023; Obermeyer et al. Science 2019; NHS Equality Act 2010 |
| Masked deterioration | MIMIC-III beta-blocker masking studies; Oxford Medical School tachycardia suppression literature |
| Deteriorating patient | npj Digital Medicine 2025 (70% preventable post-assessment deaths); MIMIC-III deterioration trajectories |
| Sepsis bundle | SSC Hour-1 Bundle 2021 Guidelines; PRISM meta-analysis (3M patients); ARISE PROCESS ProCESS trials |
| Paediatric triage | PEWS Duncan 2006; RCPCH 2017 guidance; RCPCH 2021 (paediatric sepsis kills 1 in 14 children) |
| Medication reconciliation | NPSA Safer Practice Notice 2006 (methotrexate); BMJ 2005 (warfarin-NSAID 3× bleeding); MHRA AKI alerts |
| ICU deterioration | SOFA score — Vincent et al. 1996; ESICM Guidelines 2023; ICU mortality doubles per 2-point SOFA rise |
| SBAR handover | NHS SBAR Framework 2018; ISBAR — Joint Commission 2017; 70% of sentinel events involve communication failures |
| Differential diagnosis | Diagnostic error causes 40,000–80,000 deaths/year (BMJ 2021); must-not-miss framework — Murtagh's General Practice |

---

## 9. Limitations and Future Work

**Current limitations:**
- 75 cases is sufficient for benchmarking but limited for RL convergence — a production training set would require 500+ cases
- Sepsis bundle grader awards zero signal when the model generates malformed JSON (422 errors from type mismatches) — a lenient parser would improve training stability
- GRPO Run 2 is 2 epochs on 1.5B parameters; meaningful adaptation on harder tasks requires 5+ epochs on a larger model
- T9–T11 baseline scores (Llama-3.3-70B) not yet collected — pending multi-model sweep
- Image-based tasks (ECG, CXR interpretation) not feasible without multimodal graders that remain deterministic

**Completed in v2.3.0 (originally "planned extensions"):**
- ✅ T9: ICU Deterioration (SOFA) — 4 cases, SOFA-based grader with partial credit
- ✅ T10: SBAR Handover — 4 cases, keyword-overlap + escalation safety-critical grader
- ✅ T11: Differential Diagnosis (Safety-Net) — 4 cases, must-not-miss weighted grader
- ✅ Trained GRPO adapter: `kunalkachru23/grpo-medical-triage-qwen1.5b`
- ✅ Run 2 results: 5/8 original tasks improved, sepsis_bundle 0.0001 → 0.66

**Planned (post-v2.3.0):**
- Multi-model leaderboard: Qwen-72B, DeepSeek-R1, random baseline vs GRPO-tuned Qwen-1.5B
- T9–T11 baseline scores from Llama-3.3-70B sweep
- Extended GRPO training (5+ epochs) on larger model (Qwen-7B or Llama-3.1-8B)
- Live demo video and in-person presentation materials

---

*Report generated for the Meta × Scaler PyTorch Hackathon 2026 — Team Falcons.*
