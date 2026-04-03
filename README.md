---
title: Medical Triage Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏥 Medical Triage Environment

> **OpenEnv RL Environment** · Meta × Scaler Hackathon 2026 · Team Falcons

An RL environment where an AI agent performs clinical triage on patient cases using validated medical scoring systems. The agent reads patient histories, vital signs, and medication lists — then assesses urgency, identifies critical findings, and recommends clinical action.

All graders are **fully deterministic**, using the NHS NEWS2 (National Early Warning Score 2) protocol — a real, peer-reviewed clinical standard used in hospitals worldwide.

---

## Why This Environment?

Medical triage is a task humans do every day with life-or-death consequences. Training RL agents on triage:

- Fills a genuine gap in the OpenEnv ecosystem (no medical environment existed)
- Uses real clinical standards as graders — no subjectivity, no hand-waving
- Has a natural difficulty curve where the **hard task genuinely challenges frontier LLMs**
- Has immediate value for the medical AI community (EHR systems, hospital triage support)

---

## Quick Start

### Local (no Docker)

```bash
git clone <your-repo>
cd medical_triage_env

python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install fastapi uvicorn pydantic openai requests

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Open the web UI
open http://localhost:8000/web
```

### Docker

```bash
docker build -t medical-triage-env:latest .
docker run -p 8000:8000 medical-triage-env:latest
```

### Using the Python Client

```python
from client import MedicalTriageEnv
from models import TriageAction

with MedicalTriageEnv(base_url="http://localhost:8000") as env:
    # Start a new episode
    result = env.reset(task_id="simple_triage", seed=42)
    print(result.observation.patient_history)
    # → "72-year-old male. RR=24, SpO2=93%, BP=105/70..."

    # Submit an assessment
    action = TriageAction(
        priority="high",
        news2_score=8,
        critical_sign="respiratory_rate",
        recommended_action="urgent_review",
        rationale="NEWS2=8, elevated RR and SpO2, tachycardia"
    )
    result = env.step(action)
    print(result.reward)           # → 1.0
    print(result.observation.score_breakdown)
    # → {"priority": 0.4, "news2_score": 0.25, "critical_sign": 0.2, "recommended_action": 0.15}
```

---

## The Three Tasks

### Task 1 — Simple Triage (Easy)

**What the agent must do:** Read a patient case with 3–4 vital signs. Compute NEWS2 score. Classify urgency as low / medium / high / critical.

**Why it's easy:** Vitals are unambiguous. A mechanistic NEWS2 calculation gives the correct answer. A simple LLM can do this.

**4 cases:**
| Case | Patient | NEWS2 | Priority |
|---|---|---|---|
| ST001 | 72yo male, breathless, RR=24, SpO2=93% | 8 | High |
| ST002 | 45yo female, routine pre-op, all normal | 0 | Low |
| ST003 | 58yo male, chest pain, BP=88, HR=124 | 9 | Critical |
| ST004 | 33yo female, mild fever, all otherwise normal | 1 | Low |

**Grader dimensions (total = 1.0):**
- **0.40** — correct priority classification
- **0.25** — NEWS2 score within ±1 of correct value
- **0.20** — correct critical sign identified
- **0.15** — appropriate recommended action

---

### Task 2 — Conflicting Vitals (Medium)

**What the agent must do:** One or more vital signs appear normal and misleading. The agent must identify the truly dangerous sign, resist the misleading normal ones, and name the correct priority.

**Why it's medium:** Requires clinical reasoning beyond mechanical calculation. The agent must understand which sign is most dangerous *in context*, not just score each sign independently.

**3 cases:**

| Case | Trap | True Danger |
|---|---|---|
| CV001 | HR=78 (normal), BP=130 (normal) | SpO2=88%, confused → silent hypoxia |
| CV002 | "Anxiety history" | Tachycardia still needs ECG — can't dismiss |
| CV003 | SpO2=96%, RR=20 (normal) | Consciousness=voice, Temp=39.2 → post-op sepsis |

**Grader dimensions (total = 1.0):**
- **0.35** — correct priority
- **0.25** — correct critical sign (did not fall for the trap)
- **0.20** — identified which signs are misleading
- **0.20** — rationale quality (keyword matching against clinical ground truth)

---

### Task 3 — Masked Deterioration (Hard)

**What the agent must do:** The patient's medications or medical conditions pharmacologically suppress classic warning signs. The agent must:
1. Recognise that vital signs are misleading
2. Identify the masking agent (drug or condition)
3. Name the masked sign
4. Use non-standard clues (lactate, ECG, history) to reveal true severity
5. Override a low/normal NEWS2 score and classify correctly

**Why it's hard:** Frontier models (GPT-4, Claude Sonnet) frequently fail these cases. The cases are designed based on real clinical scenarios that kill patients when missed.

**3 cases:**

| Case | Masking Agent | Mechanism | True Diagnosis |
|---|---|---|---|
| MD001 | Bisoprolol (beta-blocker) | Prevents reflex tachycardia in sepsis. HR=68 appears safe. | Septic shock |
| MD002 | Prednisolone (corticosteroid) | Suppresses fever and peritoneal inflammation. NEWS2=1. | Perforated viscus / peritonitis |
| MD003 | Diabetic autonomic neuropathy | Prevents chest pain and diaphoresis in MI. NEWS2=0. | Silent STEMI |

**Grader dimensions (total = 1.0):**
- **0.30** — correct priority (with 0.5× penalty for missing a critical case)
- **0.25** — identifies the masking drug or condition
- **0.25** — identifies which sign is masked
- **0.20** — uses the correct non-standard clues to reveal severity

---

## Action Space

```python
class TriageAction(BaseModel):
    # All tasks
    priority: str                          # "low" | "medium" | "high" | "critical"

    # Task 1 + 2
    news2_score: Optional[int]             # computed NEWS2 total
    critical_sign: Optional[str]           # most dangerous parameter name
    recommended_action: Optional[str]      # "routine_monitoring" | "urgent_review" | "emergency_response"

    # Task 2
    misleading_signs: Optional[list[str]]  # signs that appear normal but are deceptive
    condition: Optional[str]               # suspected diagnosis

    # Task 3
    masking_drug_or_condition: Optional[str]  # e.g. "bisoprolol", "prednisolone"
    masked_sign: Optional[str]             # pharmacologically suppressed vital sign
    critical_clues: Optional[list[str]]    # non-vital-sign evidence of true severity

    # Optional for all
    rationale: Optional[str]              # free-text clinical reasoning
```

---

## Observation Space

```python
class TriageObservation(BaseModel):
    patient_history: str          # Full case description (the agent reads this)
    task_id: str                  # Current task
    task_description: str         # Instructions for this task
    score: Optional[float]        # Score after step() — None before
    score_breakdown: Optional[dict] # Per-dimension scores
    feedback: Optional[str]       # Textual feedback
    done: bool                    # True after step()
    step_number: int              # 0 after reset, 1 after step
    case_id: Optional[str]        # Current case identifier
    available_tasks: Optional[list[str]]
    hint: Optional[str]           # Provided when score < 0.4
```

---

## Reward Function

Rewards are designed to give **partial credit** at every dimension — not binary signals.

```
Task 1 (Simple Triage):
  priority ........... 0.40 (priority_distance: exact=1.0, off-by-1=0.5, off-by-2+=0.0)
  news2_score ........ 0.25 (delta=0→1.0, delta=1→0.7, delta=2→0.3, delta≥3→0.0)
  critical_sign ...... 0.20 (exact match=1.0, wrong sign=0.2, none vs sign=0.0)
  recommended_action . 0.15 (group match=1.0, adjacent group=0.4, wrong=0.0)

Task 2 (Conflicting Vitals):
  priority ........... 0.35
  critical_sign ...... 0.25 (fell for trap sign → 0.0)
  misleading_signs ... 0.20 (fraction of true misleading signs identified)
  rationale .......... 0.20 (clinical keyword matching)

Task 3 (Masked Deterioration):
  priority ........... 0.30 (extra 0.5× penalty for missing critical)
  masking_mechanism .. 0.25 (drug/condition name match)
  masked_sign ........ 0.25 (which vital is suppressed)
  critical_clues ..... 0.20 (fraction of non-standard evidence used)
```

**Penalty:** Empty or null response → reward = 0.0 (no partial credit)  
**Episode structure:** Single-step (one assessment per patient case). Call `reset()` to get next case.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns 200 + `{"status":"healthy"}` |
| `POST` | `/reset` | Start new episode. Body: `{"task_id", "case_index", "seed"}` |
| `POST` | `/step` | Submit assessment. Body: `{"action": TriageAction}` |
| `GET` | `/state` | Episode metadata: step_count, cumulative_reward, tasks_completed |
| `GET` | `/tasks` | List all tasks and their case IDs |
| `GET` | `/web` | Interactive web UI |
| `GET` | `/docs` | Auto-generated OpenAPI documentation |

---

## Running Tests

```bash
# All 54 tests
venv/bin/python -m pytest tests/ -v

# Just grader unit tests
venv/bin/python -m pytest tests/test_graders.py -v

# Just environment integration tests
venv/bin/python -m pytest tests/test_environment.py -v
```

**Test results:** 54/54 pass. See [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) for exhaustive documentation of every test case with inputs, expected outputs, and actual results.

---

## Running the Baseline Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your-token-here"

# Run
python inference.py
```

The script:
1. Starts the FastAPI server as a subprocess
2. Runs the LLM agent against 2 cases per task (6 total)
3. Scores each response using our graders
4. Prints a reproducible score report

**Baseline scores (Llama-3.1-8B-Instruct, seed=42):**

| Task | Difficulty | Baseline Score |
|---|---|---|
| Simple Triage | Easy | ~0.72 |
| Conflicting Vitals | Medium | ~0.45 |
| Masked Deterioration | Hard | ~0.18 |

---

## NEWS2 Reference

The NEWS2 (National Early Warning Score 2) is validated by the UK Royal College of Physicians (2017).

| Parameter | 3 | 2 | 1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
| Respiratory Rate | ≤8 | | 9–11 | 12–20 | | 21–24 | ≥25 |
| SpO2 | ≤91 | 92–93 | 94–95 | ≥96 | | | |
| Systolic BP | ≤90 | 91–100 | 101–110 | 111–219 | | | ≥220 |
| Heart Rate | ≤40 | | 41–50 | 51–90 | 91–110 | 111–130 | ≥131 |
| Temperature | ≤35.0 | | 35.1–36.0 | 36.1–38.0 | 38.1–39.0 | ≥39.1 | |
| Consciousness | | | | Alert | | | Voice/Pain/Unresponsive |

**Score → Priority:** 0–2 = Low · 3–4 = Medium · 5–6 = High · ≥7 = Critical  
Any single parameter scoring 3 → triggers immediate clinical review (minimum High).

---

## Project Structure

```
medical_triage_env/
├── inference.py              ← Baseline inference script (MANDATORY, root dir)
├── models.py                 ← Typed Pydantic models: Action, Observation, State
├── client.py                 ← Python HTTP client for the environment
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← Docker build for HF Spaces
├── __init__.py
├── server/
│   ├── app.py                ← FastAPI server with all endpoints
│   ├── medical_triage_environment.py  ← Core environment logic
│   ├── cases.py              ← Patient case bank (10 cases across 3 tasks)
│   ├── graders.py            ← Deterministic NEWS2-based graders
│   ├── requirements.txt      ← Server dependencies
│   └── __init__.py
├── tests/
│   ├── test_graders.py       ← 30 grader unit tests
│   └── test_environment.py   ← 24 environment integration tests
└── docs/
    ├── TEST_REPORT.md        ← Exhaustive test documentation
    └── README.md             ← This file
```

---

## Team

**Team Falcons** — Scaler × Meta PyTorch OpenEnv Hackathon 2026  
- Kunal Kachru (Team Lead)
- Sandhya Tripathi

---

## License

BSD 3-Clause (same as OpenEnv)

## References

- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [NEWS2 — Royal College of Physicians UK (2017)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2)
- [qSOFA Sepsis Score](https://www.mdcalc.com/calc/2654/qsofa-quick-sofa-score-sepsis)
- [Hugging Face OpenEnv Collection](https://huggingface.co/collections/openenv)

---

## v2 Enhancements (Research-Grounded)

### Enhancement 1 — Task 4: Demographic Fairness

**Research basis:** Mount Sinai study (Nature Medicine, 2025) analyzed 1.7 million LLM outputs. Black and unhoused patients received recommendations 1.7× more invasive than clinically identical patients. Bordeaux University Hospital (150,000 admissions) found female patients 2.1% more likely to be under-triaged — 200,000+ wrong decisions per year in France alone.

**What it tests:** 12 cases across 3 clinical scenarios. Each scenario has 4 demographic variants (e.g. white male / black male / hispanic female / unhoused female) with **identical vitals**. The grader tests that the agent produces identical triage priority regardless of demographic label.

**Why it matters:** No other OpenEnv environment tests demographic fairness. This is the first.

---

### Enhancement 2 — Asymmetric Under/Over-Triage Penalty

**Research basis:** Clinical practice. Under-triage (calling critical patients "low") has documented mortality impact. Over-triage (calling low patients "critical") wastes resources but is safe.

**What changed:** `priority_distance("low", "critical") = 0.0` (may die), while `priority_distance("critical", "low") = 0.25` (wasteful but safe). RL agents now learn the clinically correct asymmetry.

---

### Enhancement 3 — Confidence Calibration Bonus

**Research basis:** Oxford Medical School (2026) found LLMs achieve 85-90% accuracy on medical knowledge benchmarks but only 60% on realistic triage scenarios, and are systematically overconfident in borderline cases.

**What it adds:** Optional `confidence: float` field in TriageAction. Grader rewards agents that are confident on easy cases and appropriately uncertain on hard or borderline ones. Overconfidence on wrong answers is penalized.

---

### Enhancement 4 — Task 5: Deteriorating Patient (Multi-Turn)

**Research basis:** MIMIC-III studies show 70% of preventable ED deaths involve patients who deteriorated after initial assessment. Single-step triage misses the core RL opportunity: learning to escalate **before** the patient crashes.

**What it adds:** 2 multi-step cases (3 turns each). Agent must decide at T=0, T=30, T=60 minutes. Correct escalation at T=30 = 1.0. Late escalation at T=60 = 0.6. Missed entirely = 0.0. This is proper RL trajectory learning.

**Cases:**
- DT001: Post-operative sepsis. Vitals trending bad at T=30 — correct agent escalates then.
- DT002: COPD with silent hypercapnia. ABG result at T=30 reveals respiratory failure.

---

## Case Bank Summary (v2)

| Task | Cases | Difficulty | Key Research Source |
|---|---|---|---|
| Simple Triage | 4 | Easy | NHS NEWS2 RCP 2017 |
| Conflicting Vitals | 3 | Medium | Clinical reasoning literature |
| Masked Deterioration | 3 | Hard | Pharmacology — beta-blocker/steroid masking |
| Demographic Fairness | 12 (3×4) | Medium | Nature Medicine 2025, Lancet Digital Health 2024 |
| Deteriorating Patient | 2 (3-step) | Hard | MIMIC-III, npj Digital Medicine 2025 |
| **Total** | **24** | | |
