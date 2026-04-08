# Educational Deep Dive: How This RL Medical Triage System Works

## Why this document exists

This guide is for learning. It explains this project from first principles to implementation details:
- what problem we are solving
- how reinforcement learning (RL) fits
- what each component does
- how all pieces interact
- how this can improve LLM performance over time

This document is intentionally detailed and educational.

---

## 1) The basic idea in plain language

Imagine a junior nurse or doctor reading patient information and deciding:
- how urgent the case is
- what should happen next

In real life, this is triage. In this project, we simulate triage in software so AI agents can practice and be measured safely.

The system gives an AI:
1. a patient case (history + vitals + context)
2. a chance to respond (priority, reasoning, action)
3. a score (reward) showing how good the response is
4. feedback and repeat opportunities over many cases

That cycle is the core RL loop: **observe -> act -> reward -> improve**.

---

## 2) What reinforcement learning means here

RL is usually explained with an "agent in an environment."

In this project:
- **Agent**: the model (or policy) deciding triage outputs.
- **Environment**: the medical triage simulator.
- **State/Observation**: patient case text and task context.
- **Action**: structured triage decision.
- **Reward**: score from deterministic clinical graders.

The goal is not "chat nicely." The goal is to maximize long-term reward by making correct clinical decisions consistently.

---

## 3) Why this is useful for LLMs

LLMs are strong in language, but triage needs:
- consistent structured outputs
- reliable prioritization under uncertainty
- safety-aware decisions
- fairness across demographics
- trend recognition over time (not just one-step answers)

This environment turns those expectations into measurable training signals.

Instead of vague "good answer / bad answer," the model gets granular feedback:
- correct priority?
- correct critical sign?
- correct action?
- recognized masked deterioration?
- calibrated confidence?

This dense reward improves learning efficiency compared to sparse pass/fail feedback.

---

## 4) The real-world triage problem modeled here

The environment includes five task families:

1. **Simple triage**
- clear vitals
- straightforward NEWS2-based urgency

2. **Conflicting vitals**
- some vitals look normal and can mislead
- model must identify the true risk signal

3. **Masked deterioration**
- medications/conditions hide classic warning signs
- model must infer severity from indirect clues

4. **Demographic fairness**
- same clinical profile across demographic variants
- model should not change urgency due to demographic attributes

5. **Deteriorating patient (multi-turn)**
- patient changes over time
- model should escalate early based on trends

These tasks represent escalating reasoning complexity and clinical realism.

---

## 5) Project architecture at a high level

Main runtime layers:

1. **Data layer**
- patient cases and ground truth definitions

2. **Grading layer**
- deterministic scoring functions with partial credit

3. **Environment layer**
- episode state machine (`reset`, `step`, `state`)

4. **API layer**
- FastAPI endpoints exposing OpenEnv-compatible interface

5. **Client/UI layer**
- browser interface and Python client for interaction

6. **Evaluation layer**
- baseline inference script and automated tests

---

## 6) Component-by-component breakdown

### A) `server/cases.py` - the case bank

Primary function:
- stores all patient scenarios and expected outcomes

Logic:
- each case has vitals, history, task id, expected priority, and ground truth details
- task-indexed case bank supports sampling by task and case id

RL contribution:
- defines the training distribution
- provides ground truth anchor for rewards
- enables targeted evaluation by difficulty domain

---

### B) `server/graders.py` - reward engine

Primary function:
- converts agent response into numeric score in `[0.0, 1.0]`

Logic:
- computes NEWS2 and related scoring dimensions
- gives partial credit across sub-dimensions
- includes specialized graders for fairness and multi-turn deterioration
- includes confidence calibration bonus and asymmetric risk penalty

RL contribution:
- this is the learning signal
- richer reward decomposition means faster, more stable policy improvements
- safety-critical mistakes (like severe under-triage) are penalized more strongly

---

### C) `server/medical_triage_environment.py` - environment state machine

Primary function:
- controls episode lifecycle

Logic:
- `reset()` chooses/loads a case and starts an episode
- `step()` applies grading, updates state, returns feedback
- handles single-step and multi-turn episodes
- tracks cumulative reward and tasks completed

RL contribution:
- defines transition dynamics and episode boundaries
- creates the sequential decision process RL needs

---

### D) `models.py` - typed interfaces

Primary function:
- defines structured contracts for actions, observations, and state

Logic:
- Pydantic models validate incoming/outgoing payloads
- catches malformed data early

RL contribution:
- stabilizes training interactions by enforcing consistent schema
- avoids hidden failures from malformed actions/responses

---

### E) `server/app.py` - API and web app host

Primary function:
- exposes environment over HTTP and serves the demo UI

Logic:
- OpenEnv endpoints: `/reset`, `/step`, `/state`, `/health`
- helper endpoints: `/tasks`, `/history`, `/stats`, `/suggest`, `/web`
- includes rule-based and model-based suggestion pathways for UI assistance

RL contribution:
- provides standard interface for rollouts/evaluations
- enables human inspection and debugging of policy behavior

---

### F) `client.py` - Python interaction wrapper

Primary function:
- convenient programmatic interface for environment calls

Logic:
- wraps reset/step/state HTTP calls
- returns typed results

RL contribution:
- easy integration into training loops and experiments

---

### G) `inference.py` - baseline evaluator

Primary function:
- runs a reference model through cases and prints benchmark scores

Logic:
- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- generates structured actions
- evaluates each task and summarizes scores

RL contribution:
- establishes baseline for comparison
- helps measure whether new training strategies actually improve results

---

### H) `train.py` - reward-conditioned RL training loop

Primary function:
- demonstrates that the environment supports active training, not just evaluation

Logic:
- runs the same fixed patient case per task across multiple repetitions
- injects past episode rewards, breakdowns, and hints into the LLM system prompt
- task-filtered feedback: agent only sees prior attempts on the same task type
- prints learning curve (first vs latest score) and trend per task

RL contribution:
- proves the dense reward signal enables measurable per-task improvement
- demonstrates reward-conditioned prompting as a practical training strategy
- validates that partial credit grading creates a meaningful learning gradient (not binary pass/fail)
- shows same-case repetition isolates learning signal from case-difficulty variation

---

### I) `tests/` - quality and regression safety

Primary function:
- ensure reliability, correctness, and contract stability across 119 tests

Types:
- `test_graders.py` (31): NEWS2 boundary, priority distance, task grader unit tests
- `test_environment.py` (24): environment lifecycle, reset/step/state, full episode flows
- `test_v2_enhancements.py` (44): fairness parity, asymmetric penalty, confidence calibration, multi-turn deterioration, adversarial robustness cases
- `test_api_contract.py` (9): API contract for all action fields (live server)
- `test_ui_contract.py` (8): UI DOM contract checks plus regression guards (live server)
- `test_inference_contract.py` (3): baseline inference reproducibility guards

RL contribution:
- protects training signal integrity (grader correctness = reward correctness)
- prevents silent reward or behavior regressions across code changes
- adversarial tests ensure environment handles malformed agent output gracefully

---

### J) `scripts/pre_submit_check.sh` - release gate

Primary function:
- one-command validation before submission/deployment

Logic:
- runs tests, docker build, health smoke check, and validator

RL contribution:
- ensures environment quality is reproducible, not accidental

---

## 7) The RL loop in this system (step-by-step)

**Evaluation loop (`inference.py`):**
1. `reset()` returns a patient scenario.
2. Agent outputs structured triage action.
3. `step()` grades action and returns reward + breakdown.
4. Repeat across all tasks for benchmark scoring.

**Training loop (`train.py`):**
1. `reset()` returns the same fixed patient case for this task.
2. Agent reads its own past attempts (score + breakdown + hint) injected into system prompt.
3. Agent outputs improved structured action.
4. `step()` grades action and returns reward + per-dimension breakdown.
5. Result is appended to task history for next repetition's system prompt.
6. Repeat for N repetitions — learning curve shows per-task improvement.

For multi-turn deterioration tasks, the loop includes time progression and early/late escalation reward effects.

---

## 8) How this improves LLM efficiency

Efficiency here means "more improvement per training/evaluation step."

This system improves efficiency through:

1. **Structured action space**
- model learns precise, machine-checkable outputs
- less token waste on irrelevant prose

2. **Dense reward decomposition**
- model sees where it was right/wrong
- learning signal is not all-or-nothing

3. **Difficulty curriculum**
- easy to hard tasks allow staged capability growth

4. **Safety-aware shaping**
- under-triage penalties steer behavior away from dangerous mistakes

5. **Confidence calibration**
- rewards not just correctness, but appropriately calibrated certainty

6. **Fairness constraints**
- discourages demographic sensitivity in clinically identical scenarios

7. **Multi-turn trend reasoning**
- teaches temporal decision policy, not only static classification

In short: the environment creates sharper and more actionable feedback than generic prompt tuning alone.

---

## 9) What this does NOT guarantee

Even with high scores, this does not automatically mean real-world clinical deployment readiness.

Why:
- simulation still abstracts reality
- true clinical validation requires retrospective studies, clinician review, shadow deployment, and governance

This project is best viewed as:
- a high-quality RL benchmark and training environment
- a strong foundation for further clinical validation work

---

## 10) Practical mental model for non-technical readers

You can think of this like a driving simulator:
- We are not putting an untested driver directly on real roads.
- We train and test in realistic scenarios first.
- We score every decision.
- We intentionally include hard and dangerous edge cases.

Here, the "driver" is the AI triage policy.
The "simulator" is this environment.
The "road test score" is the reward and test suite.

---

## 11) Summary

This project builds a complete, testable RL environment for medical triage:
- realistic tasks
- deterministic graders
- structured interfaces
- deployable API/UI
- robust validation pipeline

Its educational value is high because it demonstrates how to turn a complex real-world workflow into an RL-ready system with measurable progress signals.

