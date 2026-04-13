"""
inference.py — Mandatory Baseline Inference Script
==================================================

MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier for inference
    HF_TOKEN       Hugging Face/API key

- Defaults are intentionally set only for API_BASE_URL and MODEL_NAME
- Script is named `inference.py` at repository root
- OpenAI client is used for all LLM calls

STDOUT FORMAT
The script emits exactly these line types:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<float> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<float> rewards=<r1,r2,...,rn>
Reward/score fields use 4 decimal places so values in (0,1) never round to 0.00 or 1.00.
"""

import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import requests as req
from openai import OpenAI

# Mandatory env vars:
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
# Optional when using from_docker_image() based workflows.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("TASK_NAME", "medical_triage")
BENCHMARK = os.getenv("BENCHMARK", "openenv_medical_triage")
MAX_STEPS = 5
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.5

# ── System prompt — clinical triage agent ───────────────────
SYSTEM_PROMPT = """You are an expert clinical triage nurse. You ALWAYS respond with a single valid JSON object.
No markdown, no prose, no backticks — ONLY valid JSON.

NEWS2 scoring (mandatory knowledge):
  Respiratory Rate: ≤8=3, 9-11=1, 12-20=0, 21-24=2, ≥25=3
  SpO2:             ≤91=3, 92-93=2, 94-95=1, ≥96=0
  Systolic BP:      ≤90=3, 91-100=2, 101-110=1, 111-219=0, ≥220=3
  Heart Rate:       ≤40=3, 41-50=1, 51-90=0, 91-110=1, 111-130=2, ≥131=3
  Temperature:      ≤35=3, 35.1-36=1, 36.1-38=0, 38.1-39=1, ≥39.1=2
  Consciousness:    Alert=0, any other=3

For simple_triage and demographic_fairness, respond with:
  {"priority":"low|medium|high|critical","news2_score":<int>,"critical_sign":"<param>","recommended_action":"emergency_response|urgent_review|routine_monitoring","confidence":<0-1>}

For conflicting_vitals, also add:
  {"misleading_signs":["<param>"],"condition":"<diagnosis>","rationale":"<reasoning>"}

For masked_deterioration, respond with:
  {"priority":"<level>","masking_drug_or_condition":"<drug>","masked_sign":"<vital>","critical_clues":["<clue>"],"condition":"<diagnosis>","recommended_action":"<action>","rationale":"<reasoning>","confidence":<0-1>}

For deteriorating_patient, respond ONLY with:
  {"action":"monitor|escalate|emergency_response","rationale":"<reasoning>","confidence":<0-1>}

For sepsis_bundle, respond with:
  {"priority":"high|critical","bundle_elements":["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],"antibiotic_choice":"<antibiotic>","fluid_volume_ml":<int>,"vasopressor_indicated":true|false,"rationale":"<MAP+lactate+allergy reasoning>","confidence":<0-1>}
  CRITICAL RULES for sepsis_bundle:
  - Check allergy history — NEVER give piperacillin_tazobactam/co-amoxiclav/amoxicillin if penicillin allergy. Use meropenem or levofloxacin instead.
  - fluid_volume_ml: standard 30ml/kg (~2000ml for 70kg patient); reduce to 500ml if severe AKI (creatinine >300 or oliguria documented)
  - vasopressor_indicated: true if MAP <65 despite fluids; false if MAP >=65
  - Always include blood_cultures and lactate_measurement in bundle_elements

For paediatric_triage, respond with:
  {"priority":"low|medium|high|critical","age_group":"infant|toddler|preschool|school_age|adolescent","pews_score":<int>,"critical_sign":"<vital_sign>","recommended_action":"emergency_response|urgent_review|routine_monitoring","confidence":<0-1>}
  CRITICAL RULES for paediatric_triage:
  - Use age-appropriate vital sign thresholds (NOT adult NEWS2): infants RR 30-60, HR 100-160; toddlers RR 24-40, HR 90-150; school_age RR 18-30, HR 70-110; adolescents RR 12-20, HR 60-100
  - SpO2 <92% in any child = HIGH or CRITICAL priority
  - Prolonged capillary refill (>2s) is a critical paediatric sign not in adult NEWS2
  - Parental concern ('not right') is a validated clinical indicator — take seriously

For medication_reconciliation, respond with:
  {"issues_found":["<issue_type>",...],"severity":"low|medium|high|critical","requires_pharmacist":true|false,"recommended_action":"safe_to_prescribe|modify_dose|withhold_drug|emergency_review","rationale":"<reasoning>","confidence":<0-1>}
  CRITICAL RULES for medication_reconciliation:
  - NSAIDs are CONTRAINDICATED in AKI (worsens renal perfusion) and dangerous with warfarin (3x GI bleed risk)
  - Methotrexate is ALWAYS weekly for non-oncology use — daily dosing = life-threatening error (NPSA 2006)
  - ACE inhibitor + potassium-sparing diuretic = hyperkalaemia risk, especially in CKD
  - Morphine accumulates in renal failure (eGFR <30) — use oxycodone at reduced dose instead
  - NEVER give penicillin-class antibiotics (amoxicillin, piperacillin) with documented penicillin anaphylaxis
"""


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find first {...} block
        m = re.search(r"\{[\s\S]+?\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = "null" if not error else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def call_llm(client: OpenAI, patient_history: str, task_description: str, task_id: str) -> dict:
    """Call LLM and return parsed JSON action."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"TASK: {task_id}\n\n"
                    f"INSTRUCTIONS:\n{task_description}\n\n"
                    f"PATIENT CASE:\n{patient_history}"
                )}
            ],
            temperature=TEMPERATURE,
            max_tokens=600,
            timeout=30,
        )
        raw = completion.choices[0].message.content or ""
        return extract_json(raw)
    except Exception:
        return {}


def run_episode(client: OpenAI, task_id: str, case_index: int, server_url: str) -> tuple[float, list[float], dict]:
    """
    Run a complete episode — handles BOTH single-step and multi-turn tasks.

    For single-step tasks: reset → step → done (1 call)
    For deteriorating_patient: reset → step → step → step until done=True

    Uses session_id from reset() to ensure step() targets the correct episode
    (required for correct behaviour when multiple agents run concurrently).

    Returns (final_reward, all_step_rewards, last_action_dict)
    """
    # Reset — always creates a fresh session
    try:
        reset_resp = req.post(
            f"{server_url}/reset",
            json={"task_id": task_id, "case_index": case_index, "seed": 42},
            timeout=10
        )
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception:
        return 0.0, [], {}

    # Capture session_id so step() hits the correct episode
    session_id = reset_data.get("info", {}).get("session_id")

    obs = reset_data["observation"]
    patient_history = obs.get("patient_history", "")
    task_description = obs.get("task_description", "")

    step_rewards = []
    last_reward = 0.0
    last_action_dict: dict = {}

    # ── CRITICAL: loop until done=True (handles multi-turn episodes) ──
    step_count = 0
    done = reset_data.get("done", False)

    while not done and step_count < MAX_STEPS:
        # Get LLM action
        action_dict = call_llm(client, patient_history, task_description, task_id)

        if not action_dict:
            # Empty response — submit minimal to get partial feedback
            action_dict = {"priority": "medium", "action": "monitor"}

        # Submit step — include session_id for correct episode routing
        step_payload = {"action": action_dict}
        if session_id:
            step_payload["session_id"] = session_id

        step_error = None
        try:
            step_resp = req.post(
                f"{server_url}/step",
                json=step_payload,
                timeout=15
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            step_error = str(exc)
            reward = 0.0
            done = True
            step_rewards.append(reward)
            last_reward = reward
            step_count += 1
            action_str = json.dumps(action_dict, separators=(",", ":"), ensure_ascii=True)
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=step_error)
            break

        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", True)
        obs = step_data["observation"]
        step_rewards.append(reward)
        last_reward = reward
        last_action_dict = action_dict   # track for fairness parity scoring
        step_count += 1
        patient_history = obs.get("patient_history", patient_history)
        action_str = json.dumps(action_dict, separators=(",", ":"), ensure_ascii=True)
        raw_error = obs.get("last_action_error")
        step_error = str(raw_error) if raw_error else None
        log_step(step=step_count, action=action_str, reward=reward, done=done, error=step_error)

    return last_reward, step_rewards, last_action_dict


def wait_for_server(url: str, retries: int = 30) -> bool:
    for i in range(retries):
        try:
            if req.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _get_case_id(server_url: str, task_id: str, case_index: int) -> str:
    """Fetch case_id from the /tasks API — avoids importing server internals."""
    try:
        resp = req.get(f"{server_url}/tasks", timeout=5)
        if resp.status_code == 200:
            task_data = resp.json().get(task_id, {})
            case_ids = task_data.get("case_ids", [])
            if 0 <= case_index < len(case_ids):
                return case_ids[case_index]
    except Exception:
        pass
    return f"case_{case_index}"


def _validate_required_env() -> list[str]:
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN (or OPENAI_API_KEY/API_KEY)")
    return missing


def main():
    missing = _validate_required_env()
    if missing:
        print("Missing required environment variables:")
        for name in missing:
            print(f"  - {name}")
        print("Set the required variables and re-run inference.py")
        sys.exit(1)

    # Start local server when needed.
    server_proc = None
    if "localhost" in SERVER_URL:
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not wait_for_server(SERVER_URL):
            sys.exit(1)

    # ── MANDATORY: Use OpenAI client ─────────────────────────
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run 2 cases per task to preserve baseline comparability.
    TASKS = [
        ("simple_triage",            "Easy",   [0, 1]),
        ("conflicting_vitals",       "Medium", [0, 1]),
        ("masked_deterioration",     "Hard",   [0, 1]),
        ("demographic_fairness",     "Medium", [0, 1]),   # FP001 white_male + black_male — same group, enables parity scoring
        ("deteriorating_patient",    "Hard",   [0, 1]),   # MULTI-TURN
        ("sepsis_bundle",            "Hard",   [0, 1]),   # SB001 (shock+vasopressors) + SB002 (no shock)
        ("paediatric_triage",        "Hard",   [0, 2]),   # PD001 infant bronchiolitis + PD003 paediatric DKA
        ("medication_reconciliation","Hard",   [0, 3]),   # MR001 warfarin-NSAID + MR004 methotrexate overdose
        ("icu_deterioration",        "Hard",   [0, 1]),   # IC001 septic shock worsening + IC002 post-op AKI stable
        ("sbar_handover",            "Medium", [0, 1]),   # SH001 critical post-op sepsis + SH002 routine pneumonia
        ("differential_diagnosis",   "Hard",   [0, 1]),   # DD001 STEMI + DD002 SAH
    ]

    for task_id, _difficulty, case_indices in TASKS:
        for ci in case_indices:
            case_id = _get_case_id(SERVER_URL, task_id, ci)
            episode_task = f"{task_id}:{case_id}"
            log_start(task=episode_task, env=BENCHMARK, model=MODEL_NAME)
            reward = 0.0
            step_rewards: list[float] = []
            success = False
            try:
                reward, step_rewards, action_dict = run_episode(client, task_id, ci, SERVER_URL)
                score = max(0.0, min(1.0, reward))
                success = score >= SUCCESS_SCORE_THRESHOLD
            except Exception:
                score = 0.0
                success = False
            finally:
                log_end(success=success, steps=len(step_rewards), score=score, rewards=step_rewards)

    # Cleanup
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
