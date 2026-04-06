"""
inference.py — Mandatory Baseline Inference Script
===================================================
SPEC REQUIREMENTS (must not change these):
  - Named exactly "inference.py" in the ROOT directory
  - Uses OpenAI Client (not raw requests)
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
  - Completes in < 20 minutes on 2 vCPU / 8 GB RAM
  - No GPU assumption
"""

import os
import sys
import json
import time
import re
import subprocess

import requests as req
from openai import OpenAI

# ── MANDATORY: read from environment variables ───────────────
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")
MAX_EPISODE_STEPS = 5   # Safety cap for multi-turn episodes
TEMPERATURE = 0.0

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


def call_llm(client: OpenAI, patient_history: str, task_description: str,
             task_id: str) -> dict:
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
    except Exception as e:
        print(f"  [LLM error: {e}]")
        return {}


def run_episode(client: OpenAI, task_id: str, case_index: int,
                server_url: str) -> tuple[float, list[float], dict]:
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
    except Exception as e:
        print(f"  [Reset error: {e}]")
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

    while not done and step_count < MAX_EPISODE_STEPS:
        # Get LLM action
        action_dict = call_llm(client, patient_history, task_description, task_id)

        if not action_dict:
            # Empty response — submit minimal to get partial feedback
            action_dict = {"priority": "medium", "action": "monitor"}

        # Submit step — include session_id for correct episode routing
        step_payload = {"action": action_dict}
        if session_id:
            step_payload["session_id"] = session_id

        try:
            step_resp = req.post(
                f"{server_url}/step",
                json=step_payload,
                timeout=15
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as e:
            print(f"  [Step error: {e}]")
            break

        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", True)
        obs = step_data["observation"]
        step_rewards.append(reward)
        last_reward = reward
        last_action_dict = action_dict   # track for fairness parity scoring
        step_count += 1

        # For multi-turn: the next patient_history is the updated observation
        patient_history = obs.get("patient_history", patient_history)

        # Show step-level feedback for multi-turn
        if task_id == "deteriorating_patient":
            feedback = obs.get("feedback", "")
            print(f"    step {step_count}: action={action_dict.get('action','?')!r:12} reward={reward:.3f}  {feedback[:50]}")

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


def _validate_required_env() -> list[str]:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
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

    print("=" * 60)
    print("  Medical Triage Environment v2.0 — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print("=" * 60)

    # ── Start server ────────────────────────────────────────
    server_proc = None
    if "localhost" in SERVER_URL:
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("  Starting environment server...", end=" ", flush=True)
        if not wait_for_server(SERVER_URL):
            print("FAILED — server did not start")
            return
        print("ready\n")

    # ── MANDATORY: Use OpenAI client ─────────────────────────
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ── Run 2 cases per task ─────────────────────────────────
    TASKS = [
        ("simple_triage",       "Easy",   [0, 1]),
        ("conflicting_vitals",  "Medium", [0, 1]),
        ("masked_deterioration","Hard",   [0, 1]),
        ("demographic_fairness","Medium", [0, 1]),   # FP001 white_male + black_male — same group, enables parity scoring
        ("deteriorating_patient","Hard",  [0, 1]),   # MULTI-TURN
    ]

    all_results = {}
    total_cases = 0
    total_score = 0.0

    # Collect fairness responses per group for multi-variant parity scoring
    # Maps group_prefix → {case_id: action_dict}
    fairness_responses: dict[str, dict] = {}

    for task_id, difficulty, case_indices in TASKS:
        print(f"  [{difficulty}] {task_id}")
        task_scores = []

        for ci in case_indices:
            reward, _, action_dict = run_episode(client, task_id, ci, SERVER_URL)
            task_scores.append(reward)
            total_score += reward
            total_cases += 1

            from server.cases import CASE_BANK
            try:
                case = CASE_BANK[task_id][ci]
                case_id = case["case_id"]
            except Exception:
                case_id = f"case_{ci}"
                case = {}

            # Collect action for fairness parity scoring
            if task_id == "demographic_fairness" and case and action_dict:
                # group_id is the first 5 chars of case_id (e.g. "FP001")
                group_id = case_id[:5]
                if group_id not in fairness_responses:
                    fairness_responses[group_id] = {}
                fairness_responses[group_id][case_id] = action_dict

            if task_id != "deteriorating_patient":
                print(f"    {case_id}: {reward:.3f}")

        task_avg = sum(task_scores) / len(task_scores)
        all_results[task_id] = {"avg": task_avg, "scores": task_scores}
        print()

    # ── Demographic fairness: multi-variant parity scoring ────────
    if fairness_responses:
        print("  [Fairness] Running multi-variant parity check...")
        parity_scores = []
        for group_id, responses in fairness_responses.items():
            if len(responses) < 2:
                continue   # Need ≥2 variants to measure parity
            try:
                pr = req.post(
                    f"{SERVER_URL}/grade-fairness",
                    json={"group_id": group_id, "responses": responses},
                    timeout=10
                )
                if pr.status_code == 200:
                    pd_data = pr.json()
                    parity_score = pd_data.get("score", 0.0)
                    parity_scores.append(parity_score)
                    print(f"    {group_id}: parity={parity_score:.3f}  breakdown={pd_data.get('breakdown', {})}")
            except Exception as e:
                print(f"    [{group_id} parity error: {e}]")
        if parity_scores:
            avg_parity = sum(parity_scores) / len(parity_scores)
            all_results["demographic_fairness"]["parity_avg"] = round(avg_parity, 3)
            print(f"    Average parity score: {avg_parity:.3f}")
        print()

    # ── Summary ──────────────────────────────────────────────
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for task_id, diff, _ in TASKS:
        res = all_results.get(task_id, {})
        avg = res.get("avg", 0.0)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {task_id:<30} {avg:.3f}  {bar}")

    overall = total_score / total_cases if total_cases else 0
    print(f"\n  Overall average:               {overall:.3f}")
    print(f"  Cases evaluated:               {total_cases}")
    print("=" * 60)

    # ── Cleanup ──────────────────────────────────────────────
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
