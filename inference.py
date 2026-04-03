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
import signal

import requests as req
from openai import OpenAI

# ── MANDATORY: read from environment variables ───────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")
MAX_EPISODE_STEPS = 5   # Safety cap for multi-turn episodes

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
            temperature=0.1,
            max_tokens=600,
            timeout=30,
        )
        raw = completion.choices[0].message.content or ""
        return extract_json(raw)
    except Exception as e:
        print(f"  [LLM error: {e}]")
        return {}


def run_episode(client: OpenAI, task_id: str, case_index: int,
                server_url: str) -> tuple[float, list[float]]:
    """
    Run a complete episode — handles BOTH single-step and multi-turn tasks.

    For single-step tasks: reset → step → done (1 call)
    For deteriorating_patient: reset → step → step → step until done=True

    Returns (final_reward, all_step_rewards)
    """
    # Reset
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
        return 0.0, []

    obs = reset_data["observation"]
    patient_history = obs.get("patient_history", "")
    task_description = obs.get("task_description", "")

    step_rewards = []
    last_reward = 0.0

    # ── CRITICAL: loop until done=True (handles multi-turn episodes) ──
    step_count = 0
    done = reset_data.get("done", False)

    while not done and step_count < MAX_EPISODE_STEPS:
        # Get LLM action
        action_dict = call_llm(client, patient_history, task_description, task_id)

        if not action_dict:
            # Empty response — submit minimal to get partial feedback
            action_dict = {"priority": "medium", "action": "monitor"}

        # Submit step
        try:
            step_resp = req.post(
                f"{server_url}/step",
                json={"action": action_dict},
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
        step_count += 1

        # For multi-turn: the next patient_history is the updated observation
        patient_history = obs.get("patient_history", patient_history)

        # Show step-level feedback for multi-turn
        if task_id == "deteriorating_patient":
            feedback = obs.get("feedback", "")
            print(f"    step {step_count}: action={action_dict.get('action','?')!r:12} reward={reward:.3f}  {feedback[:50]}")

    return last_reward, step_rewards


def wait_for_server(url: str, retries: int = 30) -> bool:
    for i in range(retries):
        try:
            if req.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
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
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # ── Run 2 cases per task ─────────────────────────────────
    TASKS = [
        ("simple_triage",       "Easy",   [0, 1]),
        ("conflicting_vitals",  "Medium", [0, 1]),
        ("masked_deterioration","Hard",   [0, 1]),
        ("demographic_fairness","Medium", [0, 4]),   # Test 2 different demographics
        ("deteriorating_patient","Hard",  [0, 1]),   # MULTI-TURN
    ]

    all_results = {}
    total_cases = 0
    total_score = 0.0

    for task_id, difficulty, case_indices in TASKS:
        print(f"  [{difficulty}] {task_id}")
        task_scores = []

        for ci in case_indices:
            reward, steps = run_episode(client, task_id, ci, SERVER_URL)
            task_scores.append(reward)
            total_score += reward
            total_cases += 1

            if task_id != "deteriorating_patient":
                # Single-step tasks: print one line
                from server.cases import CASE_BANK
                try:
                    case_id = CASE_BANK[task_id][ci]["case_id"]
                except Exception:
                    case_id = f"case_{ci}"
                print(f"    {case_id}: {reward:.3f}")

        task_avg = sum(task_scores) / len(task_scores)
        all_results[task_id] = {"avg": task_avg, "scores": task_scores}
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
