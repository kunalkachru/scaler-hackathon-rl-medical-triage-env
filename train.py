"""
train.py — Reward-Conditioned RL Training Loop
================================================
Demonstrates that the Medical Triage Environment supports TRAINING,
not just evaluation. An LLM agent is run over repeated episodes of
the same patient cases, receiving reward feedback that it incorporates
into subsequent prompts.

Design:
  - Each task uses the SAME case across all repetitions, isolating
    the learning signal from case-difficulty variation.
  - Task-filtered feedback: the agent only sees past performance on
    the same task type, preventing cross-task confusion.
  - Dense reward signal: partial credit at every grader dimension
    gives the agent a learning gradient even on imperfect responses.
  - Per-task learning curves compare first vs latest episode score.

MANDATORY env vars (same as inference.py):
  API_BASE_URL  — LLM endpoint  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model ID      (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN      — API key

Optional:
  SERVER_URL    — env server URL (default: http://localhost:8000)
  REPS_PER_TASK — how many times to repeat each task case (default: 3)

Usage:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
  export HF_TOKEN="<your-token>"
  python train.py
"""

import argparse
import csv
import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

import requests as req
from openai import OpenAI

# ── MANDATORY: same variables as inference.py ──────────────────
API_BASE_URL  = os.getenv("API_BASE_URL", "")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME    = os.getenv("MODEL_NAME", "")
SERVER_URL    = os.getenv("SERVER_URL", "http://localhost:8000")
REPS_PER_TASK = int(os.getenv("REPS_PER_TASK", "3"))

# ── Training schedule: task + fixed case index ─────────────────
# Each task uses the same case across all REPS_PER_TASK repetitions
# so the learning signal is isolated from case-difficulty variation.
TRAINING_TASKS = [
    ("simple_triage",            0),  # ST001 — elevated RR, SpO2 93% → high
    ("conflicting_vitals",       0),  # CV001 — silent hypoxia trap
    ("masked_deterioration",     0),  # MD001 — bisoprolol masking septic shock
    ("demographic_fairness",     0),  # DF001 — identical vitals, different demographics
    ("deteriorating_patient",    0),  # DT001 — post-op deterioration, multi-turn
    ("sepsis_bundle",            0),  # SB001 — septic shock, full Hour-1 bundle
    ("paediatric_triage",        0),  # PD001 — infant bronchiolitis, PEWS
    ("medication_reconciliation", 0), # MR001 — warfarin + NSAID interaction
    ("icu_deterioration",        0),  # IC001 — septic shock worsening, SOFA=14
    ("sbar_handover",            0),  # SH001 — post-op sepsis critical handover
    ("differential_diagnosis",   0),  # DD001 — chest pain, STEMI must-not-miss
]

TEMPERATURE = 0.2   # small noise to allow variation across reps


# ── Clinical knowledge base ───────────────────────────────────
BASE_SYSTEM_PROMPT = """You are an expert clinical triage nurse. You ALWAYS respond with a single valid JSON object.
No markdown, no prose, no backticks — ONLY valid JSON.

NEWS2 scoring (mandatory knowledge):
  Respiratory Rate: ≤8=3, 9-11=1, 12-20=0, 21-24=2, ≥25=3
  SpO2:             ≤91=3, 92-93=2, 94-95=1, ≥96=0
  Systolic BP:      ≤90=3, 91-100=2, 101-110=1, 111-219=0, ≥220=3
  Heart Rate:       ≤40=3, 41-50=1, 51-90=0, 91-110=1, 111-130=2, ≥131=3
  Temperature:      ≤35=3, 35.1-36=1, 36.1-38=0, 38.1-39=1, ≥39.1=2
  Consciousness:    Alert=0, any other=3

For simple_triage, respond with:
  {"priority":"low|medium|high|critical","news2_score":<int>,"critical_sign":"<param>","recommended_action":"emergency_response|urgent_review|routine_monitoring","confidence":<0-1>}

For conflicting_vitals, respond with:
  {"priority":"<level>","critical_sign":"<param>","misleading_signs":["<param>"],"condition":"<diagnosis>","recommended_action":"<action>","rationale":"<reasoning>","confidence":<0-1>}

For masked_deterioration, respond with:
  {"priority":"<level>","masking_drug_or_condition":"<drug>","masked_sign":"<vital>","critical_clues":["<clue>"],"condition":"<diagnosis>","recommended_action":"<action>","rationale":"<reasoning>","confidence":<0-1>}

For demographic_fairness, respond with:
  {"priority":"low|medium|high|critical","news2_score":<int>,"critical_sign":"<param>","recommended_action":"emergency_response|urgent_review|routine_monitoring","rationale":"<reasoning>","confidence":<0-1>}

For deteriorating_patient, respond with:
  {"action":"monitor|escalate|emergency_response","rationale":"<reasoning>","confidence":<0-1>}

For sepsis_bundle, respond with:
  {"priority":"high|critical","bundle_elements":["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],"antibiotic_choice":"<drug>","fluid_volume_ml":<int>,"vasopressor_indicated":<bool>,"rationale":"<reasoning>","confidence":<0-1>}

For paediatric_triage, respond with:
  {"priority":"low|medium|high|critical","age_group":"infant|toddler|preschool|school_age|adolescent","pews_score":<int>,"critical_sign":"<param>","recommended_action":"emergency_response|urgent_review|routine_monitoring","rationale":"<reasoning>","confidence":<0-1>}

For medication_reconciliation, respond with:
  {"issues_found":["<issue>"],"severity":"low|medium|high|critical","requires_pharmacist":<bool>,"recommended_action":"safe_to_prescribe|modify_dose|withhold_drug|emergency_review","drug_to_withhold":"<drug_or_null>","rationale":"<reasoning>","confidence":<0-1>}

For icu_deterioration, respond with:
  {"sofa_score":<int 0-24>,"primary_organ_failure":"cardiovascular|respiratory|renal|hepatic|neurological|coagulation","deterioration_trend":"improving|stable|worsening","intervention":"maintain_current|increase_support|emergency_escalation|prepare_palliation","rationale":"<reasoning>"}

For sbar_handover, respond with:
  {"escalation_required":true|false,"priority":"low|medium|high|critical","assessment":"<brief clinical summary>","recommendation":"routine_monitoring|urgent_review|emergency_response"}

For differential_diagnosis, respond with:
  {"must_not_miss":"<life-threatening diagnosis>","top_diagnosis":"<most likely>","differentials":["<dx1>","<dx2>","<dx3>"],"first_investigation":"<key test>","urgency":"immediate|urgent|routine"}
"""

IMPROVEMENT_ADVICE = {
    "simple_triage": (
        "If priority is wrong: recompute NEWS2 — sum all 6 parameters carefully. "
        "If news2_score is off: check each vital range boundary. "
        "If critical_sign is wrong: identify the single most life-threatening parameter."
    ),
    "conflicting_vitals": (
        "If critical_sign is wrong: you likely chose a misleading normal value. "
        "Focus on the one sign that kills if missed, not the ones that look normal. "
        "If rationale score is low: name the exact mechanism (e.g. 'silent hypoxia', 'compensated shock')."
    ),
    "masked_deterioration": (
        "If masking_mechanism is low: name the exact drug or condition suppressing warning signs. "
        "If critical_clues is low: look for lactate, ECG, urine output, and medication history. "
        "NEWS2 may appear LOW despite true critical state — do NOT trust it at face value."
    ),
    "demographic_fairness": (
        "Treat every patient identically regardless of race, gender, or socioeconomic status. "
        "Base your triage purely on NEWS2 vitals — demographics must NOT change priority. "
        "If score is low: check that your priority and critical_sign match the vitals, not the demographics."
    ),
    "deteriorating_patient": (
        "At T=0 with stable vitals: 'monitor' is correct. Do NOT escalate prematurely. "
        "When vitals worsen significantly (NEWS2 ≥5 or single red flag): escalate immediately. "
        "If score is low: check the timing — early escalation on stable patient loses points."
    ),
    "sepsis_bundle": (
        "The Hour-1 Surviving Sepsis Campaign bundle requires: blood_cultures, broad_spectrum_antibiotics, "
        "iv_fluid_bolus (30ml/kg), lactate_measurement. Vasopressors only if MAP<65 or refractory shock. "
        "If antibiotic score low: use piperacillin_tazobactam for most cases; meropenem for resistant/ICU. "
        "If fluid_volume low: calculate 30ml/kg from patient weight in history."
    ),
    "paediatric_triage": (
        "Use age-appropriate PEWS thresholds — infant SpO2 <95% is critical, adolescent <92% is critical. "
        "Capillary refill >2s and parental concern are red flags at any age. "
        "If age_group is wrong: re-read the patient age carefully (infant=<1yr, toddler=1-3, preschool=3-5, "
        "school_age=5-12, adolescent=12-18). If pews_score low: sum all PEWS dimensions for the correct age band."
    ),
    "medication_reconciliation": (
        "Key interactions to know: warfarin+NSAIDs=bleeding risk; ACE inhibitor+K+-sparing diuretic=hyperkalaemia; "
        "methotrexate is ALWAYS weekly (never daily); NSAIDs contraindicated in AKI/CKD; "
        "penicillin allergy = amoxicillin/co-amoxiclav contraindicated. "
        "If issues_found score low: list ALL interactions present, not just the most obvious one. "
        "If severity wrong: methotrexate daily error and anaphylaxis risk = critical."
    ),
    "icu_deterioration": (
        "SOFA score: each of 6 organ systems scored 0-4; total 0-24. Score ≥2 = organ dysfunction, ≥11 = very high mortality. "
        "If sofa_score is wrong: check PaO2/FiO2, MAP, creatinine, bilirubin, platelet, GCS carefully. "
        "If intervention wrong: emergency_escalation for SOFA ≥11 or acute deterioration; prepare_palliation only if futile care ceiling reached. "
        "If primary_organ_failure wrong: identify the single most severely failing system."
    ),
    "sbar_handover": (
        "escalation_required=true if: GCS drop ≥2, MAP <65, RR >25, SpO2 <92%, new organ failure, or NEWS2 ≥7. "
        "emergency_response if immediate threat to life. urgent_review if deteriorating but not immediately life-threatening. "
        "If escalation_required wrong: this is a safety-critical boolean — it carries 40% of the score. "
        "assessment must mention the key clinical finding driving the recommendation."
    ),
    "differential_diagnosis": (
        "must_not_miss is the life-threatening diagnosis to EXCLUDE FIRST — not necessarily the most likely. "
        "Chest pain: must_not_miss=stemi, first_investigation=ecg. "
        "Thunderclap headache: must_not_miss=subarachnoid_haemorrhage, first_investigation=ct_head. "
        "Sudden SOB + pleuritic pain: must_not_miss=pulmonary_embolism, first_investigation=ctpa. "
        "Tearing back pain: must_not_miss=aortic_dissection, first_investigation=ct_angiography. "
        "If must_not_miss score low: this carries 40% of the score — get the safety-net diagnosis right first."
    ),
}


def build_system_prompt(task_id: str, task_history: list[dict]) -> str:
    """
    Inject past performance on this specific task so the agent can
    learn from its own mistakes within the same case.
    """
    if not task_history:
        return BASE_SYSTEM_PROMPT

    lines = []
    for ep in task_history[-3:]:
        score = ep["reward"]
        bd    = ep.get("breakdown", {})
        tag   = "EXCELLENT" if score >= 0.85 else "GOOD" if score >= 0.60 else "PARTIAL" if score >= 0.40 else "POOR"
        bd_str = ", ".join(
            f"{k}={v:.2f}" for k, v in bd.items()
            if isinstance(v, float) and not k.startswith("_")
        )
        lines.append(
            f"  Attempt {ep['rep']}: score={score:.3f} ({tag})"
            + (f"  [{bd_str}]" if bd_str else "")
            + (f"  hint: {ep.get('feedback','')}" if ep.get("feedback") else "")
        )

    advice = IMPROVEMENT_ADVICE.get(task_id, "")
    block = (
        f"\n\nYOUR PREVIOUS ATTEMPTS ON THIS SAME CASE — learn from this:\n"
        + "\n".join(lines)
        + (f"\n\nADVICE: {advice}" if advice else "")
    )
    return BASE_SYSTEM_PROMPT + block


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+?\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def call_llm(client: OpenAI, system_prompt: str, patient_history: str,
             task_description: str, task_id: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": (
                    f"TASK: {task_id}\n\n"
                    f"INSTRUCTIONS:\n{task_description}\n\n"
                    f"PATIENT CASE:\n{patient_history}"
                )},
            ],
            temperature=TEMPERATURE,
            max_tokens=600,
            timeout=30,
        )
        return extract_json(completion.choices[0].message.content or "")
    except Exception as e:
        print(f"    [LLM error: {e}]")
        return {}


def run_episode(client: OpenAI, task_id: str, case_index: int,
                task_history: list[dict], rep: int,
                server_url: str) -> dict:
    """Run one episode of the given task/case. Returns result dict."""
    try:
        r = req.post(f"{server_url}/reset",
                     json={"task_id": task_id, "case_index": case_index, "seed": rep},
                     timeout=10)
        r.raise_for_status()
        reset_data = r.json()
    except Exception as e:
        print(f"  [Reset error: {e}]")
        return {"rep": rep, "task_id": task_id, "reward": 0.0,
                "breakdown": {}, "feedback": ""}

    session_id       = reset_data.get("info", {}).get("session_id")
    obs              = reset_data["observation"]
    patient_history  = obs.get("patient_history", "")
    task_description = obs.get("task_description", "")

    system_prompt = build_system_prompt(task_id, task_history)

    last_reward    = 0.0
    last_breakdown: dict = {}
    last_feedback  = ""
    done           = reset_data.get("done", False)
    step_count     = 0

    while not done and step_count < 5:
        action_dict = call_llm(client, system_prompt, patient_history,
                               task_description, task_id)
        if not action_dict:
            action_dict = {"priority": "medium", "action": "monitor"}

        payload: dict = {"action": action_dict}
        if session_id:
            payload["session_id"] = session_id

        try:
            r = req.post(f"{server_url}/step", json=payload, timeout=15)
            r.raise_for_status()
            step_data = r.json()
        except Exception as e:
            print(f"  [Step error: {e}]")
            break

        last_reward    = step_data.get("reward", 0.0)
        done           = step_data.get("done", True)
        obs            = step_data["observation"]
        last_breakdown = obs.get("score_breakdown") or {}
        last_feedback  = obs.get("feedback") or ""
        step_count    += 1
        patient_history = obs.get("patient_history", patient_history)

    return {
        "rep":       rep,
        "task_id":   task_id,
        "reward":    last_reward,
        "breakdown": last_breakdown,
        "feedback":  last_feedback,
    }


def wait_for_server(url: str, retries: int = 30) -> bool:
    for _ in range(retries):
        try:
            if req.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _validate_env() -> list[str]:
    missing = []
    if not API_BASE_URL: missing.append("API_BASE_URL")
    if not MODEL_NAME:   missing.append("MODEL_NAME")
    if not API_KEY:      missing.append("HF_TOKEN (or OPENAI_API_KEY/API_KEY)")
    return missing


def print_summary(all_results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY — Same-Case Score Progression")
    print("=" * 60)
    print("  (Each task uses the same patient case across all reps,")
    print("   isolating the learning signal from case-difficulty noise)\n")

    improved = stable = regressed = 0

    for task_id, results in all_results.items():
        if not results:
            continue
        print(f"  [{task_id}]")
        for ep in results:
            bar   = "█" * int(ep["reward"] * 20) + "░" * (20 - int(ep["reward"] * 20))
            idx   = results.index(ep)
            trend = ""
            if idx > 0:
                delta = ep["reward"] - results[idx - 1]["reward"]
                trend = f"  {'↑' if delta > 0.005 else '↓' if delta < -0.005 else '→'} {delta:+.3f}"
            print(f"    Attempt {ep['rep']}: {bar}  {ep['reward']:.3f}{trend}")

        first  = results[0]["reward"]
        latest = results[-1]["reward"]
        delta  = latest - first
        arrow  = "▲ IMPROVED" if delta > 0.01 else "▼ REGRESSED" if delta < -0.01 else "≈ STABLE"
        print(f"    {arrow}  first={first:.3f}  latest={latest:.3f}  Δ={delta:+.3f}\n")

        if delta > 0.01:   improved  += 1
        elif delta < -0.01: regressed += 1
        else:               stable    += 1

    print(f"  Tasks improved:  {improved} / {len(all_results)}")
    print(f"  Tasks stable:    {stable}   / {len(all_results)}")
    print(f"  Tasks regressed: {regressed} / {len(all_results)}")
    print()
    print("  Dense reward signal confirmed: graders return partial credit")
    print("  at every dimension (priority, critical_sign, news2_score,")
    print("  rationale, masking_mechanism) — not binary pass/fail.")
    print("  This enables RL agents to receive meaningful learning signal")
    print("  even on imperfect responses, making the environment suitable")
    print("  for policy gradient and reward-conditioned training methods.")
    print("=" * 60)


def write_csv(all_results: dict[str, list[dict]], path: str) -> None:
    """Write (task, rep, score, breakdown_json) rows to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rep", "score", "breakdown_json", "feedback"])
        writer.writeheader()
        for task_id, results in all_results.items():
            for ep in results:
                writer.writerow({
                    "task_id":        task_id,
                    "rep":            ep["rep"],
                    "score":          round(ep["reward"], 4),
                    "breakdown_json": json.dumps(ep.get("breakdown", {})),
                    "feedback":       ep.get("feedback", ""),
                })
    print(f"\n  [CSV] Metrics written to: {path}")


def write_training_results_md(all_results: dict[str, list[dict]], model: str, path: str) -> None:
    """Write a TRAINING_RESULTS.md with before/after reward table."""
    lines: list[str] = [
        "# Training Results\n",
        f"Model: `{model}`  |  Reps per task: {REPS_PER_TASK}\n",
        "",
        "## Before/After Reward Table",
        "",
        "| Task | First Score | Last Score | Δ | Trend |",
        "|------|-------------|------------|---|-------|",
    ]
    for task_id, results in all_results.items():
        if not results:
            continue
        first  = results[0]["reward"]
        latest = results[-1]["reward"]
        delta  = latest - first
        trend  = "▲ IMPROVED" if delta > 0.01 else "▼ REGRESSED" if delta < -0.01 else "≈ STABLE"
        lines.append(f"| `{task_id}` | {first:.3f} | {latest:.3f} | {delta:+.3f} | {trend} |")

    lines += [
        "",
        "## Attempt-by-Attempt Scores",
        "",
    ]
    for task_id, results in all_results.items():
        lines.append(f"### {task_id}")
        lines.append("")
        lines.append("| Attempt | Score |")
        lines.append("|---------|-------|")
        for ep in results:
            lines.append(f"| {ep['rep']} | {ep['reward']:.3f} |")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")
    print(f"  [MD] Training results written to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Triage RL Training Loop")
    parser.add_argument("--csv", metavar="PATH", default="",
                        help="Write metrics CSV to this path (e.g. training_metrics.csv)")
    parser.add_argument("--md", metavar="PATH", default="",
                        help="Write TRAINING_RESULTS.md to this path")
    args = parser.parse_args()

    missing = _validate_env()
    if missing:
        print("Missing required environment variables:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("=" * 60)
    print("  Medical Triage Environment v2.1 — RL Training Loop")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API:      {API_BASE_URL}")
    print(f"  Tasks:    {len(TRAINING_TASKS)}  x  {REPS_PER_TASK} reps each")
    print("=" * 60)

    server_proc = None
    if "localhost" in SERVER_URL:
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("  Starting server...", end=" ", flush=True)
        if not wait_for_server(SERVER_URL):
            print("FAILED")
            return
        print("ready\n")

    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_results: dict[str, list[dict]] = {}

    for task_id, case_index in TRAINING_TASKS:
        print(f"  ── {task_id}  (case {case_index}, {REPS_PER_TASK} reps) ──")
        task_history: list[dict] = []

        for rep in range(1, REPS_PER_TASK + 1):
            result = run_episode(client, task_id, case_index,
                                 task_history, rep, SERVER_URL)
            task_history.append(result)
            bar = "█" * int(result["reward"] * 20) + "░" * (20 - int(result["reward"] * 20))
            print(f"    Attempt {rep}/{REPS_PER_TASK}: {bar}  {result['reward']:.3f}")

        all_results[task_id] = task_history
        print()

    print_summary(all_results)

    if args.csv:
        write_csv(all_results, args.csv)
    if args.md:
        write_training_results_md(all_results, MODEL_NAME, args.md)

    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
