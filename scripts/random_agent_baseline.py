"""
random_agent_baseline.py — Random Agent Lower Bound
====================================================
Sends structurally valid but randomly chosen responses to all 11 tasks.
Provides the theoretical lower bound for the leaderboard comparison.

Usage:
  python scripts/random_agent_baseline.py
  python scripts/random_agent_baseline.py --server-url https://kunalkachru23-medical-triage-env.hf.space
  python scripts/random_agent_baseline.py --runs 3  # average over 3 runs
"""

import argparse
import json
import random
import sys
import time
import uuid

import requests

# ── Random action generators ──────────────────────────────────────────────────

PRIORITIES      = ["low", "medium", "high", "critical"]
VITAL_SIGNS     = ["respiratory_rate", "heart_rate", "spo2", "systolic_bp", "temperature", "consciousness"]
ACTIONS_DT      = ["monitor", "escalate", "emergency_response", "comfort_care"]
BUNDLE_ELEMENTS = ["blood_cultures", "iv_antibiotics", "iv_fluids", "lactate", "vasopressors"]
ANTIBIOTICS     = ["piperacillin_tazobactam", "meropenem", "ceftriaxone", "co-amoxiclav", "levofloxacin"]
AGE_GROUPS      = ["infant", "toddler", "preschool", "school_age", "adolescent"]
ACTIONS_TRIAGE  = ["emergency_response", "urgent_review", "routine_monitoring"]
SEVERITIES      = ["low", "moderate", "high", "critical"]
ACTIONS_MR      = ["safe_to_prescribe", "modify_dose", "withhold_drug", "emergency_review"]
MR_ISSUES       = ["drug_interaction", "wrong_dose", "contraindication", "allergy_risk", "duplicate_therapy"]


def random_simple_triage() -> dict:
    return {
        "priority": random.choice(PRIORITIES),
        "news2_score": random.randint(0, 20),
        "critical_sign": random.choice(VITAL_SIGNS),
        "recommended_action": random.choice(ACTIONS_TRIAGE),
        "rationale": "random baseline",
        "confidence": round(random.uniform(0.1, 0.9), 2),
    }


def random_deteriorating_patient() -> dict:
    return {
        "action": random.choice(ACTIONS_DT),
        "rationale": "random baseline",
        "confidence": round(random.uniform(0.1, 0.9), 2),
    }


def random_sepsis_bundle() -> dict:
    n = random.randint(1, len(BUNDLE_ELEMENTS))
    return {
        "priority": random.choice(["high", "critical"]),
        "bundle_elements": random.sample(BUNDLE_ELEMENTS, n),
        "antibiotic_choice": random.choice(ANTIBIOTICS),
        "fluid_volume_ml": random.choice([500, 1000, 1500, 2000, 2500, 3000]),
        "vasopressor_indicated": random.choice([True, False]),
        "rationale": "random baseline",
        "confidence": round(random.uniform(0.1, 0.9), 2),
    }


def random_paediatric_triage() -> dict:
    return {
        "priority": random.choice(PRIORITIES),
        "age_group": random.choice(AGE_GROUPS),
        "pews_score": random.randint(0, 13),
        "critical_sign": random.choice(VITAL_SIGNS),
        "recommended_action": random.choice(ACTIONS_TRIAGE),
        "rationale": "random baseline",
        "confidence": round(random.uniform(0.1, 0.9), 2),
    }


def random_medication_reconciliation() -> dict:
    n = random.randint(1, 3)
    drugs = ["warfarin", "ibuprofen", "methotrexate", "lisinopril", "spironolactone", "morphine"]
    return {
        "issues_found": random.sample(MR_ISSUES, n),
        "severity": random.choice(SEVERITIES),
        "requires_pharmacist": random.choice([True, False]),
        "recommended_action": random.choice(ACTIONS_MR),
        "drug_to_withhold": random.choice(drugs),
        "rationale": "random baseline",
        "confidence": round(random.uniform(0.1, 0.9), 2),
    }


ORGAN_FAILURES   = ["respiratory", "cardiovascular", "renal", "hepatic", "neurological", "coagulation"]
ICU_INTERVENTIONS = ["emergency_escalation", "increase_support", "maintain_current", "prepare_palliation"]
TRENDS           = ["worsening", "stable", "improving"]
SBAR_RECS        = ["emergency_response", "urgent_review", "routine_monitoring"]
DIAGNOSES        = ["stemi", "subarachnoid_haemorrhage", "abdominal_aortic_aneurysm", "hypoglycaemia",
                    "pulmonary_embolism", "aortic_dissection", "meningitis", "stroke", "pericarditis"]
INVESTIGATIONS   = ["ecg", "ct_head", "ct_angiography", "blood_glucose", "troponin", "d_dimer", "ct_chest", "lp"]


def random_icu_deterioration() -> dict:
    return {
        "sofa_score": random.randint(0, 24),
        "primary_organ_failure": random.choice(ORGAN_FAILURES),
        "deterioration_trend": random.choice(TRENDS),
        "intervention": random.choice(ICU_INTERVENTIONS),
        "rationale": "random baseline",
    }


def random_sbar_handover() -> dict:
    return {
        "escalation_required": random.choice([True, False]),
        "priority": random.choice(PRIORITIES),
        "assessment": "random baseline assessment",
        "recommendation": random.choice(SBAR_RECS),
    }


def random_differential_diagnosis() -> dict:
    return {
        "must_not_miss": random.choice(DIAGNOSES),
        "top_diagnosis": random.choice(DIAGNOSES),
        "differentials": random.sample(DIAGNOSES, k=3),
        "first_investigation": random.choice(INVESTIGATIONS),
        "urgency": random.choice(["immediate", "urgent", "routine"]),
    }


TASK_ACTION_FN = {
    "simple_triage":            random_simple_triage,
    "conflicting_vitals":       random_simple_triage,
    "masked_deterioration":     random_simple_triage,
    "demographic_fairness":     random_simple_triage,
    "deteriorating_patient":    random_deteriorating_patient,
    "sepsis_bundle":            random_sepsis_bundle,
    "paediatric_triage":        random_paediatric_triage,
    "medication_reconciliation": random_medication_reconciliation,
    "icu_deterioration":        random_icu_deterioration,
    "sbar_handover":            random_sbar_handover,
    "differential_diagnosis":   random_differential_diagnosis,
}

TASKS = [
    ("simple_triage",             "Easy",   [0, 1]),
    ("conflicting_vitals",        "Medium", [0, 1]),
    ("masked_deterioration",      "Hard",   [0, 1]),
    ("demographic_fairness",      "Medium", [0, 1]),
    ("deteriorating_patient",     "Hard",   [0, 1]),
    ("sepsis_bundle",             "Hard",   [0, 1]),
    ("paediatric_triage",         "Hard",   [0, 2]),
    ("medication_reconciliation", "Hard",   [0, 3]),
    ("icu_deterioration",         "Hard",   [0, 1]),
    ("sbar_handover",             "Medium", [0, 1]),
    ("differential_diagnosis",    "Hard",   [0, 1]),
]

MAX_STEPS = 5


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(server_url: str, task_id: str, case_index: int) -> tuple[float, list[float]]:
    """Run one episode with a random agent. Returns (final_reward, all_step_rewards)."""
    try:
        sid = str(uuid.uuid4())
        r = requests.post(
            f"{server_url}/reset",
            json={"task_id": task_id, "case_index": case_index, "session_id": sid, "seed": 42},
            timeout=15,
        )
        r.raise_for_status()
        reset_data = r.json()
    except Exception as exc:
        print(f"    [reset error] {task_id}[{case_index}]: {exc}", file=sys.stderr)
        return 0.0, []

    # Use session_id from response if present, else our own
    session_id = reset_data.get("info", {}).get("session_id") or sid
    done = reset_data.get("done", False)
    step_rewards: list[float] = []
    last_reward = 0.0
    action_fn = TASK_ACTION_FN.get(task_id, random_simple_triage)

    step = 0
    while not done and step < MAX_STEPS:
        action = action_fn()
        try:
            sr = requests.post(
                f"{server_url}/step",
                json={"session_id": session_id, "action": action},
                timeout=15,
            )
            sr.raise_for_status()
            step_data = sr.json()
        except Exception as exc:
            print(f"    [step error] {task_id} step {step}: {exc}", file=sys.stderr)
            break
        reward = float(step_data.get("reward", 0.0))
        done = step_data.get("done", True)
        step_rewards.append(reward)
        last_reward = reward
        step += 1

    return last_reward, step_rewards


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Random agent baseline for Medical Triage Environment")
    p.add_argument("--server-url", default="http://localhost:8000", help="Environment server URL")
    p.add_argument("--runs", type=int, default=1, help="Number of independent runs to average (default: 1)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Health check
    try:
        r = requests.get(f"{args.server_url}/health", timeout=10)
        r.raise_for_status()
        print(f"Server: {args.server_url}  (v{r.json().get('version', '?')})")
    except Exception as exc:
        print(f"[ERROR] Cannot reach server at {args.server_url}: {exc}")
        sys.exit(1)

    print(f"Random seed: {args.seed} | Runs per case: {args.runs}\n")

    # Aggregate scores across runs
    task_scores: dict[str, list[float]] = {t[0]: [] for t in TASKS}

    for run in range(args.runs):
        if args.runs > 1:
            print(f"── Run {run + 1}/{args.runs} ──")
        for task_id, difficulty, case_indices in TASKS:
            for ci in case_indices:
                reward, step_rewards = run_episode(args.server_url, task_id, ci)
                task_scores[task_id].append(reward)
                rewards_str = ", ".join(f"{r:.4f}" for r in step_rewards)
                print(f"  {task_id}[{ci}]  reward={reward:.4f}  steps=[{rewards_str}]")
        if args.runs > 1:
            print()

    # Summary table
    print("\n" + "=" * 65)
    print(f"  RANDOM AGENT BASELINE — Medical Triage Environment v2.3.0")
    print(f"  Seed: {args.seed} | Runs per case: {args.runs}")
    print("=" * 65)
    print(f"  {'Task':<30} {'Difficulty':<10} {'Avg Score':>10} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 62)

    grand_scores: list[float] = []
    for task_id, difficulty, _ in TASKS:
        scores = task_scores[task_id]
        avg = sum(scores) / len(scores) if scores else 0.0
        mn  = min(scores) if scores else 0.0
        mx  = max(scores) if scores else 0.0
        grand_scores.extend(scores)
        print(f"  {task_id:<30} {difficulty:<10} {avg:>10.4f} {mn:>8.4f} {mx:>8.4f}")

    grand_avg = sum(grand_scores) / len(grand_scores) if grand_scores else 0.0
    print("  " + "-" * 62)
    print(f"  {'OVERALL':<30} {'—':<10} {grand_avg:>10.4f}")
    print("=" * 65)
    print(f"\nExpected theoretical random lower bound: 0.10–0.15")
    print(f"This run overall avg: {grand_avg:.4f}")


if __name__ == "__main__":
    main()
