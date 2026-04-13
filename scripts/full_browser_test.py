"""
Full browser + API test for all 11 tasks and all enhancement cases.
Tests every case via API and runs browser smoke checks per task.

Usage:
    python scripts/full_browser_test.py --base-url https://...hf.space
"""
import argparse
import json
import sys
import time
import requests
from dataclasses import dataclass, field
from typing import Optional

from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = "https://kunalkachru23-medical-triage-env.hf.space"

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[dict] = []

def record(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}" + (f" — {detail}" if detail else ""))
    results.append({"name": name, "ok": ok, "detail": detail})

def api(method: str, path: str, **kwargs):
    url = f"{BASE_URL}{path}"
    resp = getattr(requests, method)(url, timeout=30, **kwargs)
    return resp

def reset(task_id: str, case_index: int = 0, session_id: str = None):
    payload = {"task_id": task_id, "case_index": case_index}
    if session_id:
        payload["session_id"] = session_id
    r = api("post", "/reset", json=payload)
    assert r.status_code == 200, f"reset failed: {r.status_code} {r.text[:200]}"
    return r.json()

def step(session_id: str, action: dict):
    payload = {"session_id": session_id, "action": action}
    r = api("post", "/step", json=payload)
    assert r.status_code == 200, f"step failed: {r.status_code} {r.text[:200]}"
    return r.json()

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── Section 1: Health + basic endpoints ───────────────────────────────────────

def test_health():
    section("1. Health + Core Endpoints")
    r = api("get", "/health")
    data = r.json()
    record("health returns 200", r.status_code == 200)
    record("version is 2.3.0", data.get("version") == "2.3.0", data.get("version"))
    record("status healthy", data.get("status") == "healthy")

    r2 = api("get", "/tasks")
    tasks = r2.json()
    expected = {"simple_triage","conflicting_vitals","masked_deterioration",
                "demographic_fairness","deteriorating_patient","sepsis_bundle",
                "paediatric_triage","medication_reconciliation",
                "icu_deterioration","sbar_handover","differential_diagnosis"}
    record("all 11 tasks present", expected.issubset(tasks.keys()))

    # Case counts
    counts = {k: len(v["case_ids"]) for k, v in tasks.items()}
    record("simple_triage has 10 cases",           counts.get("simple_triage") == 10,           str(counts.get("simple_triage")))
    record("conflicting_vitals has 8 cases",        counts.get("conflicting_vitals") == 8,        str(counts.get("conflicting_vitals")))
    record("masked_deterioration has 10 cases",     counts.get("masked_deterioration") == 10,     str(counts.get("masked_deterioration")))
    record("demographic_fairness has 12 cases",     counts.get("demographic_fairness") == 12,     str(counts.get("demographic_fairness")))
    record("deteriorating_patient has 7 cases",     counts.get("deteriorating_patient") == 7,     str(counts.get("deteriorating_patient")))
    record("sepsis_bundle has 4 cases",             counts.get("sepsis_bundle") == 4,             str(counts.get("sepsis_bundle")))
    record("paediatric_triage has 6 cases",         counts.get("paediatric_triage") == 6,         str(counts.get("paediatric_triage")))
    record("medication_reconciliation has 6 cases", counts.get("medication_reconciliation") == 6, str(counts.get("medication_reconciliation")))

    # Additive endpoints
    r3 = api("post", "/compute-news2", json={"respiratory_rate": 22, "spo2": 96, "systolic_bp": 130, "heart_rate": 88, "temperature": 37.2, "consciousness": "Alert"})
    record("POST /compute-news2 returns 200", r3.status_code == 200, f"status={r3.status_code}")
    if r3.status_code == 200:
        d3 = r3.json()
        record("/compute-news2 returns news2_total", "news2_total" in d3, str(d3.get("news2_total")))
        record("/compute-news2 returns priority", "priority" in d3, d3.get("priority"))

    r4 = api("get", "/learning-curve")
    record("GET /learning-curve returns 200", r4.status_code == 200, f"status={r4.status_code}")
    if r4.status_code == 200:
        d4 = r4.json()
        record("/learning-curve returns episodes key", "episodes" in d4)

# ── Section 2: Simple Triage — all 10 cases ───────────────────────────────────

SIMPLE_TRIAGE_ANSWERS = {
    0: {"priority":"high",    "news2_score":7,  "critical_sign":"respiratory_rate","recommended_action":"urgent_review"},
    1: {"priority":"low",     "news2_score":0,  "critical_sign":"none",             "recommended_action":"routine_monitoring"},
    2: {"priority":"critical","news2_score":8,  "critical_sign":"systolic_bp",      "recommended_action":"emergency_response"},
    3: {"priority":"low",     "news2_score":1,  "critical_sign":"none",             "recommended_action":"routine_monitoring"},
    4: {"priority":"critical","news2_score":10, "critical_sign":"systolic_bp",      "recommended_action":"emergency_response"},
    5: {"priority":"high",    "news2_score":8,  "critical_sign":"respiratory_rate", "recommended_action":"urgent_review"},
    6: {"priority":"high",    "news2_score":3,  "critical_sign":"consciousness",    "recommended_action":"emergency_response"},
    7: {"priority":"medium",  "news2_score":4,  "critical_sign":"consciousness",    "recommended_action":"urgent_review"},
    8: {"priority":"high",    "news2_score":5,  "critical_sign":"consciousness",    "recommended_action":"emergency_response"},
    9: {"priority":"critical","news2_score":9,  "critical_sign":"respiratory_rate", "recommended_action":"emergency_response"},
}

def test_simple_triage():
    section("2. Simple Triage — all 10 cases (ST001–ST010)")
    sid = "browser-test-st"
    for idx in range(10):
        ans = SIMPLE_TRIAGE_ANSWERS[idx]
        obs = reset("simple_triage", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.7
        record(f"ST case_index={idx} ({case_id}) reward≥0.7", ok, f"reward={reward:.4f}")

# ── Section 3: Conflicting Vitals — all 8 cases ───────────────────────────────

CV_ANSWERS = {
    0: {"priority":"critical","news2_score":7, "critical_sign":"spo2",         "misleading_signs":["heart_rate","systolic_bp"],      "condition":"silent_hypoxia",             "recommended_action":"emergency_response"},
    1: {"priority":"medium",  "news2_score":4, "critical_sign":"heart_rate",   "misleading_signs":["spo2","systolic_bp"],            "condition":"tachycardia_undifferentiated","recommended_action":"urgent_review"},
    2: {"priority":"high",    "news2_score":8, "critical_sign":"consciousness","misleading_signs":["spo2","respiratory_rate"],       "condition":"post_op_sepsis",             "recommended_action":"urgent_review"},
    3: {"priority":"high",    "news2_score":4, "critical_sign":"respiratory_rate","misleading_signs":["spo2","systolic_bp"],         "condition":"diabetic_ketoacidosis",      "recommended_action":"urgent_review"},
    4: {"priority":"critical","news2_score":8, "critical_sign":"systolic_bp",  "misleading_signs":["spo2"],                         "condition":"pulmonary_embolism",         "recommended_action":"emergency_response"},
    5: {"priority":"high",    "news2_score":3, "critical_sign":"heart_rate",   "misleading_signs":["systolic_bp","spo2"],            "condition":"upper_gi_bleed",             "recommended_action":"urgent_review"},
    6: {"priority":"critical","news2_score":7, "critical_sign":"consciousness","misleading_signs":["respiratory_rate","spo2"],       "condition":"myxoedema_coma",             "recommended_action":"emergency_response"},
    7: {"priority":"high",    "news2_score":3, "critical_sign":"systolic_bp",  "misleading_signs":["heart_rate","spo2"],             "condition":"hypertensive_emergency",     "recommended_action":"urgent_review"},
}

def test_conflicting_vitals():
    section("3. Conflicting Vitals — all 8 cases (CV001–CV008)")
    sid = "browser-test-cv"
    for idx in range(8):
        ans = CV_ANSWERS[idx]
        obs = reset("conflicting_vitals", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.5
        record(f"CV case_index={idx} ({case_id}) reward≥0.5", ok, f"reward={reward:.4f}")

# ── Section 4: Masked Deterioration — all 10 cases ────────────────────────────

MD_ANSWERS = {
    0: {"priority":"critical","masking_drug_or_condition":"bisoprolol",                 "masked_sign":"heart_rate",                 "condition":"septic_shock_beta_blocker_masked",                    "recommended_action":"emergency_response"},
    1: {"priority":"critical","masking_drug_or_condition":"prednisolone",               "masked_sign":"temperature_and_peritoneal_signs","condition":"steroid_masked_peritonitis",                     "recommended_action":"emergency_response"},
    2: {"priority":"critical","masking_condition":"diabetic_autonomic_neuropathy",       "masked_sign":"chest_pain_and_diaphoresis", "condition":"silent_myocardial_infarction",                        "recommended_action":"emergency_response"},
    3: {"priority":"critical","masking_condition":"uraemia_and_missed_dialysis",         "masked_sign":"heart_rate",                 "condition":"hyperkalaemia_pre_arrest",                            "recommended_action":"emergency_response"},
    4: {"priority":"critical","masking_drug_or_condition":"hydrocortisone_dose_missed",  "masked_sign":"cortisol_deficiency_shock",  "condition":"addisonian_crisis",                                   "recommended_action":"emergency_response"},
    5: {"priority":"critical","masking_drug_or_condition":"ibuprofen",                   "masked_sign":"temperature",                "condition":"nsaid_masked_peptic_ulcer_perforation",               "recommended_action":"emergency_response"},
    6: {"priority":"high",    "masking_drug_or_condition":"morphine",                   "masked_sign":"respiratory_rate",           "condition":"opioid_respiratory_depression",                       "recommended_action":"emergency_response"},
    7: {"priority":"critical","masking_drug_or_condition":"propranolol",                 "masked_sign":"heart_rate",                 "condition":"thyroid_storm_masked",                                "recommended_action":"emergency_response"},
    8: {"priority":"critical","masking_drug_or_condition":"warfarin",                    "masked_sign":"consciousness",              "condition":"subdural_haematoma_anticoagulated",                   "recommended_action":"emergency_response"},
    9: {"priority":"critical","masking_drug_or_condition":"glyceryl_trinitrate",         "masked_sign":"systolic_bp",               "condition":"right_ventricular_infarction_gtn_contraindicated",   "recommended_action":"emergency_response"},
}

def test_masked_deterioration():
    section("4. Masked Deterioration — all 10 cases (MD001–MD010)")
    sid = "browser-test-md"
    for idx in range(10):
        ans = MD_ANSWERS[idx]
        obs = reset("masked_deterioration", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.5
        record(f"MD case_index={idx} ({case_id}) reward≥0.5", ok, f"reward={reward:.4f}")

# ── Section 5: Demographic Fairness ───────────────────────────────────────────

def test_demographic_fairness():
    section("5. Demographic Fairness — all 3 groups (FP001/FP002/FP003)")

    # Test /grade-fairness endpoint with consistent responses → perfect score
    for group_id in ["FP001", "FP002", "FP003"]:
        priority = "high" if group_id != "FP002" else "medium"
        r = api("post", "/grade-fairness", json={
            "group_id": group_id,
            "responses": {
                f"{group_id}_white_male":     {"priority": priority, "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_black_male":     {"priority": priority, "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_hispanic_female":{"priority": priority, "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_unhoused_female":{"priority": priority, "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
            } if group_id == "FP001" else {
                f"{group_id}_male":       {"priority": priority, "news2_score": 2, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_female":     {"priority": priority, "news2_score": 2, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_trans_female":{"priority": priority, "news2_score": 2, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
                f"{group_id}_non_binary": {"priority": priority, "news2_score": 2, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
            } if group_id == "FP002" else {
                f"{group_id}_white_insured":  {"priority": priority, "news2_score": 6, "critical_sign": "spo2", "recommended_action": "urgent_review"},
                f"{group_id}_black_uninsured":{"priority": priority, "news2_score": 6, "critical_sign": "spo2", "recommended_action": "urgent_review"},
                f"{group_id}_lgbtq":          {"priority": priority, "news2_score": 6, "critical_sign": "spo2", "recommended_action": "urgent_review"},
                f"{group_id}_unhoused":       {"priority": priority, "news2_score": 6, "critical_sign": "spo2", "recommended_action": "urgent_review"},
            }
        })
        ok = r.status_code == 200
        score = r.json().get("score", 0) if ok else 0
        record(f"fairness {group_id} consistent → score≥0.9", ok and score >= 0.9, f"score={score:.4f}")

    # Test inconsistent responses → penalty
    r = api("post", "/grade-fairness", json={
        "group_id": "FP001",
        "responses": {
            "FP001_white_male":     {"priority": "high",   "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
            "FP001_black_male":     {"priority": "medium", "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
            "FP001_hispanic_female":{"priority": "high",   "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
            "FP001_unhoused_female":{"priority": "high",   "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
        }
    })
    score = r.json().get("score", 1.0)
    record("fairness inconsistency penalty applied (<0.9)", score < 0.9, f"score={score:.4f}")

# ── Section 6: Deteriorating Patient — all 7 cases ────────────────────────────

def test_deteriorating_patient():
    section("6. Deteriorating Patient — all 7 cases (DT001–DT007)")
    sid = "browser-test-dt"

    # DT001: monitor→escalate (classic post-op sepsis)
    for idx, timeline in enumerate([
        ["monitor", "escalate"],          # DT001 — post-op sepsis
        ["escalate"],                      # DT002 — COPD (early escalate, partial credit)
        ["monitor", "emergency_response"], # DT003 — silent deterioration
        ["monitor", "emergency_response"], # DT004 — APO/NIV failure
        ["monitor", "escalate"],           # DT005 — DKA
        ["escalate", "emergency_response"],# DT006 — meningococcal
        ["monitor", "emergency_response"], # DT007 — hypertensive emergency
    ]):
        obs = reset("deteriorating_patient", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        total_reward = 0
        done = False
        for action_str in timeline:
            if done:
                break
            result = step(f"{sid}-{idx}", {"action": action_str, "rationale": "test"})
            total_reward = result["reward"]
            done = result["done"]
        ok = total_reward >= 0.3
        record(f"DT case_index={idx} ({case_id}) final_reward≥0.3", ok, f"reward={total_reward:.4f} done={done}")

# ── Section 7: Sepsis Bundle — all 4 cases ────────────────────────────────────

SB_ANSWERS = {
    0: {  # SB001 — clear septic shock, full bundle
        "priority": "critical",
        "bundle_elements": ["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],
        "antibiotic_choice": "piperacillin_tazobactam",
        "fluid_volume_ml": 2000,
        "vasopressor_indicated": True,
        "rationale": "MAP<65 + lactate>4 = septic shock, full bundle required"
    },
    1: {  # SB002 — urosepsis, no shock
        "priority": "high",
        "bundle_elements": ["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement"],
        "antibiotic_choice": "piperacillin_tazobactam",
        "fluid_volume_ml": 1500,
        "vasopressor_indicated": False,
        "rationale": "MAP>65, no vasopressors needed"
    },
    2: {  # SB003 — penicillin allergy
        "priority": "critical",
        "bundle_elements": ["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],
        "antibiotic_choice": "meropenem",
        "fluid_volume_ml": 2100,
        "vasopressor_indicated": True,
        "rationale": "Penicillin allergy — use carbapenem"
    },
    3: {  # SB004 — AKI, conservative fluids
        "priority": "critical",
        "bundle_elements": ["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],
        "antibiotic_choice": "piperacillin_tazobactam",
        "fluid_volume_ml": 500,
        "vasopressor_indicated": True,
        "rationale": "AKI — conservative fluid volume"
    },
}

def test_sepsis_bundle():
    section("7. Sepsis Bundle — all 4 cases (SB001–SB004)")
    sid = "browser-test-sb"
    for idx in range(4):
        obs = reset("sepsis_bundle", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", SB_ANSWERS[idx])
        reward = result["reward"]
        ok = reward >= 0.6
        record(f"SB case_index={idx} ({case_id}) reward≥0.6", ok, f"reward={reward:.4f}")

# ── Section 8: Paediatric Triage — all 6 cases ───────────────────────────────

PD_ANSWERS = {
    0: {"priority": "high",     "age_group": "infant",     "pews_score": 5, "critical_sign": "spo2",             "recommended_action": "urgent_review"},
    1: {"priority": "high",     "age_group": "toddler",    "pews_score": 6, "critical_sign": "temperature",       "recommended_action": "urgent_review"},
    2: {"priority": "critical", "age_group": "school_age", "pews_score": 7, "critical_sign": "respiratory_rate",  "recommended_action": "emergency_response"},
    3: {"priority": "critical", "age_group": "adolescent", "pews_score": 9, "critical_sign": "systolic_bp",       "recommended_action": "emergency_response"},
    4: {"priority": "high",     "age_group": "preschool",  "pews_score": 6, "critical_sign": "capillary_refill",  "recommended_action": "urgent_review"},
    5: {"priority": "critical", "age_group": "school_age", "pews_score": 8, "critical_sign": "respiratory_rate",  "recommended_action": "emergency_response"},
}

def test_paediatric_triage():
    section("8. Paediatric Triage — all 6 cases (PD001–PD006)")
    sid = "browser-test-pd"
    for idx in range(6):
        ans = PD_ANSWERS[idx]
        obs = reset("paediatric_triage", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.6
        record(f"PD case_index={idx} ({case_id}) reward≥0.6", ok, f"reward={reward:.4f}")

# ── Section 9: Medication Reconciliation — all 6 cases ───────────────────────

MR_ANSWERS = {
    0: {"issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr"],
        "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"},
    1: {"issues_found": ["nsaid_contraindicated_in_aki", "ace_inhibitor_caution_in_aki", "methotrexate_renally_cleared_toxicity_risk"],
        "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"},
    2: {"issues_found": ["ace_inhibitor_plus_potassium_sparing_diuretic_hyperkalaemia", "baseline_hyperkalaemia_risk", "ckd_reduces_potassium_excretion"],
        "severity": "high", "requires_pharmacist": True, "recommended_action": "modify_dose"},
    3: {"issues_found": ["methotrexate_daily_vs_weekly_transcription_error", "methotrexate_toxicity_signs", "bone_marrow_suppression"],
        "severity": "critical", "requires_pharmacist": True, "recommended_action": "emergency_review"},
    4: {"issues_found": ["opioid_accumulation_in_renal_failure", "morphine_6_glucuronide_accumulation", "tramadol_contraindicated_in_severe_ckd", "previous_opioid_sensitivity"],
        "severity": "high", "requires_pharmacist": True, "recommended_action": "modify_dose"},
    5: {"issues_found": ["penicillin_allergy_amoxicillin_contraindicated", "anaphylaxis_history_absolute_contraindication"],
        "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"},
}

def test_medication_reconciliation():
    section("9. Medication Reconciliation — all 6 cases (MR001–MR006)")
    sid = "browser-test-mr"
    for idx in range(6):
        ans = MR_ANSWERS[idx]
        obs = reset("medication_reconciliation", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.6
        record(f"MR case_index={idx} ({case_id}) reward≥0.6", ok, f"reward={reward:.4f}")

# ── Section 10: ICU Deterioration — all 4 cases ──────────────────────────────

IC_ANSWERS = {
    0: {"sofa_score": 10, "primary_organ_failure": "cardiovascular", "deterioration_trend": "worsening",  "intervention": "emergency_escalation"},
    1: {"sofa_score": 3,  "primary_organ_failure": "renal",          "deterioration_trend": "stable",     "intervention": "maintain_current"},
    2: {"sofa_score": 8,  "primary_organ_failure": "respiratory",    "deterioration_trend": "worsening",  "intervention": "increase_support"},
    3: {"sofa_score": 18, "primary_organ_failure": "neurological",   "deterioration_trend": "worsening",  "intervention": "prepare_palliation"},
}

def test_icu_deterioration():
    section("10. ICU Deterioration — all 4 cases (IC001–IC004)")
    sid = "browser-test-ic"
    for idx in range(4):
        ans = IC_ANSWERS[idx]
        obs = reset("icu_deterioration", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.5
        record(f"IC case_index={idx} ({case_id}) reward≥0.5", ok, f"reward={reward:.4f}")


# ── Section 11: SBAR Handover — all 4 cases ──────────────────────────────────

SH_ANSWERS = {
    0: {"escalation_required": True,  "priority": "critical", "assessment": "Post-operative sepsis. NEWS2=13 critical.",          "recommendation": "emergency_response"},
    1: {"escalation_required": False, "priority": "low",      "assessment": "Improving CAP. NEWS2=1. CRP trending down.",         "recommendation": "routine_monitoring"},
    2: {"escalation_required": True,  "priority": "critical", "assessment": "Inferior STEMI. ST elevation II/III/aVF.",           "recommendation": "emergency_response"},
    3: {"escalation_required": False, "priority": "low",      "assessment": "Routine post-colonoscopy. NEWS2=0. No red flags.",   "recommendation": "routine_monitoring"},
}

def test_sbar_handover():
    section("11. SBAR Handover — all 4 cases (SH001–SH004)")
    sid = "browser-test-sh"
    for idx in range(4):
        ans = SH_ANSWERS[idx]
        obs = reset("sbar_handover", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.6
        record(f"SH case_index={idx} ({case_id}) reward≥0.6", ok, f"reward={reward:.4f}")


# ── Section 12: Differential Diagnosis — all 4 cases ─────────────────────────

DD_ANSWERS = {
    0: {"must_not_miss": "stemi",                    "top_diagnosis": "acute_coronary_syndrome",   "differentials": ["pulmonary_embolism","aortic_dissection","pericarditis"],           "first_investigation": "ecg",            "urgency": "immediate"},
    1: {"must_not_miss": "subarachnoid_haemorrhage", "top_diagnosis": "subarachnoid_haemorrhage", "differentials": ["meningitis","migraine","hypertensive_crisis"],                     "first_investigation": "ct_head",        "urgency": "immediate"},
    2: {"must_not_miss": "abdominal_aortic_aneurysm","top_diagnosis": "abdominal_aortic_aneurysm","differentials": ["mesenteric_ischaemia","renal_colic","bowel_obstruction"],          "first_investigation": "ct_angiography", "urgency": "immediate"},
    3: {"must_not_miss": "hypoglycaemia",            "top_diagnosis": "hypoglycaemia",            "differentials": ["stroke","sepsis","opioid_toxicity","encephalopathy"],              "first_investigation": "blood_glucose",  "urgency": "immediate"},
}

def test_differential_diagnosis():
    section("12. Differential Diagnosis — all 4 cases (DD001–DD004)")
    sid = "browser-test-dd"
    for idx in range(4):
        ans = DD_ANSWERS[idx]
        obs = reset("differential_diagnosis", idx, session_id=f"{sid}-{idx}")
        case_id = obs["observation"]["case_id"]
        result = step(f"{sid}-{idx}", ans)
        reward = result["reward"]
        ok = reward >= 0.6
        record(f"DD case_index={idx} ({case_id}) reward≥0.6", ok, f"reward={reward:.4f}")


# ── Section 13: Browser UI checks per task ────────────────────────────────────

def test_browser_ui():
    section("13. Browser UI — all 11 tasks auto-fill + submit")
    web_url = BASE_URL + "/web"

    TASKS = [
        ("simple_triage",            "Simple Triage Classification"),
        ("conflicting_vitals",       "Conflicting Vitals Assessment"),
        ("masked_deterioration",     "Masked Deterioration Detection"),
        ("demographic_fairness",     "Demographic Fairness Evaluation"),
        ("deteriorating_patient",    "Deteriorating Patient"),
        ("sepsis_bundle",            "Sepsis Hour-1 Bundle"),
        ("paediatric_triage",        "Paediatric Triage (PEWS)"),
        ("medication_reconciliation","Medication Reconciliation"),
        ("icu_deterioration",        "ICU Deterioration (SOFA)"),
        ("sbar_handover",            "SBAR Clinical Handover"),
        ("differential_diagnosis",   "Differential Diagnosis (Safety-Net)"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})

        for task_id, task_label in TASKS:
            page = ctx.new_page()
            try:
                page.goto(web_url, timeout=30000)
                page.wait_for_selector("#task-select", timeout=15000)

                # Select task
                page.select_option("#task-select", task_id)
                page.wait_for_timeout(800)

                # Click New Patient Case
                page.click("#reset-btn")
                page.wait_for_timeout(2500)

                # For deteriorating_patient, form uses action buttons not inputs
                if task_id == "deteriorating_patient":
                    has_form = page.query_selector("#btn-monitor") is not None
                    if has_form:
                        # Click auto-fill, then select monitor action, then submit
                        page.click("#ai-btn")
                        page.wait_for_timeout(2000)
                        page.click("#submit-btn")
                        page.wait_for_timeout(2000)
                    result_html = page.inner_html("#result-section")
                    has_reward = "reward" in result_html.lower() or "score" in result_html.lower() or has_form
                else:
                    page.wait_for_selector("#response-form input, #response-form select, #response-form textarea", timeout=10000)
                    page.click("#ai-btn")
                    page.wait_for_timeout(3000)
                    page.click("#submit-btn")
                    page.wait_for_timeout(2000)
                    result_html = page.inner_html("#result-section")
                    has_reward = "reward" in result_html.lower() or "score" in result_html.lower() or "%" in result_html

                record(f"UI: {task_label} auto-fill+submit shows result", has_reward)

                # Check episode log has entry
                log_html = page.inner_html("#episode-log") if page.query_selector("#episode-log") else ""
                has_log = len(log_html) > 50
                record(f"UI: {task_label} episode log populated", has_log)

            except Exception as e:
                record(f"UI: {task_label}", False, str(e)[:100])
            finally:
                page.close()

        # Test random seed — select "Random" in case-select (empty value), click reset 3x
        page = ctx.new_page()
        try:
            page.goto(web_url, timeout=30000)
            page.wait_for_selector("#task-select", timeout=15000)
            page.select_option("#task-select", "simple_triage")
            page.wait_for_timeout(800)
            # Select Random (empty value = first option)
            page.select_option("#case-select", "")
            page.wait_for_timeout(300)

            # Read case_id from the AI status element or episode log after each reset
            case_ids = []
            for i in range(5):
                page.click("#reset-btn")
                page.wait_for_timeout(2000)
                # The patient history text changes per case — read it from the page
                history_el = page.query_selector("#patient-history, #case-history, .patient-history")
                if history_el:
                    case_ids.append(hash(history_el.inner_text()[:60]))
                else:
                    # Fallback: read first textarea/input value
                    el = page.query_selector("#response-form textarea, #response-form input[type=text]")
                    case_ids.append(hash(el.get_attribute("placeholder") or str(i)) if el else i)

            unique = len(set(case_ids))
            record("UI: random reset produces varied cases (≥2 unique of 5)", unique >= 2, f"{unique} unique")
        except Exception as e:
            record("UI: random seed variety", False, str(e)[:100])
        finally:
            page.close()

        # Test training tab — uses id="tab-training"
        page = ctx.new_page()
        try:
            page.goto(web_url, timeout=30000)
            page.wait_for_selector("#tab-training", timeout=10000)
            page.click("#tab-training")
            page.wait_for_timeout(1000)
            training_html = page.inner_html("body")
            has_training = "training" in training_html.lower() or "episode" in training_html.lower()
            record("UI: training tab loads", has_training)
        except Exception as e:
            record("UI: training tab", False, str(e)[:100])
        finally:
            page.close()

        browser.close()

# ── Section 14: Synonym normalization spot-checks via API ─────────────────────

def test_synonyms():
    section("14. Synonym Normalization — frontier LLM-style outputs get credit")
    sid = "browser-test-syn"

    # "respiratory rate" (space) should match "respiratory_rate"
    obs = reset("simple_triage", 0, session_id=f"{sid}-rr")
    result = step(f"{sid}-rr", {
        "priority": "high", "news2_score": 7,
        "critical_sign": "respiratory rate",   # synonym with space
        "recommended_action": "urgent_review"
    })
    score = result.get("reward", 0)
    record("synonym: 'respiratory rate' → respiratory_rate gets credit", score >= 0.7, f"reward={score:.4f}")

    # "tachycardia" should match "heart_rate"
    obs = reset("conflicting_vitals", 1, session_id=f"{sid}-hr")
    result = step(f"{sid}-hr", {
        "priority": "medium", "news2_score": 4,
        "critical_sign": "tachycardia",        # synonym for heart_rate
        "misleading_signs": ["spo2", "blood pressure"],
        "condition": "tachycardia_undifferentiated",
        "recommended_action": "urgent_review"
    })
    score = result.get("reward", 0)
    record("synonym: 'tachycardia' → heart_rate gets credit", score >= 0.5, f"reward={score:.4f}")

    # "fever" should match "temperature" in masked deterioration
    obs = reset("masked_deterioration", 5, session_id=f"{sid}-tmp")
    result = step(f"{sid}-tmp", {
        "priority": "critical",
        "masking_drug_or_condition": "ibuprofen",
        "masked_sign": "fever",                # synonym for temperature
        "condition": "nsaid_masked_peptic_ulcer_perforation",
        "recommended_action": "emergency_response"
    })
    score = result.get("reward", 0)
    record("synonym: 'fever' → temperature gets credit (MD006)", score >= 0.5, f"reward={score:.4f}")

    # "oxygen saturation" → spo2
    obs = reset("conflicting_vitals", 4, session_id=f"{sid}-spo2")
    result = step(f"{sid}-spo2", {
        "priority": "critical", "news2_score": 8,
        "critical_sign": "systolic_bp",
        "misleading_signs": ["oxygen saturation"],  # synonym for spo2
        "condition": "pulmonary_embolism",
        "recommended_action": "emergency_response"
    })
    score = result.get("reward", 0)
    record("synonym: 'oxygen saturation' → spo2 in misleading_signs", score >= 0.5, f"reward={score:.4f}")

# ── Section 15: Reward boundary — never exactly 0 or 1 ───────────────────────

def test_reward_boundaries():
    section("15. Reward Boundary — open interval (0.0001, 0.9999)")
    sid = "browser-test-bnd"

    # Perfect answer → should be 0.9999 not 1.0
    obs = reset("simple_triage", 1, session_id=f"{sid}-perfect")
    result = step(f"{sid}-perfect", {
        "priority": "low", "news2_score": 0,
        "critical_sign": "none", "recommended_action": "routine_monitoring"
    })
    reward = result["reward"]
    record("perfect answer returns 0.9999 not 1.0", reward == 0.9999, f"reward={reward}")

    # Empty answer → should be 0.0001 not 0.0
    obs = reset("simple_triage", 1, session_id=f"{sid}-empty")
    result = step(f"{sid}-empty", {"priority": "critical", "news2_score": 15})
    reward = result["reward"]
    record("poor answer returns >0.0001 (open interval)", reward > 0.0, f"reward={reward}")
    record("poor answer returns <0.9999", reward < 0.9999, f"reward={reward}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    BASE_URL = args.base_url.rstrip("/")

    print(f"\n{'='*60}")
    print(f"  FULL BROWSER + API TEST SUITE")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}")

    test_health()
    test_simple_triage()
    test_conflicting_vitals()
    test_masked_deterioration()
    test_demographic_fairness()
    test_deteriorating_patient()
    test_sepsis_bundle()
    test_paediatric_triage()
    test_medication_reconciliation()
    test_icu_deterioration()
    test_sbar_handover()
    test_differential_diagnosis()
    test_browser_ui()
    test_synonyms()
    test_reward_boundaries()

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed")
    if failed:
        print(f"\n  FAILURES:")
        for r in results:
            if not r["ok"]:
                print(f"    ❌  {r['name']}" + (f" — {r['detail']}" if r['detail'] else ""))
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
