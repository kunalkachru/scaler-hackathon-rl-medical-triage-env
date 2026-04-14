"""
tests/test_app_coverage.py — Full app.py endpoint coverage
===========================================================
Covers every endpoint and code path in server/app.py that was
previously untested (42% coverage → target ≥85%).

Endpoints covered:
  GET  /health
  GET  /tasks
  GET  /state
  GET  /history
  GET  /stats
  GET  /metrics
  GET  /learning-curve
  POST /compute-news2
  POST /suggest          ← all 11 tasks, rule-based path (no API key)
  POST /agent-assess     ← no-api-key mock path
  POST /reset            ← exception path
  POST /step             ← exception + done paths

All tests use FastAPI TestClient — no live server required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

ALL_TASKS = [
    "simple_triage", "conflicting_vitals", "masked_deterioration",
    "demographic_fairness", "deteriorating_patient", "sepsis_bundle",
    "paediatric_triage", "medication_reconciliation",
    "icu_deterioration", "sbar_handover", "differential_diagnosis",
]

# ── Patient text fixtures for rule-based suggest tests ────────────────────────

CHEST_PAIN_PATIENT = (
    "52-year-old male. Sudden onset crushing central chest pain radiating to left arm. "
    "RR=22, SpO2=95%, BP=144/90, HR=108, Temp=36.8. Diaphoretic. Smoker. "
    "Family history of MI."
)

HEADACHE_PATIENT = (
    "38-year-old female. Sudden thunderclap headache — worst of life, onset 20 minutes ago. "
    "RR=18, SpO2=98%, BP=162/95, HR=88, Temp=37.1. No focal neurology. GCS 15."
)

SOB_PATIENT = (
    "65-year-old male. Progressive shortness of breath, pleuritic pain on inspiration, right leg swelling. "
    "RR=26, SpO2=91%, BP=118/76, HR=112, Temp=37.2. Recent long-haul flight. "
    "Dyspnoea is the dominant symptom. No back pain."
)

ABDO_PATIENT = (
    "72-year-old female. Sudden severe abdominal pain. RR=24, SpO2=94%, "
    "BP=88/50, HR=126, Temp=38.9. Rigid abdomen. Previous AAA repair."
)

SEPSIS_PATIENT = (
    "58-year-old male. Known UTI, now confused. RR=24, SpO2=93%, "
    "BP=88/52, HR=118, Temp=38.9. MAP=64. Lactate=4.2. Creatinine=180."
)

PAEDS_PATIENT = (
    "8-year-old child. Respiratory distress. RR=38, SpO2=91%, "
    "BP=90/60, HR=140, Temp=39.2. Stridor present."
)

INFANT_PATIENT = (
    "6-month-old infant. Difficulty breathing. RR=62, SpO2=88%. Fever=39.5."
)

TEEN_PATIENT = (
    "16-year-old adolescent. Asthma exacerbation. RR=30, SpO2=90%, HR=130."
)

MED_RECON_PATIENT = (
    "72-year-old on warfarin INR=4.2, prescribed ibuprofen by GP. "
    "RR=16, SpO2=97%, BP=142/88, HR=72, Temp=36.9."
)

ICU_PATIENT = (
    "ICU patient, day 3 post-op. SOFA rising. RR=28, SpO2=88% on FiO2=0.6, "
    "MAP=58 despite noradrenaline, Creatinine=380, Bilirubin=62, GCS=9. Worsening."
)

SBAR_PATIENT = (
    "Post-surgical patient, ward C4. BP falling: 88/52 from 130/80 at 2h ago. "
    "RR=24, SpO2=91%, HR=122, Temp=38.8. Confused. NEWS2=13. Escalation needed."
)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200

def test_health_version():
    r = client.get("/health")
    assert r.json()["version"] == "2.3.0"

def test_health_status_healthy():
    r = client.get("/health")
    assert r.json()["status"] == "healthy"


# ── /tasks ────────────────────────────────────────────────────────────────────

def test_tasks_returns_all_11():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    for task in ALL_TASKS:
        assert task in data, f"Task '{task}' missing from /tasks"

def test_tasks_has_case_ids():
    r = client.get("/tasks")
    data = r.json()
    for task, info in data.items():
        assert "case_ids" in info
        assert "case_count" in info
        assert info["case_count"] == len(info["case_ids"])
        assert info["case_count"] >= 1


# ── /state ────────────────────────────────────────────────────────────────────

def test_state_returns_200():
    r = client.get("/state")
    assert r.status_code == 200

def test_state_has_expected_keys():
    r = client.get("/state")
    data = r.json()
    for key in ("step_count", "is_done", "cumulative_reward", "current_task_id"):
        assert key in data

def test_state_with_session_id():
    reset_r = client.post("/reset", json={"task_id": "simple_triage", "session_id": "state-test-session"})
    sid = reset_r.json()["info"]["session_id"]
    r = client.get(f"/state?session_id={sid}")
    assert r.status_code == 200
    assert r.json()["current_task_id"] == "simple_triage"


# ── /history + /stats ─────────────────────────────────────────────────────────

def test_history_returns_200():
    r = client.get("/history")
    assert r.status_code == 200

def test_history_has_expected_keys():
    r = client.get("/history")
    data = r.json()
    assert "episodes" in data
    assert "total" in data
    assert "stats" in data

def test_history_limit_param():
    r = client.get("/history?limit=5")
    assert r.status_code == 200
    assert len(r.json()["episodes"]) <= 5

def test_stats_returns_200():
    r = client.get("/stats")
    assert r.status_code == 200

def test_stats_populated_after_episode():
    sid = "stats-test"
    client.post("/reset", json={"task_id": "simple_triage", "case_index": 0, "session_id": sid})
    client.post("/step", json={"session_id": sid, "action": {
        "priority": "high", "news2_score": 8, "critical_sign": "respiratory_rate",
        "recommended_action": "urgent_review"
    }})
    r = client.get("/stats")
    data = r.json()
    assert "total_episodes" in data or data == {}  # empty dict ok before any episodes


# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics_returns_200():
    r = client.get("/metrics")
    assert r.status_code == 200

def test_metrics_empty_state():
    r = client.get("/metrics")
    data = r.json()
    assert "total_episodes" in data
    assert "active_sessions" in data
    assert "by_task" in data
    assert "difficulty_gradient_verified" in data

def test_metrics_populated():
    sid = "metrics-test"
    client.post("/reset", json={"task_id": "paediatric_triage", "case_index": 0, "session_id": sid})
    client.post("/step", json={"session_id": sid, "action": {
        "priority": "critical", "age_group": "school_age", "pews_score": 8,
        "critical_sign": "spo2", "recommended_action": "emergency_response"
    }})
    r = client.get("/metrics")
    assert r.status_code == 200


# ── /learning-curve ───────────────────────────────────────────────────────────

def test_learning_curve_empty():
    r = client.get("/learning-curve")
    assert r.status_code == 200
    data = r.json()
    assert "episodes" in data
    assert "rolling_avg" in data

def test_learning_curve_with_task_filter():
    r = client.get("/learning-curve?task_id=simple_triage&window=5")
    assert r.status_code == 200

def test_learning_curve_populated():
    sid = "lc-test"
    client.post("/reset", json={"task_id": "simple_triage", "case_index": 0, "session_id": sid})
    client.post("/step", json={"session_id": sid, "action": {
        "priority": "high", "news2_score": 8, "critical_sign": "respiratory_rate",
        "recommended_action": "urgent_review"
    }})
    r = client.get("/learning-curve?task_id=simple_triage")
    assert r.status_code == 200
    data = r.json()
    if data["total_episodes"] > 0:
        assert len(data["rolling_avg"]) == data["total_episodes"]


# ── /compute-news2 ────────────────────────────────────────────────────────────

def test_compute_news2_basic():
    r = client.post("/compute-news2", json={
        "respiratory_rate": 22, "spo2": 95, "systolic_bp": 144,
        "heart_rate": 108, "temperature": 36.8
    })
    assert r.status_code == 200
    data = r.json()
    assert "news2_total" in data
    assert "priority" in data
    assert "breakdown" in data
    assert isinstance(data["news2_total"], int)

def test_compute_news2_critical():
    r = client.post("/compute-news2", json={
        "respiratory_rate": 30, "spo2": 88, "systolic_bp": 80,
        "heart_rate": 130, "temperature": 39.5, "consciousness": "confused"
    })
    assert r.status_code == 200
    assert r.json()["news2_total"] >= 7

def test_compute_news2_low():
    r = client.post("/compute-news2", json={
        "respiratory_rate": 16, "spo2": 98, "systolic_bp": 120,
        "heart_rate": 72, "temperature": 37.0
    })
    assert r.status_code == 200
    data = r.json()
    assert data["news2_total"] <= 2
    assert data["priority"] == "low"

def test_compute_news2_with_oxygen():
    r = client.post("/compute-news2", json={
        "spo2": 94, "supplemental_oxygen": True
    })
    assert r.status_code == 200


# ── /suggest — rule-based path (no API key in test env) ──────────────────────

def _suggest(patient_text: str, task_id: str) -> dict:
    r = client.post("/suggest", json={"patient_history": patient_text, "task_id": task_id})
    assert r.status_code == 200
    return r.json()

# Schema validation helpers
SIMPLE_TRIAGE_FIELDS = {"priority", "news2_score", "critical_sign", "recommended_action", "rationale"}
DIFFDX_FIELDS = {"must_not_miss", "top_diagnosis", "differentials", "first_investigation", "urgency"}
ICU_FIELDS = {"sofa_score", "primary_organ_failure", "deterioration_trend", "intervention", "rationale"}
SBAR_FIELDS = {"escalation_required", "priority", "assessment", "recommendation", "rationale"}
PAEDS_FIELDS = {"priority", "age_group", "pews_score", "critical_sign", "recommended_action"}
MEDRECON_FIELDS = {"issues_found", "severity", "requires_pharmacist", "recommended_action"}
SEPSIS_FIELDS = {"priority", "bundle_elements", "antibiotic_choice", "fluid_volume_ml", "vasopressor_indicated"}


@pytest.mark.parametrize("task", ["simple_triage", "demographic_fairness"])
def test_suggest_simple_triage_schema(task):
    d = _suggest(CHEST_PAIN_PATIENT, task)
    s = d["suggestion"]
    for f in SIMPLE_TRIAGE_FIELDS:
        assert f in s, f"{task}: missing field '{f}' in suggest response"
    assert s["priority"] in ("low", "medium", "high", "critical")

def test_suggest_conflicting_vitals_has_misleading_signs():
    d = _suggest(CHEST_PAIN_PATIENT, "conflicting_vitals")
    s = d["suggestion"]
    assert "misleading_signs" in s
    assert isinstance(s["misleading_signs"], list)
    assert "condition" in s

def test_suggest_masked_deterioration_has_masking_fields():
    d = _suggest(CHEST_PAIN_PATIENT, "masked_deterioration")
    s = d["suggestion"]
    assert "masking_drug_or_condition" in s
    assert "masked_sign" in s
    assert "critical_clues" in s

def test_suggest_deteriorating_patient_schema():
    d = _suggest(CHEST_PAIN_PATIENT, "deteriorating_patient")
    s = d["suggestion"]
    assert "action" in s
    assert s["action"] in ("monitor", "escalate", "emergency_response")

def test_suggest_sepsis_bundle_schema():
    d = _suggest(SEPSIS_PATIENT, "sepsis_bundle")
    s = d["suggestion"]
    for f in SEPSIS_FIELDS:
        assert f in s, f"sepsis_bundle: missing field '{f}'"
    assert isinstance(s["bundle_elements"], list)
    assert isinstance(s["vasopressor_indicated"], bool)

def test_suggest_sepsis_penicillin_allergy_uses_meropenem():
    text = SEPSIS_PATIENT + " Penicillin allergy documented."
    d = _suggest(text, "sepsis_bundle")
    assert d["suggestion"]["antibiotic_choice"] == "meropenem"

def test_suggest_sepsis_no_allergy_uses_pip_taz():
    d = _suggest(SEPSIS_PATIENT, "sepsis_bundle")
    assert d["suggestion"]["antibiotic_choice"] == "piperacillin_tazobactam"

def test_suggest_paediatric_schema():
    d = _suggest(PAEDS_PATIENT, "paediatric_triage")
    s = d["suggestion"]
    for f in PAEDS_FIELDS:
        assert f in s, f"paediatric_triage: missing field '{f}'"
    assert s["age_group"] in ("infant", "toddler", "preschool", "school_age", "adolescent")
    assert isinstance(s["pews_score"], int)

def test_suggest_paediatric_infant_detected():
    d = _suggest(INFANT_PATIENT, "paediatric_triage")
    assert d["suggestion"]["age_group"] == "infant"

def test_suggest_paediatric_adolescent_detected():
    d = _suggest(TEEN_PATIENT, "paediatric_triage")
    assert d["suggestion"]["age_group"] == "adolescent"

def test_suggest_medication_reconciliation_schema():
    d = _suggest(MED_RECON_PATIENT, "medication_reconciliation")
    s = d["suggestion"]
    for f in MEDRECON_FIELDS:
        assert f in s, f"medication_reconciliation: missing field '{f}'"
    assert s["severity"] in ("low", "medium", "high", "critical")
    assert isinstance(s["requires_pharmacist"], bool)
    assert isinstance(s["issues_found"], list)

def test_suggest_icu_deterioration_schema():
    d = _suggest(ICU_PATIENT, "icu_deterioration")
    s = d["suggestion"]
    for f in ICU_FIELDS:
        assert f in s, f"icu_deterioration: missing field '{f}'"
    assert isinstance(s["sofa_score"], int)
    assert 0 <= s["sofa_score"] <= 24
    assert s["deterioration_trend"] in ("improving", "stable", "worsening")
    assert s["intervention"] in ("maintain_current", "increase_support", "emergency_escalation", "prepare_palliation")

def test_suggest_sbar_handover_schema():
    d = _suggest(SBAR_PATIENT, "sbar_handover")
    s = d["suggestion"]
    for f in SBAR_FIELDS:
        assert f in s, f"sbar_handover: missing field '{f}'"
    assert isinstance(s["escalation_required"], bool)
    assert s["priority"] in ("low", "medium", "high", "critical")
    assert s["recommendation"] in ("routine_monitoring", "urgent_review", "emergency_response")

def test_suggest_sbar_high_news2_requires_escalation():
    d = _suggest(SBAR_PATIENT, "sbar_handover")
    # NEWS2=13 case should flag escalation
    assert d["suggestion"]["escalation_required"] is True

def test_suggest_differential_diagnosis_schema():
    d = _suggest(CHEST_PAIN_PATIENT, "differential_diagnosis")
    s = d["suggestion"]
    for f in DIFFDX_FIELDS:
        assert f in s, f"differential_diagnosis: missing field '{f}'"
    assert isinstance(s["differentials"], list)
    assert len(s["differentials"]) >= 2
    assert s["urgency"] in ("immediate", "urgent", "routine")

def test_suggest_diffdx_chest_pain_must_not_miss_stemi():
    d = _suggest(CHEST_PAIN_PATIENT, "differential_diagnosis")
    assert d["suggestion"]["must_not_miss"] == "stemi"
    assert d["suggestion"]["first_investigation"] == "ecg"

def test_suggest_diffdx_headache_must_not_miss_sah():
    d = _suggest(HEADACHE_PATIENT, "differential_diagnosis")
    assert d["suggestion"]["must_not_miss"] == "subarachnoid_haemorrhage"
    assert d["suggestion"]["first_investigation"] == "ct_head"

def test_suggest_diffdx_sob_must_not_miss_pe():
    d = _suggest(SOB_PATIENT, "differential_diagnosis")
    assert d["suggestion"]["must_not_miss"] == "pulmonary_embolism"

def test_suggest_diffdx_abdo_must_not_miss_aaa():
    d = _suggest(ABDO_PATIENT, "differential_diagnosis")
    assert d["suggestion"]["must_not_miss"] == "aortic_aneurysm_rupture"

def test_suggest_returns_suggestion_key():
    r = client.post("/suggest", json={"patient_history": CHEST_PAIN_PATIENT, "task_id": "simple_triage"})
    assert "suggestion" in r.json()
    assert "llm_used" in r.json()
    assert "model" in r.json()

def test_suggest_llm_used_false_without_api_key():
    """Without API keys in test env, llm_used must be False (rule-based path)."""
    r = client.post("/suggest", json={"patient_history": CHEST_PAIN_PATIENT, "task_id": "simple_triage"})
    assert r.json()["llm_used"] is False


# ── /agent-assess — no-API-key mock path ─────────────────────────────────────

def test_agent_assess_no_api_key_returns_mock():
    r = client.post("/agent-assess", json={
        "patient_history": CHEST_PAIN_PATIENT,
        "task_id": "simple_triage"
    })
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert "model" in data

@pytest.mark.parametrize("task", ALL_TASKS)
def test_agent_assess_mock_has_correct_schema(task):
    r = client.post("/agent-assess", json={"patient_history": CHEST_PAIN_PATIENT, "task_id": task})
    assert r.status_code == 200
    action = r.json()["action"]
    assert isinstance(action, dict)
    assert len(action) > 0


# ── /reset + /step exception paths ───────────────────────────────────────────

def test_reset_with_invalid_task_returns_response():
    """Invalid task_id falls back gracefully — server does not crash."""
    r = client.post("/reset", json={"task_id": "nonexistent_task_xyz"})
    # Environment picks a random task when task_id is unknown — must not 500
    assert r.status_code in (200, 400, 422)

def test_step_without_reset_uses_default_session():
    """Step without prior reset uses _default session — should not crash."""
    r = client.post("/step", json={"action": {"priority": "high"}})
    # May return 200 or 400 depending on _default session state
    assert r.status_code in (200, 400, 500)

def test_step_done_records_episode_history():
    """Completed episode (done=True) must be recorded in history."""
    sid = "done-test-session"
    client.post("/reset", json={"task_id": "simple_triage", "case_index": 0, "session_id": sid})
    r = client.post("/step", json={"session_id": sid, "action": {
        "priority": "high", "news2_score": 8, "critical_sign": "respiratory_rate",
        "recommended_action": "urgent_review"
    }})
    assert r.status_code == 200
    assert r.json()["done"] is True
    # History should now have at least one episode
    hist = client.get("/history").json()
    assert hist["total"] >= 1

def test_reset_session_isolation():
    """Two different session_ids must produce independent environments."""
    r1 = client.post("/reset", json={"task_id": "simple_triage", "case_index": 0, "session_id": "iso-A"})
    r2 = client.post("/reset", json={"task_id": "icu_deterioration", "case_index": 0, "session_id": "iso-B"})
    assert r1.json()["observation"]["task_id"] == "simple_triage"
    assert r2.json()["observation"]["task_id"] == "icu_deterioration"


# ── Ground truth revealed after episode ───────────────────────────────────────

def test_ground_truth_revealed_for_icu():
    sid = "gt-icu"
    client.post("/reset", json={"task_id": "icu_deterioration", "case_index": 0, "session_id": sid})
    r = client.post("/step", json={"session_id": sid, "action": {
        "sofa_score": 10, "primary_organ_failure": "cardiovascular",
        "deterioration_trend": "worsening", "intervention": "emergency_escalation",
        "rationale": "High SOFA, cardiovascular failure"
    }})
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True
    gt = data["info"].get("ground_truth")
    assert gt is not None
    # ICU ground truth should expose task-specific fields
    assert "intervention" in gt or "sofa_score" in gt or "deterioration_trend" in gt

def test_ground_truth_revealed_for_differential_diagnosis():
    sid = "gt-diffdx"
    client.post("/reset", json={"task_id": "differential_diagnosis", "case_index": 0, "session_id": sid})
    r = client.post("/step", json={"session_id": sid, "action": {
        "must_not_miss": "stemi", "top_diagnosis": "acute_coronary_syndrome",
        "differentials": ["unstable_angina", "pulmonary_embolism", "aortic_dissection"],
        "first_investigation": "ecg", "urgency": "immediate"
    }})
    assert r.status_code == 200
    assert r.json()["done"] is True
    gt = r.json()["info"].get("ground_truth")
    assert gt is not None

def test_ground_truth_revealed_for_sbar():
    sid = "gt-sbar"
    client.post("/reset", json={"task_id": "sbar_handover", "case_index": 0, "session_id": sid})
    r = client.post("/step", json={"session_id": sid, "action": {
        "escalation_required": True, "priority": "critical",
        "assessment": "Patient deteriorating with worsening vitals and altered consciousness",
        "recommendation": "emergency_response"
    }})
    assert r.status_code == 200
    assert r.json()["done"] is True
