"""
tests/test_graders.py — Exhaustive Grader Unit Tests
======================================================
Tests every grader function with:
  - Correct responses (should score 1.0 or close)
  - Partially correct responses (should score 0.3-0.7)
  - Completely wrong responses (should score 0.0)
  - Edge cases (empty, null, wrong types)

Run with: pytest tests/test_graders.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import TASK_SCORE_OPEN_EPS
from server.graders import (
    compute_news2,
    news2_to_priority,
    priority_distance,
    grade_simple_triage,
    grade_conflicting_vitals,
    grade_masked_deterioration,
    grade_sepsis_bundle,
    grade_paediatric_triage,
    grade_medication_reconciliation,
    grade_response,
    grade_response_raw,
    _normalize_vital_sign,
    _normalize_condition,
)
from server.cases import (
    SIMPLE_TRIAGE_CASES, CONFLICTING_VITALS_CASES, MASKED_DETERIORATION_CASES,
    SEPSIS_BUNDLE_CASES, PAEDIATRIC_TRIAGE_CASES, MEDICATION_RECONCILIATION_CASES,
)


# ─────────────────────────────────────────────────────────────
# NEWS2 COMPUTATION TESTS
# ─────────────────────────────────────────────────────────────

class TestNEWS2Computation:
    """Tests for the core NEWS2 scoring algorithm."""

    def test_all_normal_vitals_scores_zero(self):
        """Patient with perfect vitals should score 0."""
        vitals = {
            "respiratory_rate": 16,
            "spo2": 98,
            "systolic_bp": 120,
            "heart_rate": 72,
            "temperature": 37.0,
            "consciousness": "alert"
        }
        score, breakdown = compute_news2(vitals)
        assert score == 0, f"Expected 0, got {score}"
        assert all(v == 0 for v in breakdown.values()), f"All should be 0: {breakdown}"

    def test_critical_patient_scores_high(self):
        """Patient with multiple abnormal vitals should score high."""
        vitals = {
            "respiratory_rate": 30,   # =3
            "spo2": 88,               # =3
            "systolic_bp": 85,        # =3
            "heart_rate": 135,        # =3
            "temperature": 34.0,      # =3
            "consciousness": "voice"  # =3
        }
        score, breakdown = compute_news2(vitals)
        assert score == 18, f"Expected 18, got {score}"
        assert all(v == 3 for v in breakdown.values()), f"All should be 3: {breakdown}"

    def test_respiratory_rate_boundaries(self):
        """Test RR scoring at each boundary."""
        test_cases = [
            (8, 3), (9, 1), (11, 1), (12, 0), (20, 0),
            (21, 2), (24, 2), (25, 3), (35, 3)
        ]
        for rr, expected in test_cases:
            vitals = {"respiratory_rate": rr, "spo2": 98, "systolic_bp": 120,
                     "heart_rate": 72, "temperature": 37.0, "consciousness": "alert"}
            _, breakdown = compute_news2(vitals)
            assert breakdown["respiratory_rate"] == expected, \
                f"RR={rr}: expected {expected}, got {breakdown['respiratory_rate']}"

    def test_spo2_boundaries(self):
        """Test SpO2 scoring at each boundary."""
        test_cases = [(91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (99, 0)]
        for spo2, expected in test_cases:
            vitals = {"respiratory_rate": 16, "spo2": spo2, "systolic_bp": 120,
                     "heart_rate": 72, "temperature": 37.0, "consciousness": "alert"}
            _, breakdown = compute_news2(vitals)
            assert breakdown["spo2"] == expected, \
                f"SpO2={spo2}: expected {expected}, got {breakdown['spo2']}"

    def test_consciousness_alert_vs_other(self):
        """Alert=0, anything else=3."""
        for val, expected in [("alert", 0), ("voice", 3), ("pain", 3), ("unresponsive", 3), ("confused", 3)]:
            vitals = {"respiratory_rate": 16, "spo2": 98, "systolic_bp": 120,
                     "heart_rate": 72, "temperature": 37.0, "consciousness": val}
            _, breakdown = compute_news2(vitals)
            assert breakdown["consciousness"] == expected, \
                f"Consciousness='{val}': expected {expected}, got {breakdown['consciousness']}"

    def test_news2_to_priority_thresholds(self):
        """NEWS2 → priority conversion at key thresholds (no red flags)."""
        flat_scores = {"respiratory_rate": 0, "spo2": 0, "systolic_bp": 0,
                      "heart_rate": 0, "temperature": 0, "consciousness": 0}
        assert news2_to_priority(0, flat_scores) == "low"
        assert news2_to_priority(2, flat_scores) == "low"
        assert news2_to_priority(3, flat_scores) == "medium"
        assert news2_to_priority(6, flat_scores) == "high"
        # NEWS2 7-8 with no haemodynamic red flag = "high" (not "critical")
        # Consistent with ST001 (NEWS2=8, no BP/HR=3, expected="high")
        assert news2_to_priority(7, flat_scores) == "high"
        assert news2_to_priority(8, flat_scores) == "high"
        # NEWS2 ≥ 9 always = "critical"
        assert news2_to_priority(9, flat_scores) == "critical"
        assert news2_to_priority(12, flat_scores) == "critical"

    def test_news2_to_priority_haemodynamic_red_flag(self):
        """NEWS2 ≥ 7 WITH a haemodynamic red flag (BP or HR=3) = critical."""
        # BP=3 (systolic ≤ 90) — haemodynamic collapse
        bp_red_flag = {"respiratory_rate": 2, "spo2": 1, "systolic_bp": 3,
                       "heart_rate": 2, "temperature": 0, "consciousness": 0}
        # Total = 8, has haemodynamic red flag → "critical"
        assert news2_to_priority(8, bp_red_flag) == "critical"
        # Consistent with ST003 (NEWS2=8, BP=88→3, expected="critical")

    def test_red_flag_overrides_total(self):
        """Any single parameter score of 3 should trigger at least 'high'."""
        # AVPU=3 (consciousness) — not a haemodynamic flag — gives "high" not "critical"
        avpu_red_flag = {"respiratory_rate": 0, "spo2": 0, "systolic_bp": 0,
                        "heart_rate": 0, "temperature": 0, "consciousness": 3}
        result_low_news2 = news2_to_priority(3, avpu_red_flag)
        assert result_low_news2 in ("high", "critical"), \
            f"AVPU red flag at low NEWS2 should give high/critical, got {result_low_news2}"
        # CV003 pattern: NEWS2=7, AVPU=3 (not haemodynamic) → "high"
        result_news2_7 = news2_to_priority(7, avpu_red_flag)
        assert result_news2_7 == "high", \
            f"NEWS2=7 with AVPU red flag only (no haemodynamic collapse) should be 'high', got {result_news2_7}"


class TestPriorityDistance:
    def test_exact_match(self):
        assert priority_distance("low", "low") == 1.0
        assert priority_distance("critical", "critical") == 1.0

    def test_one_level_off(self):
        assert priority_distance("low", "medium") == 0.5
        assert priority_distance("high", "critical") == 0.5

    def test_two_or_more_levels_off(self):
        assert priority_distance("low", "critical") == 0.0
        assert priority_distance("low", "high") == 0.0

    def test_invalid_priority(self):
        assert priority_distance("unknown", "low") == 0.0
        assert priority_distance("", "critical") == 0.0


# ─────────────────────────────────────────────────────────────
# TASK 1 GRADER TESTS: Simple Triage
# ─────────────────────────────────────────────────────────────

class TestSimpleTriageGrader:

    def test_perfect_response_scores_1(self):
        """A clinically perfect response should score at or near 1.0."""
        case = SIMPLE_TRIAGE_CASES[0]  # ST001: high priority, news2=8
        perfect_response = {
            "priority": "high",
            "news2_score": 8,
            "critical_sign": "respiratory_rate",
            "recommended_action": "urgent_review",
            "rationale": "NEWS2=8, RR and SpO2 elevated, tachycardia present"
        }
        score, breakdown = grade_simple_triage(perfect_response, case)
        assert score >= 0.90, f"Perfect response should score ≥0.90, got {score}. Breakdown: {breakdown}"

    def test_correct_priority_wrong_news2(self):
        """Correct priority but NEWS2 off by 3 → partial credit."""
        case = SIMPLE_TRIAGE_CASES[0]  # high, news2=8
        response = {
            "priority": "high",
            "news2_score": 5,   # Off by 3 — no NEWS2 credit
            "critical_sign": "respiratory_rate",
            "recommended_action": "urgent_review"
        }
        score, breakdown = grade_simple_triage(response, case)
        assert 0.50 < score < 0.85, f"Partial response should score 0.50-0.85, got {score}"
        assert breakdown["news2_score"] == 0.0, "NEWS2 off by 3 should score 0"
        assert breakdown["priority"] > 0, "Priority was correct"

    def test_wrong_priority_penalized(self):
        """Calling 'critical' a 'low' patient should score very low."""
        case = SIMPLE_TRIAGE_CASES[1]  # ST002: LOW priority
        response = {
            "priority": "critical",
            "news2_score": 0,
            "critical_sign": "none",
            "recommended_action": "emergency_response"
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score < 0.55, f"Wrong priority + wrong action should score <0.55, got {score}"

    def test_news2_within_1_gets_partial_credit(self):
        """NEWS2 off by 1 should still get 70% of that dimension's credit."""
        case = SIMPLE_TRIAGE_CASES[0]  # news2=8
        response = {
            "priority": "high",
            "news2_score": 7,   # Off by 1
            "critical_sign": "respiratory_rate",
            "recommended_action": "urgent_review"
        }
        score, breakdown = grade_simple_triage(response, case)
        expected_news2_credit = round(0.7 * 0.25, 3)
        assert breakdown["news2_score"] == expected_news2_credit, \
            f"Off-by-1 NEWS2 should score {expected_news2_credit}, got {breakdown['news2_score']}"

    def test_critical_case_correct(self):
        """ST003 is critical — agent must respond with critical."""
        case = SIMPLE_TRIAGE_CASES[2]  # ST003: critical, BP=88
        response = {
            "priority": "critical",
            "news2_score": 9,
            "critical_sign": "systolic_bp",
            "recommended_action": "emergency_response"
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score >= 0.85, f"Correct critical case should score ≥0.85, got {score}"

    def test_empty_response_scores_zero(self):
        """Empty response should score 0."""
        case = SIMPLE_TRIAGE_CASES[0]
        score, _ = grade_simple_triage({}, case)
        assert score == 0.0

    def test_all_simple_triage_cases(self):
        """All 4 simple triage cases should be gradeable."""
        for case in SIMPLE_TRIAGE_CASES:
            gt = case["ground_truth"]
            # Build perfect response from ground truth
            response = {
                "priority": case["expected_priority"],
                "news2_score": case["news2_score"],
                "critical_sign": gt["critical_sign"],
                "recommended_action": gt["recommended_action"]
            }
            score, breakdown = grade_simple_triage(response, case)
            assert score >= 0.80, \
                f"Case {case['case_id']}: perfect response scored {score}. Breakdown: {breakdown}"


# ─────────────────────────────────────────────────────────────
# TASK 2 GRADER TESTS: Conflicting Vitals
# ─────────────────────────────────────────────────────────────

class TestConflictingVitalsGrader:

    def test_correct_response_scores_high(self):
        """Correct identification of critical sign despite misleading normals."""
        case = CONFLICTING_VITALS_CASES[0]  # CV001: spo2 critical, HR misleading
        response = {
            "priority": "critical",
            "critical_sign": "spo2",
            "misleading_signs": ["heart_rate", "systolic_bp"],
            "condition": "silent_hypoxia",
            "recommended_action": "emergency_response",
            "rationale": "SpO2=88% and RR=28 are critical despite normal HR. Confusion indicates cerebral hypoxia."
        }
        score, breakdown = grade_conflicting_vitals(response, case)
        assert score >= 0.80, f"Correct conflicting vitals response should score ≥0.80, got {score}"

    def test_falls_for_trap_scores_zero_on_critical_sign(self):
        """Agent that picks the misleading 'normal' sign as critical should score 0 on that dim."""
        case = CONFLICTING_VITALS_CASES[0]  # spo2 is critical, HR is misleading
        response = {
            "priority": "low",
            "critical_sign": "heart_rate",   # This is the TRAP — HR is normal and misleading
            "misleading_signs": [],
            "condition": "anxiety",
            "recommended_action": "routine_monitoring",
            "rationale": "HR normal, no acute concern"
        }
        score, breakdown = grade_conflicting_vitals(response, case)
        assert breakdown["critical_sign"] == 0.0, \
            f"Falling for the trap should give 0 on critical_sign, got {breakdown['critical_sign']}"
        assert score < 0.30, f"Trap response should score <0.30, got {score}"

    def test_partially_correct_gets_partial_credit(self):
        """Right priority but wrong critical sign → partial credit."""
        case = CONFLICTING_VITALS_CASES[0]
        response = {
            "priority": "critical",
            "critical_sign": "respiratory_rate",  # Close but not the primary killer
            "misleading_signs": ["heart_rate"],
            "recommended_action": "emergency_response",
            "rationale": "RR very high, patient deteriorating"
        }
        score, breakdown = grade_conflicting_vitals(response, case)
        assert 0.35 < score < 0.80, f"Partial credit expected, got {score}"
        assert breakdown["priority"] > 0, "Priority was correct (critical)"

    def test_all_conflicting_vitals_cases_gradeable(self):
        """All 3 conflicting vitals cases should be gradeable with perfect responses."""
        for case in CONFLICTING_VITALS_CASES:
            gt = case["ground_truth"]
            response = {
                "priority": case["expected_priority"],
                "critical_sign": gt["critical_sign"],
                "misleading_signs": gt.get("misleading_signs", []),
                "condition": gt.get("condition", ""),
                "recommended_action": gt["recommended_action"],
                "rationale": gt["rationale"]
            }
            score, _ = grade_conflicting_vitals(response, case)
            assert score >= 0.75, \
                f"Case {case['case_id']}: perfect response scored {score}"


# ─────────────────────────────────────────────────────────────
# TASK 3 GRADER TESTS: Masked Deterioration
# ─────────────────────────────────────────────────────────────

class TestMaskedDeteriorationGrader:

    def test_correct_response_scores_high(self):
        """Agent that catches beta-blocker masking should score high."""
        case = MASKED_DETERIORATION_CASES[0]  # MD001: bisoprolol masking
        response = {
            "priority": "critical",
            "masking_drug_or_condition": "bisoprolol",
            "masked_sign": "heart_rate",
            "critical_clues": ["lactate", "urine_output_reduced", "hypotension"],
            "condition": "septic_shock",
            "recommended_action": "emergency_response",
            "rationale": (
                "Beta-blocker bisoprolol suppresses reflex tachycardia. "
                "HR=68 is pharmacologically blunted. True severity revealed by "
                "lactate=3.2, hypotension, and reduced urine output."
            )
        }
        score, breakdown = grade_masked_deterioration(response, case)
        assert score >= 0.80, f"Correct masked deterioration should score ≥0.80, got {score}\n{breakdown}"

    def test_misled_by_low_news2_penalized(self):
        """Agent saying 'low' priority because NEWS2=8 should be penalized hard."""
        case = MASKED_DETERIORATION_CASES[0]
        response = {
            "priority": "low",   # WRONG — should be critical
            "news2_score": 8,
            "masking_drug_or_condition": "",
            "masked_sign": "",
            "critical_clues": [],
            "recommended_action": "routine_monitoring",
            "rationale": "NEWS2=8, some concern but BP may be artifact"
        }
        score, breakdown = grade_masked_deterioration(response, case)
        # Priority is wrong (not critical) → gets penalized with 0.5x
        assert breakdown["priority"] < 0.15, \
            f"Missing critical priority should score <0.15, got {breakdown['priority']}"
        assert score < 0.30, f"Misled agent should score <0.30, got {score}"

    def test_steroid_masking_case(self):
        """Agent catches steroid-masked peritonitis in MD002."""
        case = MASKED_DETERIORATION_CASES[1]
        response = {
            "priority": "critical",
            "masking_drug_or_condition": "prednisolone",
            "masked_sign": "temperature",
            "critical_clues": ["steroid_use", "immunosuppression", "age"],
            "condition": "peritonitis",
            "recommended_action": "emergency_response",
            "rationale": "Steroids mask fever and peritoneal signs. NEWS2=1 is misleading in immunosuppressed patient."
        }
        score, breakdown = grade_masked_deterioration(response, case)
        assert score >= 0.70, f"Steroid masking detection should score ≥0.70, got {score}"

    def test_silent_mi_case(self):
        """Agent catches silent MI in diabetic autonomic neuropathy case (MD003)."""
        case = MASKED_DETERIORATION_CASES[2]
        response = {
            "priority": "critical",
            "masking_drug_or_condition": "diabetic_autonomic_neuropathy",
            "masked_sign": "chest_pain",
            "critical_clues": ["ecg_changes", "diabetes_history", "troponin_pending"],
            "condition": "myocardial_infarction",
            "recommended_action": "emergency_response",
            "rationale": "Autonomic neuropathy prevents classic MI symptoms. ECG changes override normal vitals."
        }
        score, breakdown = grade_masked_deterioration(response, case)
        assert score >= 0.65, f"Silent MI detection should score ≥0.65, got {score}"

    def test_partial_clues_gives_partial_credit(self):
        """Catching 2 of 4 critical clues should give 50% of that dimension."""
        case = MASKED_DETERIORATION_CASES[0]  # 4 critical clues
        response = {
            "priority": "critical",
            "masking_drug_or_condition": "bisoprolol",
            "masked_sign": "heart_rate",
            "critical_clues": ["lactate", "urine_output_reduced"],  # 2/4 clues
            "recommended_action": "emergency_response",
            "rationale": "Bisoprolol masking. Elevated lactate and reduced UO indicate sepsis."
        }
        score, breakdown = grade_masked_deterioration(response, case)
        # 2/4 clues = 50% of 0.20 = 0.10
        assert 0.08 <= breakdown["critical_clues"] <= 0.12, \
            f"2/4 clues should give ~0.10, got {breakdown['critical_clues']}"


# ─────────────────────────────────────────────────────────────
# MASTER DISPATCH TESTS
# ─────────────────────────────────────────────────────────────

class TestGradeDispatch:

    def test_correct_task_routing(self):
        """grade_response routes to correct grader per task_id."""
        for case in SIMPLE_TRIAGE_CASES[:1]:
            score, _ = grade_response("simple_triage", {"priority": "low"}, case)
            assert isinstance(score, float)

        for case in CONFLICTING_VITALS_CASES[:1]:
            score, _ = grade_response("conflicting_vitals", {"priority": "low"}, case)
            assert isinstance(score, float)

        for case in MASKED_DETERIORATION_CASES[:1]:
            score, _ = grade_response("masked_deterioration", {"priority": "low"}, case)
            assert isinstance(score, float)

    def test_unknown_task_returns_zero(self):
        """Unknown task_id: raw dispatch is 0.0; public grade_response maps to open-interval floor."""
        raw, breakdown = grade_response_raw("unknown_task", {}, SIMPLE_TRIAGE_CASES[0])
        assert raw == 0.0
        assert "error" in breakdown
        score, _ = grade_response("unknown_task", {}, SIMPLE_TRIAGE_CASES[0])
        assert score == TASK_SCORE_OPEN_EPS

    def test_score_always_in_range(self):
        """grade_response() scores are strictly in (0, 1) for hackathon validators."""
        for task_id, cases in [
            ("simple_triage", SIMPLE_TRIAGE_CASES),
            ("conflicting_vitals", CONFLICTING_VITALS_CASES),
            ("masked_deterioration", MASKED_DETERIORATION_CASES),
        ]:
            for case in cases:
                for response in [{}, {"priority": "low"}, {"priority": "critical", "news2_score": 15}]:
                    score, _ = grade_response(task_id, response, case)
                    assert TASK_SCORE_OPEN_EPS <= score <= 1.0 - TASK_SCORE_OPEN_EPS, \
                        f"{task_id}/{case['case_id']}: score {score} out of (0,1)"


# ─────────────────────────────────────────────────────────────
# SYNONYM NORMALIZATION TESTS
# Verify that clinically equivalent agent answers are accepted.
# All tests must be purely additive — existing golden tests unchanged.
# ─────────────────────────────────────────────────────────────

class TestSynonymNormalization:
    """Unit tests for _normalize_vital_sign and _normalize_condition."""

    def test_vital_sign_exact_passthrough(self):
        """Canonical names pass through unchanged."""
        assert _normalize_vital_sign("respiratory_rate") == "respiratory_rate"
        assert _normalize_vital_sign("heart_rate") == "heart_rate"
        assert _normalize_vital_sign("spo2") == "spo2"
        assert _normalize_vital_sign("systolic_bp") == "systolic_bp"
        assert _normalize_vital_sign("temperature") == "temperature"
        assert _normalize_vital_sign("consciousness") == "consciousness"

    def test_vital_sign_common_synonyms(self):
        """Common LLM synonym outputs normalise to canonical form."""
        assert _normalize_vital_sign("tachycardia") == "heart_rate"
        assert _normalize_vital_sign("bradycardia") == "heart_rate"
        assert _normalize_vital_sign("hr") == "heart_rate"
        assert _normalize_vital_sign("heart rate") == "heart_rate"
        assert _normalize_vital_sign("oxygen saturation") == "spo2"
        assert _normalize_vital_sign("o2 sat") == "spo2"
        assert _normalize_vital_sign("pulse ox") == "spo2"
        assert _normalize_vital_sign("respiratory rate") == "respiratory_rate"
        assert _normalize_vital_sign("rr") == "respiratory_rate"
        assert _normalize_vital_sign("resp rate") == "respiratory_rate"
        assert _normalize_vital_sign("blood pressure") == "systolic_bp"
        assert _normalize_vital_sign("bp") == "systolic_bp"
        assert _normalize_vital_sign("hypotension") == "systolic_bp"
        assert _normalize_vital_sign("fever") == "temperature"
        assert _normalize_vital_sign("temp") == "temperature"
        assert _normalize_vital_sign("confusion") == "consciousness"
        assert _normalize_vital_sign("altered consciousness") == "consciousness"
        assert _normalize_vital_sign("gcs") == "consciousness"

    def test_vital_sign_case_insensitive(self):
        """Normalization is case-insensitive."""
        assert _normalize_vital_sign("TACHYCARDIA") == "heart_rate"
        assert _normalize_vital_sign("Oxygen Saturation") == "spo2"
        assert _normalize_vital_sign("Respiratory Rate") == "respiratory_rate"

    def test_vital_sign_unknown_passthrough(self):
        """Unknown strings pass through cleaned (no crash)."""
        result = _normalize_vital_sign("some_unknown_sign")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_condition_synonyms(self):
        """Condition synonym normalization."""
        assert _normalize_condition("sepsis") == "septic_shock"
        assert _normalize_condition("septicaemia") == "septic_shock"
        assert _normalize_condition("dka") == "diabetic_ketoacidosis"
        assert _normalize_condition("diabetic ketoacidosis") == "diabetic_ketoacidosis"
        assert _normalize_condition("pe") == "pulmonary_embolism"
        assert _normalize_condition("pulmonary embolism") == "pulmonary_embolism"
        assert _normalize_condition("stemi") == "silent_mi"
        assert _normalize_condition("myocardial infarction") == "silent_mi"
        assert _normalize_condition("hyperkalemia") == "hyperkalaemia"


class TestSynonymInGraders:
    """Integration tests: synonym normalisation applied in real grader calls."""

    def test_simple_triage_synonym_critical_sign_rr(self):
        """Agent says 'respiratory rate' (with space), GT is 'respiratory_rate' — must score full credit."""
        case = SIMPLE_TRIAGE_CASES[0]  # ST001: critical_sign = respiratory_rate
        response = {
            "priority": case["expected_priority"],
            "news2_score": case["news2_score"],
            "critical_sign": "respiratory rate",   # synonym: space instead of underscore
            "recommended_action": case["ground_truth"]["recommended_action"],
        }
        score, breakdown = grade_simple_triage(response, case)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3), \
            f"Expected full critical_sign credit for synonym 'respiratory rate', got {breakdown['critical_sign']}"

    def test_simple_triage_synonym_critical_sign_tachycardia(self):
        """Agent says 'tachycardia', GT is 'heart_rate' — must score full credit."""
        # Use a synthetic minimal case dict to test the synonym directly.
        synthetic_case = {
            "case_id": "SYN001",
            "task_id": "simple_triage",
            "expected_priority": "high",
            "news2_score": 6,
            "ground_truth": {
                "priority": "high",
                "news2_score": 6,
                "critical_sign": "heart_rate",
                "recommended_action": "urgent_review",
            },
        }
        response = {
            "priority": "high",
            "news2_score": 6,
            "critical_sign": "tachycardia",      # synonym for heart_rate
            "recommended_action": "urgent_review",
        }
        _, breakdown = grade_simple_triage(response, synthetic_case)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3), \
            f"Expected full credit for 'tachycardia' synonym, got {breakdown['critical_sign']}"

    def test_simple_triage_synonym_critical_sign_bp(self):
        """Agent says 'blood pressure' or 'hypotension', GT is 'systolic_bp' — full credit."""
        bp_case = next(c for c in SIMPLE_TRIAGE_CASES if c["ground_truth"]["critical_sign"] == "systolic_bp")
        for synonym in ["blood pressure", "hypotension", "bp"]:
            response = {
                "priority": bp_case["expected_priority"],
                "news2_score": bp_case["news2_score"],
                "critical_sign": synonym,
                "recommended_action": bp_case["ground_truth"]["recommended_action"],
            }
            _, breakdown = grade_simple_triage(response, bp_case)
            assert breakdown["critical_sign"] == round(1.0 * 0.20, 3), \
                f"Synonym '{synonym}' should give full credit, got {breakdown['critical_sign']}"

    def test_conflicting_vitals_synonym_misleading_signs(self):
        """Agent says ['tachycardia', 'blood pressure'], GT is ['heart_rate', 'systolic_bp'] — full misleading credit."""
        case = CONFLICTING_VITALS_CASES[0]  # CV001: misleading_signs = [heart_rate, systolic_bp]
        gt_misleading = case["ground_truth"].get("misleading_signs", [])
        if "heart_rate" not in gt_misleading and "systolic_bp" not in gt_misleading:
            pytest.skip("CV001 misleading_signs don't include heart_rate/systolic_bp")
        response = {
            "priority": case["expected_priority"],
            "critical_sign": case["ground_truth"]["critical_sign"],
            "misleading_signs": ["tachycardia", "blood pressure"],  # synonyms
            "recommended_action": case["ground_truth"]["recommended_action"],
            "rationale": "The SpO2 is critically low despite normal heart rate and blood pressure.",
        }
        _, breakdown = grade_conflicting_vitals(response, case)
        assert breakdown["misleading_signs"] == round(1.0 * 0.20, 3), \
            f"Synonym misleading_signs should score full credit, got {breakdown['misleading_signs']}"

    def test_conflicting_vitals_synonym_critical_sign(self):
        """Agent uses synonym for critical_sign in conflicting vitals — must get credit, not fall for trap."""
        case = CONFLICTING_VITALS_CASES[0]  # CV001: critical_sign = spo2
        response = {
            "priority": case["expected_priority"],
            "critical_sign": "oxygen saturation",  # synonym for spo2
            "misleading_signs": case["ground_truth"].get("misleading_signs", []),
            "recommended_action": case["ground_truth"]["recommended_action"],
            "rationale": "oxygen saturation is critically low",
        }
        _, breakdown = grade_conflicting_vitals(response, case)
        assert breakdown["critical_sign"] == round(1.0 * 0.25, 3), \
            f"Synonym 'oxygen saturation' for spo2 should score full credit, got {breakdown['critical_sign']}"

    def test_masked_deterioration_synonym_masked_sign(self):
        """Agent says 'heart rate' (with space), GT is 'heart_rate' — full credit."""
        case = MASKED_DETERIORATION_CASES[0]  # MD001: masked_sign = heart_rate
        if case["ground_truth"]["masked_sign"] != "heart_rate":
            pytest.skip("MD001 masked_sign is not heart_rate")
        response = {
            "priority": "critical",
            "masking_drug_or_condition": case["ground_truth"].get("masking_drug", "bisoprolol"),
            "masked_sign": "heart rate",  # synonym for heart_rate
            "critical_clues": case["ground_truth"].get("critical_clues", []),
            "recommended_action": "emergency_response",
        }
        _, breakdown = grade_masked_deterioration(response, case)
        assert breakdown["masked_sign"] == round(1.0 * 0.25, 3), \
            f"Synonym 'heart rate' for heart_rate should score full, got {breakdown['masked_sign']}"

    def test_existing_exact_matches_unchanged(self):
        """Existing exact-match golden tests still score the same after normalization."""
        case = SIMPLE_TRIAGE_CASES[0]
        gt = case["ground_truth"]
        perfect = {
            "priority": case["expected_priority"],
            "news2_score": case["news2_score"],
            "critical_sign": gt["critical_sign"],         # exact canonical string
            "recommended_action": gt["recommended_action"],
        }
        score, _ = grade_simple_triage(perfect, case)
        assert score >= 0.90, f"Perfect exact response should still score ≥0.90, got {score}"


# ─────────────────────────────────────────────────────────────
# NEW CASE TESTS (ST005–ST008, CV004–CV005)
# One perfect-response test per new case to confirm grader works.
# ─────────────────────────────────────────────────────────────

class TestNewSimpleTriageCases:
    """Verify the 4 new simple_triage cases grade correctly with perfect responses."""

    def test_st005_anaphylaxis_perfect(self):
        """ST005 — Anaphylaxis: critical, systolic_bp, emergency_response."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST005")
        response = {
            "priority": "critical",
            "news2_score": 10,
            "critical_sign": "systolic_bp",
            "recommended_action": "emergency_response",
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score >= 0.90, f"ST005 perfect response should score ≥0.90, got {score}"
        assert breakdown["priority"] == round(1.0 * 0.40, 3)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3)

    def test_st005_anaphylaxis_synonym_bp(self):
        """ST005 — Synonym 'hypotension' for systolic_bp gets full critical_sign credit."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST005")
        response = {
            "priority": "critical",
            "news2_score": 10,
            "critical_sign": "hypotension",
            "recommended_action": "emergency_response",
        }
        _, breakdown = grade_simple_triage(response, case)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3), \
            f"'hypotension' synonym should score full credit, got {breakdown['critical_sign']}"

    def test_st006_copd_perfect(self):
        """ST006 — COPD exacerbation: high, respiratory_rate, urgent_review."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST006")
        response = {
            "priority": "high",
            "news2_score": 7,
            "critical_sign": "respiratory_rate",
            "recommended_action": "urgent_review",
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score >= 0.90, f"ST006 perfect response should score ≥0.90, got {score}"
        assert breakdown["priority"] == round(1.0 * 0.40, 3)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3)

    def test_st007_stroke_perfect(self):
        """ST007 — Stroke: high, consciousness, emergency_response."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST007")
        response = {
            "priority": "high",
            "news2_score": 3,
            "critical_sign": "consciousness",
            "recommended_action": "emergency_response",
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score >= 0.90, f"ST007 perfect response should score ≥0.90, got {score}"
        assert breakdown["priority"] == round(1.0 * 0.40, 3)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3)

    def test_st007_stroke_synonym_consciousness(self):
        """ST007 — Synonym 'confusion' for consciousness gets full critical_sign credit."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST007")
        response = {
            "priority": "high",
            "news2_score": 3,
            "critical_sign": "confusion",
            "recommended_action": "emergency_response",
        }
        _, breakdown = grade_simple_triage(response, case)
        assert breakdown["critical_sign"] == round(1.0 * 0.20, 3), \
            f"'confusion' synonym should score full credit, got {breakdown['critical_sign']}"

    def test_st008_aki_perfect(self):
        """ST008 — AKI: medium, consciousness, urgent_review."""
        from server.cases import SIMPLE_TRIAGE_CASES
        case = next(c for c in SIMPLE_TRIAGE_CASES if c["case_id"] == "ST008")
        response = {
            "priority": "medium",
            "news2_score": 4,
            "critical_sign": "consciousness",
            "recommended_action": "urgent_review",
        }
        score, breakdown = grade_simple_triage(response, case)
        assert score >= 0.90, f"ST008 perfect response should score ≥0.90, got {score}"

    def test_new_cases_news2_match_computed(self):
        """All new ST cases: stored news2_score must match compute_news2()."""
        from server.cases import SIMPLE_TRIAGE_CASES
        from server.graders import compute_news2
        new_cases = [c for c in SIMPLE_TRIAGE_CASES if c["case_id"] in ("ST005", "ST006", "ST007", "ST008")]
        for case in new_cases:
            computed, _ = compute_news2(case["vitals"])
            assert computed == case["news2_score"], \
                f"{case['case_id']}: stored={case['news2_score']} computed={computed}"


class TestNewConflictingVitalsCases:
    """Verify the 2 new conflicting_vitals cases grade correctly with perfect responses."""

    def test_cv004_dka_perfect(self):
        """CV004 — DKA: high, respiratory_rate, misleading=[systolic_bp, spo2]."""
        case = next(c for c in CONFLICTING_VITALS_CASES if c["case_id"] == "CV004")
        response = {
            "priority": "high",
            "critical_sign": "respiratory_rate",
            "misleading_signs": ["systolic_bp", "spo2"],
            "recommended_action": "urgent_review",
            "rationale": "Kussmaul breathing with RR=28 is the critical sign indicating diabetic ketoacidosis metabolic acidosis compensation.",
        }
        score, breakdown = grade_conflicting_vitals(response, case)
        assert score >= 0.80, f"CV004 perfect response should score ≥0.80, got {score}"
        assert breakdown["critical_sign"] == round(1.0 * 0.25, 3)
        assert breakdown["misleading_signs"] == round(1.0 * 0.20, 3)

    def test_cv004_dka_synonym_rr(self):
        """CV004 — Synonym 'respiratory rate' (space) for respiratory_rate — full credit."""
        case = next(c for c in CONFLICTING_VITALS_CASES if c["case_id"] == "CV004")
        response = {
            "priority": "high",
            "critical_sign": "respiratory rate",
            "misleading_signs": ["blood pressure", "oxygen saturation"],
            "recommended_action": "urgent_review",
            "rationale": "Kussmaul breathing indicates metabolic acidosis from DKA.",
        }
        _, breakdown = grade_conflicting_vitals(response, case)
        assert breakdown["critical_sign"] == round(1.0 * 0.25, 3), \
            f"'respiratory rate' synonym should score full, got {breakdown['critical_sign']}"
        assert breakdown["misleading_signs"] == round(1.0 * 0.20, 3), \
            f"Synonym misleading_signs should score full, got {breakdown['misleading_signs']}"

    def test_cv005_pe_perfect(self):
        """CV005 — PE: critical, systolic_bp, misleading=[spo2], emergency_response."""
        case = next(c for c in CONFLICTING_VITALS_CASES if c["case_id"] == "CV005")
        response = {
            "priority": "critical",
            "critical_sign": "systolic_bp",
            "misleading_signs": ["spo2"],
            "recommended_action": "emergency_response",
            "rationale": "Massive PE with haemodynamic compromise: hypotension and tachycardia despite normal SpO2.",
        }
        score, breakdown = grade_conflicting_vitals(response, case)
        assert score >= 0.80, f"CV005 perfect response should score ≥0.80, got {score}"
        assert breakdown["critical_sign"] == round(1.0 * 0.25, 3)
        assert breakdown["misleading_signs"] == round(1.0 * 0.20, 3)

    def test_cv005_pe_trap_spo2(self):
        """CV005 — Agent falls for SpO2 trap (says spo2 is critical) → zero on critical_sign."""
        case = next(c for c in CONFLICTING_VITALS_CASES if c["case_id"] == "CV005")
        response = {
            "priority": "critical",
            "critical_sign": "spo2",          # wrong — SpO2=96% is the misleading sign
            "misleading_signs": [],
            "recommended_action": "emergency_response",
            "rationale": "SpO2 seems slightly low.",
        }
        _, breakdown = grade_conflicting_vitals(response, case)
        assert breakdown["critical_sign"] == 0.0, \
            f"Falling for spo2 trap should score 0, got {breakdown['critical_sign']}"

    def test_new_cv_cases_news2_match_computed(self):
        """CV004 and CV005: stored news2_score must match compute_news2()."""
        from server.graders import compute_news2
        new_cases = [c for c in CONFLICTING_VITALS_CASES if c["case_id"] in ("CV004", "CV005")]
        for case in new_cases:
            computed, _ = compute_news2(case["vitals"])
            assert computed == case["news2_score"], \
                f"{case['case_id']}: stored={case['news2_score']} computed={computed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSepsisBundleGrader:
    """Tests for grade_sepsis_bundle — Hour-1 Sepsis Bundle compliance."""

    def test_sb001_perfect_response(self):
        """SB001 — Septic shock: perfect bundle including vasopressors and 2000ml."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB001")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement", "vasopressors"],
            "antibiotic_choice": "piperacillin_tazobactam",
            "fluid_volume_ml": 2000,
            "vasopressor_indicated": True,
            "rationale": "Septic shock: MAP=60 <65, lactate=4.8 >4. Full bundle + vasopressors. 30ml/kg=~2000ml. Pip-taz for CAP.",
        }
        score, breakdown = grade_sepsis_bundle(response, case)
        assert score >= 0.85, f"SB001 perfect response should score ≥0.85, got {score}"
        assert breakdown["_ab_verdict"] == "correct"
        assert breakdown["_vasopr_verdict"] == "correct"
        assert breakdown["_fluid_verdict"] == "correct"

    def test_sb001_missing_vasopressors(self):
        """SB001 — Agent misses vasopressors in shock → bundle completeness penalty + vasopressor penalty."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB001")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"],
            "antibiotic_choice": "piperacillin_tazobactam",
            "fluid_volume_ml": 2000,
            "vasopressor_indicated": False,  # wrong — shock is present
            "rationale": "Sepsis with standard bundle.",
        }
        score, breakdown = grade_sepsis_bundle(response, case)
        # Missing vasopressors in bundle + wrong vasopressor_indicated = significant penalty
        assert score <= 0.80, f"Missing vasopressors in shock should lower score, got {score}"
        assert breakdown["_vasopr_verdict"] == "wrong"

    def test_sb002_no_vasopressors_correct(self):
        """SB002 — Urosepsis without shock: vasopressors NOT required, correct bundle."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB002")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"],
            "antibiotic_choice": "ceftriaxone",
            "fluid_volume_ml": 500,
            "vasopressor_indicated": False,
            "rationale": "Sepsis no shock: MAP=82 >65. No vasopressors. Ceftriaxone for UTI source.",
        }
        score, breakdown = grade_sepsis_bundle(response, case)
        assert score >= 0.85, f"SB002 perfect no-vasopressor response should score ≥0.85, got {score}"
        assert breakdown["_vasopr_verdict"] == "correct"
        assert breakdown["_ab_verdict"] == "correct"

    def test_sb003_contraindicated_antibiotic(self):
        """SB003 — Penicillin allergy: pip-taz is contraindicated → zero antibiotic score."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB003")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"],
            "antibiotic_choice": "piperacillin_tazobactam",  # CONTRAINDICATED
            "fluid_volume_ml": 1500,
            "vasopressor_indicated": False,
            "rationale": "Standard sepsis bundle with pip-taz.",
        }
        score, breakdown = grade_sepsis_bundle(response, case)
        assert breakdown["_ab_verdict"] == "contraindicated", \
            f"Pip-taz in penicillin allergy should be contraindicated, got {breakdown['_ab_verdict']}"
        assert breakdown["antibiotic"] == 0.0, \
            f"Contraindicated antibiotic should score 0, got {breakdown['antibiotic']}"

    def test_sb003_correct_allergy_safe_antibiotic(self):
        """SB003 — Penicillin allergy: meropenem is correct alternative."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB003")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"],
            "antibiotic_choice": "meropenem",
            "fluid_volume_ml": 1500,
            "vasopressor_indicated": False,
            "rationale": "Penicillin allergy — using meropenem. MAP=72 >65 no vasopressors.",
        }
        score, breakdown = grade_sepsis_bundle(response, case)
        assert score >= 0.85, f"SB003 meropenem response should score ≥0.85, got {score}"
        assert breakdown["_ab_verdict"] == "correct"

    def test_sb004_aki_conservative_fluid(self):
        """SB004 — Sepsis + severe AKI: 500ml only (conservative). 2000ml should penalise."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB004")
        response_correct = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement", "vasopressors"],
            "antibiotic_choice": "piperacillin_tazobactam",
            "fluid_volume_ml": 500,
            "vasopressor_indicated": True,
            "rationale": "Severe AKI — conservative 500ml only. MAP <70 needs vasopressors.",
        }
        response_wrong_fluid = dict(response_correct, fluid_volume_ml=2000)
        score_correct, bd_correct = grade_sepsis_bundle(response_correct, case)
        score_wrong, bd_wrong = grade_sepsis_bundle(response_wrong_fluid, case)
        assert score_correct > score_wrong, \
            f"Conservative fluid (500ml) should score higher than standard (2000ml) in AKI: {score_correct} vs {score_wrong}"
        assert bd_correct["_fluid_verdict"] == "correct"
        assert bd_wrong["_fluid_verdict"] in ("off", "far_off")

    def test_sb001_alias_antibiotic(self):
        """SB001 — 'tazocin' is alias for piperacillin_tazobactam → accepted."""
        case = next(c for c in SEPSIS_BUNDLE_CASES if c["case_id"] == "SB001")
        response = {
            "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement", "vasopressors"],
            "antibiotic_choice": "tazocin",  # alias
            "fluid_volume_ml": 2000,
            "vasopressor_indicated": True,
        }
        _, breakdown = grade_sepsis_bundle(response, case)
        assert breakdown["_ab_verdict"] == "correct", \
            f"'tazocin' alias should be accepted, got {breakdown['_ab_verdict']}"

    def test_missing_all_fields(self):
        """Empty response should score near zero but not crash."""
        case = SEPSIS_BUNDLE_CASES[0]
        score, breakdown = grade_sepsis_bundle({}, case)
        assert score < 0.15, f"Empty response should score near 0, got {score}"
        assert "feedback" in breakdown

    def test_grade_response_raw_dispatch(self):
        """grade_response_raw dispatches to grade_sepsis_bundle for 'sepsis_bundle' task_id."""
        case = SEPSIS_BUNDLE_CASES[0]
        score, breakdown = grade_response_raw("sepsis_bundle", {}, case)
        assert isinstance(score, float)
        assert score < 0.15


# ─────────────────────────────────────────────────────────────
# PAEDIATRIC TRIAGE GRADER TESTS
# ─────────────────────────────────────────────────────────────

class TestPaediatricTriageGrader:
    """Tests for grade_paediatric_triage — PEWS-based paediatric scoring."""

    def test_pd001_perfect_response(self):
        """PD001 — Infant bronchiolitis: perfect response scores ≥0.85."""
        case = next(c for c in PAEDIATRIC_TRIAGE_CASES if c["case_id"] == "PD001")
        response = {
            "priority": "high",
            "age_group": "infant",
            "pews_score": 5,
            "critical_sign": "spo2",
            "recommended_action": "urgent_review",
            "rationale": "SpO2=87% in 4-month-old is critically low. PEWS=5 → urgent review.",
        }
        score, breakdown = grade_paediatric_triage(response, case)
        assert score >= 0.85, f"PD001 perfect response should score ≥0.85, got {score}"

    def test_pd003_critical_dka_perfect(self):
        """PD003 — School-age DKA: critical priority + emergency_response scores ≥0.85."""
        case = next(c for c in PAEDIATRIC_TRIAGE_CASES if c["case_id"] == "PD003")
        response = {
            "priority": "critical",
            "age_group": "school_age",
            "pews_score": 7,
            "critical_sign": "respiratory_rate",
            "recommended_action": "emergency_response",
        }
        score, breakdown = grade_paediatric_triage(response, case)
        assert score >= 0.85, f"PD003 perfect response should score ≥0.85, got {score}"

    def test_priority_partial_credit(self):
        """Adjacent priority gets partial credit (e.g. medium instead of high)."""
        case = next(c for c in PAEDIATRIC_TRIAGE_CASES if c["case_id"] == "PD001")
        response = {"priority": "medium", "age_group": "infant",
                    "critical_sign": "spo2", "recommended_action": "urgent_review"}
        score, breakdown = grade_paediatric_triage(response, case)
        # Should not be full score but not zero either
        assert 0.3 < score < 0.9, f"Adjacent priority should give partial credit, got {score}"

    def test_wrong_age_group_loses_points(self):
        """Wrong age group loses 0.25 weight."""
        case = next(c for c in PAEDIATRIC_TRIAGE_CASES if c["case_id"] == "PD001")
        response_correct = {"priority": "high", "age_group": "infant",
                            "critical_sign": "spo2", "recommended_action": "urgent_review"}
        response_wrong = {"priority": "high", "age_group": "school_age",
                          "critical_sign": "spo2", "recommended_action": "urgent_review"}
        score_correct, _ = grade_paediatric_triage(response_correct, case)
        score_wrong, _ = grade_paediatric_triage(response_wrong, case)
        assert score_correct > score_wrong + 0.2, "Wrong age group should cost ≥0.20 points"

    def test_synonym_age_group_toddler(self):
        """'toddler' synonym resolves correctly."""
        case = next(c for c in PAEDIATRIC_TRIAGE_CASES if c["case_id"] == "PD002")
        response = {"priority": "high", "age_group": "toddler",
                    "critical_sign": "temperature", "recommended_action": "urgent_review"}
        score, _ = grade_paediatric_triage(response, case)
        assert score >= 0.85, f"PD002 toddler perfect response should score ≥0.85, got {score}"

    def test_empty_response_scores_near_zero(self):
        """Empty response scores near zero."""
        case = PAEDIATRIC_TRIAGE_CASES[0]
        score, breakdown = grade_paediatric_triage({}, case)
        assert score < 0.15, f"Empty response should score near zero, got {score}"
        assert "feedback" in breakdown

    def test_dispatch_paediatric_triage(self):
        """grade_response_raw dispatches to grade_paediatric_triage."""
        case = PAEDIATRIC_TRIAGE_CASES[0]
        score, breakdown = grade_response_raw("paediatric_triage", {}, case)
        assert isinstance(score, float)
        assert score < 0.15


# ─────────────────────────────────────────────────────────────
# MEDICATION RECONCILIATION GRADER TESTS
# ─────────────────────────────────────────────────────────────

class TestMedicationReconciliationGrader:
    """Tests for grade_medication_reconciliation — drug safety scoring."""

    def test_mr001_perfect_response(self):
        """MR001 — Warfarin+NSAID: perfect response scores ≥0.85."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR001")
        response = {
            "issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr"],
            "severity": "critical",
            "requires_pharmacist": True,
            "recommended_action": "withhold_drug",
            "drug_to_withhold": "ibuprofen",
            "rationale": "Ibuprofen+Warfarin: major GI bleed risk. INR=4.8 elevated. Withhold NSAID.",
        }
        score, breakdown = grade_medication_reconciliation(response, case)
        assert score >= 0.85, f"MR001 perfect response should score ≥0.85, got {score}"

    def test_mr002_aki_perfect(self):
        """MR002 — AKI+NSAIDs: identifies all critical issues."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR002")
        response = {
            "issues_found": ["nsaid_contraindicated_in_aki", "ace_inhibitor_caution_in_aki", "methotrexate_renally_cleared_toxicity_risk"],
            "severity": "critical",
            "requires_pharmacist": True,
            "recommended_action": "withhold_drug",
        }
        score, breakdown = grade_medication_reconciliation(response, case)
        assert score >= 0.85, f"MR002 perfect response should score ≥0.85, got {score}"

    def test_partial_issues_gives_partial_credit(self):
        """Identifying some but not all issues gives partial credit on issues dimension."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR001")
        response = {
            "issues_found": ["warfarin_nsaid_interaction"],  # only 1 of 3
            "severity": "critical",
            "requires_pharmacist": True,
            "recommended_action": "withhold_drug",
        }
        score, breakdown = grade_medication_reconciliation(response, case)
        # Issues score = 1/3 * 0.40 = 0.133; rest correct = 0.60; total ~0.73
        assert 0.3 < score < 0.85, f"Partial issues should give partial credit, got {score}"

    def test_wrong_severity_loses_points(self):
        """Wrong severity costs 0.30 weight."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR001")
        response_correct = {"issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr"],
                            "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"}
        response_wrong   = {"issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr"],
                            "severity": "low", "requires_pharmacist": True, "recommended_action": "withhold_drug"}
        score_correct, _ = grade_medication_reconciliation(response_correct, case)
        score_wrong, _   = grade_medication_reconciliation(response_wrong, case)
        assert score_correct > score_wrong + 0.25, "Wrong severity should cost ≥0.25 points"

    def test_spurious_issues_penalised(self):
        """Hallucinated issues get a small penalty."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR001")
        response_clean = {"issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr"],
                          "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"}
        response_spurious = {"issues_found": ["warfarin_nsaid_interaction", "nsaid_potentiates_anticoagulation", "supratherapeutic_inr",
                                              "hallucinated_issue_a", "hallucinated_issue_b", "hallucinated_issue_c"],
                             "severity": "critical", "requires_pharmacist": True, "recommended_action": "withhold_drug"}
        score_clean, _    = grade_medication_reconciliation(response_clean, case)
        score_spurious, _ = grade_medication_reconciliation(response_spurious, case)
        assert score_clean >= score_spurious, "Spurious issues should not improve score"

    def test_empty_response_near_zero(self):
        """Empty response scores near zero."""
        case = MEDICATION_RECONCILIATION_CASES[0]
        score, breakdown = grade_medication_reconciliation({}, case)
        assert score < 0.15, f"Empty response should score near zero, got {score}"
        assert "feedback" in breakdown

    def test_mr003_modify_dose_action(self):
        """MR003 — ACE+K+-sparing diuretic: recommended_action = modify_dose."""
        case = next(c for c in MEDICATION_RECONCILIATION_CASES if c["case_id"] == "MR003")
        response = {
            "issues_found": ["ace_inhibitor_plus_potassium_sparing_diuretic_hyperkalaemia", "baseline_hyperkalaemia_risk", "ckd_reduces_potassium_excretion"],
            "severity": "high",
            "requires_pharmacist": True,
            "recommended_action": "modify_dose",
        }
        score, breakdown = grade_medication_reconciliation(response, case)
        assert score >= 0.85, f"MR003 perfect response should score ≥0.85, got {score}"

    def test_dispatch_medication_reconciliation(self):
        """grade_response_raw dispatches to grade_medication_reconciliation."""
        case = MEDICATION_RECONCILIATION_CASES[0]
        score, breakdown = grade_response_raw("medication_reconciliation", {}, case)
        assert isinstance(score, float)
        assert score < 0.15
