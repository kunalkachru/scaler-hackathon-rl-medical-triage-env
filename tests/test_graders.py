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
from server.graders import (
    compute_news2,
    news2_to_priority,
    priority_distance,
    grade_simple_triage,
    grade_conflicting_vitals,
    grade_masked_deterioration,
    grade_response,
)
from server.cases import SIMPLE_TRIAGE_CASES, CONFLICTING_VITALS_CASES, MASKED_DETERIORATION_CASES


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
        """Unknown task_id should return 0.0 with error info."""
        score, breakdown = grade_response("unknown_task", {}, SIMPLE_TRIAGE_CASES[0])
        assert score == 0.0
        assert "error" in breakdown

    def test_score_always_in_range(self):
        """All graders must return scores in [0.0, 1.0]."""
        for task_id, cases in [
            ("simple_triage", SIMPLE_TRIAGE_CASES),
            ("conflicting_vitals", CONFLICTING_VITALS_CASES),
            ("masked_deterioration", MASKED_DETERIORATION_CASES),
        ]:
            for case in cases:
                for response in [{}, {"priority": "low"}, {"priority": "critical", "news2_score": 15}]:
                    score, _ = grade_response(task_id, response, case)
                    assert 0.0 <= score <= 1.0, \
                        f"{task_id}/{case['case_id']}: score {score} out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
