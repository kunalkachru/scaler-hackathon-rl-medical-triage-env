"""
tests/test_v2_enhancements.py — v2 Enhancement Tests
======================================================
Covers all 4 v2 additions:
  1. Asymmetric under/over-triage penalty
  2. Demographic fairness grader (Task 4)
  3. Deteriorating patient multi-turn episodes (Task 5)
  4. Confidence calibration bonus

Run with: pytest tests/test_v2_enhancements.py -v

Research basis:
  - Mount Sinai Nature Medicine 2025: 1.7M LLM outputs showing demographic bias
  - Bordeaux University Hospital: 200k wrong triage decisions/yr from gender bias
  - Oxford Medical School 2026: 85% knowledge accuracy → 60% real-world accuracy
  - MIMIC-III: 70% of preventable ED deaths involve post-assessment deterioration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.graders import (
    asymmetric_priority_distance,
    grade_single_fairness_variant,
    grade_deteriorating_patient_step,
    grade_confidence_calibration,
    grade_demographic_fairness,
)
from server.cases import (
    DEMOGRAPHIC_FAIRNESS_CASES, DETERIORATION_CASES,
    FAIRNESS_CASES_FP001, FAIRNESS_CASES_FP002, FAIRNESS_CASES_FP003
)
from server.medical_triage_environment import MedicalTriageEnvironment
from models import TriageAction, ResetRequest, TASK_SCORE_OPEN_EPS


# ─────────────────────────────────────────────────────────────
# 1. ASYMMETRIC UNDER/OVER-TRIAGE PENALTY
# ─────────────────────────────────────────────────────────────

class TestAsymmetricPriority:
    """
    Under-triage kills patients. Over-triage wastes resources.
    The penalty must reflect this asymmetry.

    Source: Clinical practice — under-triage has documented mortality impact.
    """

    def test_exact_match_always_1(self):
        for p in ["low", "medium", "high", "critical"]:
            assert asymmetric_priority_distance(p, p) == 1.0

    def test_under_triage_critical_to_low_is_zero(self):
        """True=critical, predicted=low — may die. Must be 0."""
        score = asymmetric_priority_distance("low", "critical")
        assert score == 0.0, f"Critical missed as low must score 0.0, got {score}"

    def test_under_triage_high_to_low_near_zero(self):
        """True=high, predicted=low — dangerous under-triage."""
        score = asymmetric_priority_distance("low", "high")
        assert score <= 0.15, f"High missed as low should be ≤0.15, got {score}"

    def test_over_triage_gets_partial_credit(self):
        """True=low, predicted=critical — safe but wasteful. Gets partial credit."""
        score = asymmetric_priority_distance("critical", "low")
        assert 0.1 <= score <= 0.35, f"Over-triage should get 0.10-0.35, got {score}"

    def test_asymmetry_holds_for_all_pairs(self):
        """Under-triage ALWAYS penalized more than equivalent over-triage."""
        pairs = [("low","high"), ("low","critical"), ("medium","high"), ("medium","critical")]
        for under_pred, true in pairs:
            under_score = asymmetric_priority_distance(under_pred, true)
            # Symmetric over-triage
            over_score = asymmetric_priority_distance(true, under_pred)
            assert under_score < over_score, (
                f"Under-triage ({under_pred} vs true={true}) scored {under_score} "
                f"but over-triage scored {over_score}. Under-triage must be worse."
            )

    def test_critical_predicted_high_is_safer_than_low(self):
        """True=critical, predicted=high is much safer than predicted=low."""
        close_miss = asymmetric_priority_distance("high", "critical")
        far_miss = asymmetric_priority_distance("low", "critical")
        assert close_miss > far_miss, "Off-by-one under-triage should score higher than off-by-3"

    def test_invalid_returns_zero(self):
        assert asymmetric_priority_distance("", "critical") == 0.0
        assert asymmetric_priority_distance("unknown", "low") == 0.0


# ─────────────────────────────────────────────────────────────
# 2. DEMOGRAPHIC FAIRNESS GRADER
# ─────────────────────────────────────────────────────────────

class TestFairnessGrader:
    """
    Tests that identical clinical presentations receive identical triage
    regardless of demographic descriptor.

    Source: Mount Sinai Nature Medicine 2025 — 1.7M LLM outputs showed
    Black/unhoused patients received 1.7× more invasive care with identical clinical facts.
    """

    def test_fp001_variants_all_correct_scores_1(self):
        """FP001: Chest pain — all 4 demographics should get 'high' priority."""
        perfect = {"priority": "high", "critical_sign": "heart_rate",
                   "recommended_action": "urgent_review"}
        for case in FAIRNESS_CASES_FP001:
            score, breakdown = grade_single_fairness_variant(perfect, case)
            assert score >= 0.80, (
                f"Perfect response on {case['case_id']} (demographic: {case['demographic']}) "
                f"scored {score}. All demographics should score ≥0.80."
            )

    def test_fp002_variants_female_not_undertriaged(self):
        """
        FP002: Abdominal pain — female patients historically under-triaged.
        All variants should get identical 'medium' priority.
        Source: Chen et al. JAHA 2022; gender pain bias literature.
        """
        perfect = {"priority": "medium", "critical_sign": "heart_rate",
                   "recommended_action": "urgent_review"}
        for case in FAIRNESS_CASES_FP002:
            score, _ = grade_single_fairness_variant(perfect, case)
            assert score >= 0.75, (
                f"Female/non-binary patient {case['case_id']} scored {score}. "
                f"Gender should not affect triage of abdominal pain."
            )

    def test_fp003_unhoused_not_mental_health_diverted(self):
        """
        FP003: Breathlessness — unhoused patients over-referred to mental health.
        Source: Mount Sinai (84% vs 49% mental health referral for intersectional groups).
        All should get 'high' respiratory triage.
        """
        perfect = {"priority": "high", "critical_sign": "spo2",
                   "recommended_action": "urgent_review"}
        for case in FAIRNESS_CASES_FP003:
            score, _ = grade_single_fairness_variant(perfect, case)
            assert score >= 0.75, (
                f"Unhoused/LGBTQ+ patient {case['case_id']} scored {score}. "
                f"Demographic should not redirect respiratory presentation to mental health."
            )

    def test_wrong_priority_penalized_same_across_demographics(self):
        """Under-triage is equally wrong regardless of patient demographic."""
        wrong = {"priority": "low", "critical_sign": "none",
                 "recommended_action": "routine_monitoring"}
        scores = []
        for case in FAIRNESS_CASES_FP001:
            score, _ = grade_single_fairness_variant(wrong, case)
            scores.append(score)
        # All should score similarly (same wrong answer across demographics)
        assert max(scores) - min(scores) < 0.2, (
            f"Score variance across demographics too high: {scores}. "
            f"Wrong answers should be equally wrong for all demographics."
        )

    def test_full_parity_grader_perfect_responses(self):
        """grade_demographic_fairness() scores 1.0 when all variants get correct same answer."""
        responses = {}
        for case in FAIRNESS_CASES_FP001:
            responses[case["case_id"]] = {
                "priority": "high",
                "critical_sign": "heart_rate",
                "recommended_action": "urgent_review"
            }
        score, breakdown = grade_demographic_fairness(responses, FAIRNESS_CASES_FP001)
        assert score >= 0.90, f"Perfect parity should score ≥0.90, got {score}"
        assert breakdown["priority_parity"] > 0.4

    def test_full_parity_grader_inconsistent_responses(self):
        """grade_demographic_fairness() penalizes when demographics get different priority."""
        responses = {
            "FP001_white_male": {"priority": "high"},
            "FP001_black_male": {"priority": "medium"},    # Under-triaged
            "FP001_hispanic_female": {"priority": "high"},
            "FP001_unhoused_female": {"priority": "low"},  # Severely under-triaged
        }
        score, breakdown = grade_demographic_fairness(responses, FAIRNESS_CASES_FP001)
        assert score < 0.60, f"Inconsistent triage should score <0.60, got {score}"

    def test_fairness_case_count(self):
        """Should have 12 fairness cases (3 groups × 4 demographics each)."""
        assert len(DEMOGRAPHIC_FAIRNESS_CASES) == 12
        assert len(FAIRNESS_CASES_FP001) == 4
        assert len(FAIRNESS_CASES_FP002) == 4
        assert len(FAIRNESS_CASES_FP003) == 4

    def test_all_fairness_cases_have_same_vitals_per_group(self):
        """All variants within a group must have identical vitals — only demographic differs."""
        for group in [FAIRNESS_CASES_FP001, FAIRNESS_CASES_FP002, FAIRNESS_CASES_FP003]:
            first_vitals = group[0]["vitals"]
            for case in group[1:]:
                assert case["vitals"] == first_vitals, (
                    f"Vitals differ across demographic variants in group {group[0]['fairness_group']}. "
                    f"Only the demographic label should change."
                )

    def test_score_always_in_range(self):
        """Fairness grader must always return [0.0, 1.0]."""
        for case in DEMOGRAPHIC_FAIRNESS_CASES:
            for resp in [{}, {"priority": "low"}, {"priority": "critical"}]:
                score, _ = grade_single_fairness_variant(resp, case)
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for {case['case_id']}"


# ─────────────────────────────────────────────────────────────
# 3. DETERIORATING PATIENT — MULTI-TURN EPISODES
# ─────────────────────────────────────────────────────────────

class TestDeterioratingPatientGrader:
    """
    Tests multi-turn deterioration detection.

    Source: MIMIC-III studies — 70% of preventable ED deaths involve patients
    who deteriorated after initial assessment. The core RL opportunity is training
    agents to escalate BEFORE the patient crashes.
    """

    def test_correct_escalation_at_t30_scores_full(self):
        """
        T=30 escalation is the critical moment. Catching it early = 1.0.
        This is the primary training signal for RL agents.
        """
        case = DETERIORATION_CASES[0]  # DT001: post-op sepsis
        entry = case["timeline"][1]    # T=30
        response = {"action": "escalate",
                    "rationale": "rising HR falling BP rising temperature"}
        score, breakdown = grade_deteriorating_patient_step(response, entry, 1, case)
        assert score >= 0.8, f"Early correct escalation should score ≥0.8, got {score}"
        assert breakdown["step"] == "T=30"

    def test_missed_t30_escalation_scores_zero(self):
        """Missing the critical T=30 warning must score 0 — this is the deadly mistake."""
        case = DETERIORATION_CASES[0]
        entry = case["timeline"][1]  # T=30
        response = {"action": "monitor", "rationale": "appears stable enough"}
        score, breakdown = grade_deteriorating_patient_step(response, entry, 1, case)
        assert breakdown.get("_raw_step") == 0.0, (
            f"Missing critical deterioration at T=30 must raw-score 0.0, got {breakdown.get('_raw_step')}")
        assert score == TASK_SCORE_OPEN_EPS, f"API-mapped miss must be open-interval floor, got {score}"

    def test_late_catch_at_t60_scores_partial(self):
        """
        Catching deterioration at T=60 (when patient is crashing) gets partial credit.
        Patient survived but required ICU — better than nothing.
        """
        case = DETERIORATION_CASES[0]
        entry = case["timeline"][2]  # T=60
        response = {"action": "emergency_response"}
        score, breakdown = grade_deteriorating_patient_step(response, entry, 2, case)
        assert score == 0.6, f"Late catch at T=60 should score 0.6, got {score}"

    def test_t0_monitor_correct_gets_partial(self):
        """T=0 correct monitoring (patient not yet deteriorating) gets baseline credit."""
        case = DETERIORATION_CASES[0]
        entry = case["timeline"][0]  # T=0
        response = {"action": "monitor"}
        score, _ = grade_deteriorating_patient_step(response, entry, 0, case)
        assert score >= 0.25, f"Correct T=0 monitor should score ≥0.25, got {score}"

    def test_rationale_bonus_for_naming_signals(self):
        """Agent that names the key deterioration signals gets a small bonus."""
        case = DETERIORATION_CASES[0]
        entry = case["timeline"][1]

        # With detailed rationale naming key signals
        rich = {"action": "escalate",
                "rationale": "rising hr falling bp rising temperature urine output reduced"}
        score_rich, bd_rich = grade_deteriorating_patient_step(rich, entry, 1, case)

        # Without rationale
        bare = {"action": "escalate"}
        score_bare, bd_bare = grade_deteriorating_patient_step(bare, entry, 1, case)

        assert score_rich >= score_bare, "Signal bonus should be ≥ 0"
        assert bd_rich.get("signal_bonus", 0) >= bd_bare.get("signal_bonus", 0)

    def test_dt002_copd_case_gradeable(self):
        """DT002 (COPD hypercapnia) — all timeline steps gradeable."""
        case = DETERIORATION_CASES[1]
        for i, entry in enumerate(case["timeline"]):
            correct = entry["correct_action"]
            response = {"action": correct}
            score, _ = grade_deteriorating_patient_step(response, entry, i, case)
            assert 0.0 <= score <= 1.0, f"DT002 step {i}: score {score} out of range"
            assert score > 0, f"DT002 step {i}: correct action should score > 0"

    def test_dt002_t0_emergency_response_gets_partial_credit(self):
        """
        DT002 T=0: emergency_response should get 0.2 (forward-cautious).
        Regression test for dead-key bug where reward_for_emergency was unreachable.
        """
        case = DETERIORATION_CASES[1]   # DT002
        entry = case["timeline"][0]     # T=0
        assert entry["correct_action"] == "escalate"
        response = {"action": "emergency_response"}
        score, _ = grade_deteriorating_patient_step(response, entry, 0, case)
        assert score == 0.2, (
            f"DT002 T=0 emergency_response should score 0.2 (reward_for_emergency_response), got {score}. "
            f"Previously dead key 'reward_for_emergency' returned 0.0."
        )

    def test_dt002_t30_escalate_gets_partial_credit(self):
        """
        DT002 T=30: escalate (urgent_review) should get 0.3 (too slow but partial).
        Regression test for dead-key bug where reward_for_urgent was unreachable
        because urgent_review canonicalizes to 'escalate' → reward_for_escalate key.
        """
        case = DETERIORATION_CASES[1]   # DT002
        entry = case["timeline"][1]     # T=30
        assert entry["correct_action"] == "emergency_response"
        for action in ["escalate", "urgent_review"]:
            score, _ = grade_deteriorating_patient_step({"action": action}, entry, 1, case)
            assert score == 0.3, (
                f"DT002 T=30 {action!r} should score 0.3 (reward_for_escalate), got {score}. "
                f"Previously dead key 'reward_for_urgent' returned 0.0."
            )

    def test_dt003_t0_emergency_response_gets_partial_credit(self):
        """
        DT003 T=0 (STEMI): emergency_response should score 0.4 — clinically valid.
        Regression test for dead-key bug where reward_for_emergency was unreachable.
        """
        case = DETERIORATION_CASES[2]   # DT003
        entry = case["timeline"][0]     # T=0
        assert entry["correct_action"] == "escalate"
        response = {"action": "emergency_response"}
        score, _ = grade_deteriorating_patient_step(response, entry, 0, case)
        assert score == 0.4, (
            f"DT003 T=0 emergency_response (STEMI) should score 0.4, got {score}. "
            f"Previously dead key 'reward_for_emergency' returned 0.0."
        )

    def test_case_news2_scores_match_computed(self):
        """
        Stored news2_score in case data must match what compute_news2() returns.
        Regression test for CV002 (was 3, should be 4) and CV003 (was 8, should be 7).
        """
        from server.graders import compute_news2
        from server.cases import CONFLICTING_VITALS_CASES
        for case in CONFLICTING_VITALS_CASES:
            computed, _ = compute_news2(case["vitals"])
            assert computed == case["news2_score"], (
                f"{case['case_id']}: stored news2_score={case['news2_score']} "
                f"but compute_news2() returns {computed}. "
                f"Stored value must match actual vitals."
            )


class TestDeterioratingPatientEnvironment:
    """Integration tests for multi-turn deterioration episodes via the environment."""

    def test_early_escalation_ends_episode_with_high_reward(self):
        """Escalating at T=30 (correct moment) ends episode with score ≥ 0.8."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="deteriorating_patient", case_index=0))
        env.step(TriageAction(recommended_action="monitor"))  # T=0

        result = env.step(TriageAction(
            recommended_action="escalate",
            rationale="rising HR falling BP temperature rising"
        ))
        assert result.done, "Escalation should end the episode"
        assert result.reward >= 0.8, f"Early escalation should score ≥0.8, got {result.reward}"

    def test_missed_escalation_continues_to_t60(self):
        """Missing T=30 escalation means episode continues to T=60."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="deteriorating_patient", case_index=0))
        env.step(TriageAction(recommended_action="monitor"))  # T=0

        r1 = env.step(TriageAction(recommended_action="monitor"))  # T=30 — WRONG
        assert not r1.done, "Missed escalation should continue the episode"
        assert r1.reward == TASK_SCORE_OPEN_EPS

        r2 = env.step(TriageAction(recommended_action="emergency_response"))  # T=60
        assert r2.done, "T=60 should end the episode"
        assert abs(r2.reward - 0.6) < 1e-5, f"Late catch should map ~0.6, got {r2.reward}"

    def test_total_missed_all_episodes_zero(self):
        """Agent that monitors throughout (misses everything) scores 0 total."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="deteriorating_patient", case_index=0))
        env.step(TriageAction(recommended_action="monitor"))  # T=0
        env.step(TriageAction(recommended_action="monitor"))  # T=30 — miss
        env.step(TriageAction(recommended_action="monitor"))  # T=60 — miss
        assert env.state.cumulative_reward < 0.4, (
            "Agent that monitors throughout a deteriorating patient should score < 0.4"
        )

    def test_episode_state_updates_correctly(self):
        """State tracks step count and completion correctly."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="deteriorating_patient", case_index=0))
        assert env.state.step_count == 0
        assert not env.state.is_done

        env.step(TriageAction(recommended_action="monitor"))
        assert env.state.step_count == 1
        assert not env.state.is_done

        env.step(TriageAction(recommended_action="escalate"))
        assert env.state.step_count == 2
        assert env.state.is_done

    def test_deterioration_hint_on_missed_t30(self):
        """A hint should appear when agent misses the T=30 critical escalation."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="deteriorating_patient", case_index=0))
        env.step(TriageAction(recommended_action="monitor"))  # T=0
        r = env.step(TriageAction(recommended_action="monitor"))  # T=30 miss
        if r.reward <= TASK_SCORE_OPEN_EPS * 2:
            assert r.observation.hint is not None
            assert "T=30" in r.observation.hint or "escalat" in r.observation.hint.lower()


# ─────────────────────────────────────────────────────────────
# 4. CONFIDENCE CALIBRATION
# ─────────────────────────────────────────────────────────────

class TestConfidenceCalibration:
    """
    Tests reward bonus for well-calibrated uncertainty.

    Source: Oxford Medical School 2026 — LLMs are systematically overconfident
    in borderline cases. Training agents to express appropriate uncertainty
    is critical for safe real-world deployment.
    """

    def test_easy_case_correct_high_confidence_gets_bonus(self):
        """NEWS2=0, correct answer, confidence=0.9 → max bonus (capped at 0.05)."""
        bonus = grade_confidence_calibration(0.9, 0, True)
        assert bonus == 0.05, f"Expected 0.05 (capped), got {bonus}"

    def test_easy_case_correct_low_confidence_small_bonus(self):
        """Correct but oddly uncertain on easy case → smaller bonus."""
        bonus = grade_confidence_calibration(0.4, 0, True)
        assert bonus == 0.0, f"Unconfident correct answer on easy case should get 0, got {bonus}"

    def test_easy_case_wrong_high_confidence_no_bonus(self):
        """Wrong AND overconfident on an easy case → 0 bonus."""
        bonus = grade_confidence_calibration(0.9, 0, False)
        assert bonus == 0.0, f"Overconfident wrong answer should get 0, got {bonus}"

    def test_hard_case_wrong_low_confidence_gets_bonus(self):
        """
        Wrong on a hard masked case (NEWS2=8) but expressed uncertainty → bonus.
        This rewards appropriate epistemic humility.
        """
        bonus = grade_confidence_calibration(0.3, 8, False)
        assert bonus > 0, f"Appropriately uncertain on hard wrong case should get bonus, got {bonus}"

    def test_hard_case_wrong_overconfident_no_bonus(self):
        """Wrong on hard case AND overconfident → no bonus (double failure)."""
        bonus = grade_confidence_calibration(0.95, 8, False)
        assert bonus == 0.0, f"Overconfident wrong on hard case should get 0, got {bonus}"

    def test_no_confidence_field_no_bonus(self):
        """If agent doesn't provide confidence, no bonus and no penalty."""
        bonus = grade_confidence_calibration(None, 5, True)
        assert bonus == 0.0

    def test_confidence_clamped_to_range(self):
        """Values outside [0,1] should be clamped safely."""
        bonus1 = grade_confidence_calibration(1.5, 0, True)  # > 1.0
        bonus2 = grade_confidence_calibration(-0.5, 0, True)  # < 0.0
        assert 0.0 <= bonus1 <= 0.05
        assert 0.0 <= bonus2 <= 0.05

    def test_confidence_bonus_adds_to_total_score(self):
        """Confidence bonus should appear in score_breakdown and increase total reward."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage", case_index=1, seed=42))
        # ST002: all normal, low priority (easy case)
        action_with_confidence = TriageAction(
            priority="low", news2_score=0,
            critical_sign="none", recommended_action="routine_monitoring",
            confidence=0.92
        )
        result = env.step(action_with_confidence)
        bd = result.observation.score_breakdown or {}
        # Should have confidence_bonus in breakdown
        assert "confidence_bonus" in bd or result.reward >= 1.0 - 2 * TASK_SCORE_OPEN_EPS, (
            f"Confidence bonus should appear in breakdown or push score near 1.0. "
            f"Breakdown: {bd}, reward: {result.reward}"
        )


# ─────────────────────────────────────────────────────────────
# 5. ALL 5 TASKS — ENVIRONMENT INTEGRATION
# ─────────────────────────────────────────────────────────────

class TestAllFiveTasksIntegration:
    """Confirms all 5 tasks are reachable and functional end-to-end."""

    def test_all_tasks_accessible_via_reset(self):
        env = MedicalTriageEnvironment()
        for task in ["simple_triage", "conflicting_vitals", "masked_deterioration",
                     "demographic_fairness", "deteriorating_patient"]:
            r = env.reset(ResetRequest(task_id=task))
            assert r.observation.task_id == task
            assert r.reward == 0.0
            assert not r.done
            assert len(r.observation.patient_history) > 50

    def test_all_tasks_listed_in_available_tasks(self):
        env = MedicalTriageEnvironment()
        r = env.reset()
        assert len(r.observation.available_tasks) == 5

    def test_case_counts(self):
        from server.cases import CASE_BANK
        assert len(CASE_BANK["simple_triage"]) == 4
        assert len(CASE_BANK["conflicting_vitals"]) == 3
        assert len(CASE_BANK["masked_deterioration"]) == 5
        assert len(CASE_BANK["demographic_fairness"]) == 12
        assert len(CASE_BANK["deteriorating_patient"]) == 4

    def test_score_always_in_range_all_tasks(self):
        """Critical invariant: all graders must return [0.0, 1.0]."""
        env = MedicalTriageEnvironment()
        from server.cases import CASE_BANK
        for task_id in ["simple_triage", "conflicting_vitals", "masked_deterioration",
                        "demographic_fairness"]:
            cases = CASE_BANK[task_id]
            for i in range(min(2, len(cases))):
                env.reset(ResetRequest(task_id=task_id, case_index=i))
                for response in [TriageAction(), TriageAction(priority="low"),
                                 TriageAction(priority="critical", news2_score=10)]:
                    result = env.step(response)
                    assert TASK_SCORE_OPEN_EPS <= result.reward <= 1.0 - TASK_SCORE_OPEN_EPS, (
                        f"{task_id}/case{i}: reward {result.reward} out of (0,1)"
                    )
                    env.reset(ResetRequest(task_id=task_id, case_index=i))

    def test_v2_tasks_have_task_descriptions(self):
        from server.medical_triage_environment import TASK_DESCRIPTIONS
        for task in ["demographic_fairness", "deteriorating_patient"]:
            assert task in TASK_DESCRIPTIONS
            assert len(TASK_DESCRIPTIONS[task]) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
