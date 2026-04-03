"""
tests/test_environment.py — Environment Integration Tests
===========================================================
Tests the full reset → step → state flow.
These test the environment logic directly (no HTTP server needed).

Run with: pytest tests/test_environment.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from server.medical_triage_environment import MedicalTriageEnvironment
from models import TriageAction, ResetRequest


class TestEnvironmentReset:

    def test_reset_returns_patient_history(self):
        """After reset(), observation must contain a non-empty patient history."""
        env = MedicalTriageEnvironment()
        result = env.reset()
        assert result.observation.patient_history, "Patient history must not be empty"
        assert len(result.observation.patient_history) > 50, "History should be detailed"

    def test_reset_returns_task_id(self):
        """reset() observation must include a valid task_id."""
        env = MedicalTriageEnvironment()
        result = env.reset()
        from server.cases import ALL_TASKS
        assert result.observation.task_id in ALL_TASKS, \
            f"task_id '{result.observation.task_id}' not in {ALL_TASKS}"

    def test_reset_with_specific_task(self):
        """Requesting a specific task should return that task's case."""
        env = MedicalTriageEnvironment()
        for task in ["simple_triage", "conflicting_vitals", "masked_deterioration"]:
            result = env.reset(ResetRequest(task_id=task))
            assert result.observation.task_id == task, \
                f"Requested {task}, got {result.observation.task_id}"

    def test_reset_with_seed_is_reproducible(self):
        """Same seed should produce same case."""
        env1 = MedicalTriageEnvironment()
        env2 = MedicalTriageEnvironment()
        r1 = env1.reset(ResetRequest(seed=42, task_id="simple_triage"))
        r2 = env2.reset(ResetRequest(seed=42, task_id="simple_triage"))
        assert r1.observation.case_id == r2.observation.case_id
        assert r1.observation.patient_history == r2.observation.patient_history

    def test_reset_initializes_reward_to_zero(self):
        """Reset should return reward=0.0."""
        env = MedicalTriageEnvironment()
        result = env.reset()
        assert result.reward == 0.0

    def test_reset_done_is_false(self):
        """After reset, done must be False."""
        env = MedicalTriageEnvironment()
        result = env.reset()
        assert result.done is False

    def test_reset_lists_available_tasks(self):
        """Observation should list all available tasks."""
        env = MedicalTriageEnvironment()
        result = env.reset()
        assert result.observation.available_tasks is not None
        assert len(result.observation.available_tasks) == 5


class TestEnvironmentStep:

    def test_step_without_reset_raises(self):
        """Calling step() before reset() should raise RuntimeError."""
        env = MedicalTriageEnvironment()
        action = TriageAction(priority="low")
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)

    def test_step_returns_reward_in_range(self):
        """step() reward must always be in [0.0, 1.0]."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage", seed=0))
        for priority in ["low", "medium", "high", "critical"]:
            env.reset(ResetRequest(task_id="simple_triage", seed=0))
            action = TriageAction(priority=priority, news2_score=5, critical_sign="spo2")
            result = env.step(action)
            assert 0.0 <= result.reward <= 1.0, \
                f"Reward {result.reward} out of range for priority={priority}"

    def test_step_returns_done_true(self):
        """After step(), done must be True (single-step episodes)."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage"))
        result = env.step(TriageAction(priority="low"))
        assert result.done is True

    def test_step_returns_score_breakdown(self):
        """score_breakdown must be present in observation."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage"))
        result = env.step(TriageAction(priority="high", news2_score=7))
        assert result.observation.score_breakdown is not None
        assert isinstance(result.observation.score_breakdown, dict)

    def test_step_reveals_ground_truth(self):
        """After step(), info must contain ground_truth."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage", case_index=0))
        result = env.step(TriageAction(priority="high"))
        assert "ground_truth" in result.info, "Ground truth should be revealed after step"

    def test_perfect_simple_triage_scores_high(self):
        """A clinically perfect triage of ST001 should score ≥ 0.85."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage", case_index=0, seed=42))
        # ST001 ground truth: priority=high, news2=8, critical_sign=respiratory_rate
        action = TriageAction(
            priority="high",
            news2_score=8,
            critical_sign="respiratory_rate",
            recommended_action="urgent_review",
            rationale="NEWS2=8, elevated RR and SpO2, tachycardia"
        )
        result = env.step(action)
        assert result.reward >= 0.85, \
            f"Perfect ST001 response should score ≥0.85, got {result.reward}"

    def test_empty_action_scores_zero(self):
        """Empty TriageAction should score 0."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage"))
        result = env.step(TriageAction())
        assert result.reward == 0.0

    def test_feedback_present_after_step(self):
        """Feedback string must be present in observation after step."""
        env = MedicalTriageEnvironment()
        env.reset()
        result = env.step(TriageAction(priority="high"))
        assert result.observation.feedback is not None
        assert len(result.observation.feedback) > 0

    def test_hint_on_low_score(self):
        """When score is low, a hint should be provided."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="masked_deterioration"))
        # Deliberately wrong response
        action = TriageAction(priority="low", recommended_action="routine_monitoring")
        result = env.step(action)
        if result.reward < 0.4:
            assert result.observation.hint is not None, \
                "Hint should be provided when score < 0.40"


class TestEnvironmentState:

    def test_state_reflects_episode_id(self):
        """State should have a unique episode_id after reset."""
        env = MedicalTriageEnvironment()
        env.reset()
        state = env.state
        assert state.episode_id is not None
        assert state.episode_id.startswith("ep-")

    def test_state_step_count_increments(self):
        """step_count in state should increment with each step."""
        env = MedicalTriageEnvironment()
        env.reset()
        assert env.state.step_count == 0
        env.step(TriageAction(priority="high"))
        assert env.state.step_count == 1

    def test_state_cumulative_reward_tracks(self):
        """Cumulative reward in state should accumulate across steps."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage", case_index=0))
        env.step(TriageAction(priority="high", news2_score=8))
        assert env.state.cumulative_reward > 0.0, "Cumulative reward should be > 0"

    def test_state_is_done_after_step(self):
        """is_done should be True after a step on single-step tasks."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage"))
        assert env.state.is_done is False
        env.step(TriageAction(priority="low"))
        assert env.state.is_done is True

    def test_tasks_completed_accumulates(self):
        """tasks_completed should grow across resets."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="simple_triage"))
        env.step(TriageAction(priority="high"))
        assert "simple_triage" in env.state.tasks_completed

        env.reset(ResetRequest(task_id="conflicting_vitals"))
        env.step(TriageAction(priority="critical"))
        assert "conflicting_vitals" in env.state.tasks_completed
        assert "simple_triage" in env.state.tasks_completed


class TestEpisodeFlows:

    def test_full_episode_simple_triage(self):
        """Complete episode flow: reset → step → state check."""
        env = MedicalTriageEnvironment()

        # reset
        r = env.reset(ResetRequest(task_id="simple_triage", seed=1))
        assert r.observation.task_id == "simple_triage"
        assert r.reward == 0.0
        assert not r.done

        # step
        action = TriageAction(
            priority="high",
            news2_score=7,
            critical_sign="spo2",
            recommended_action="urgent_review"
        )
        r2 = env.step(action)
        assert r2.done
        assert 0.0 <= r2.reward <= 1.0
        assert r2.observation.score is not None

        # state
        s = env.state
        assert s.step_count == 1
        assert s.is_done

    def test_full_episode_masked_deterioration(self):
        """Complete hard-task episode: agent must detect beta-blocker masking."""
        env = MedicalTriageEnvironment()
        env.reset(ResetRequest(task_id="masked_deterioration", case_index=0))

        action = TriageAction(
            priority="critical",
            masking_drug_or_condition="bisoprolol",
            masked_sign="heart_rate",
            critical_clues=["lactate", "urine_output_reduced", "hypotension"],
            condition="septic_shock",
            recommended_action="emergency_response",
            rationale="Beta-blocker masks tachycardia. Elevated lactate and hypotension indicate septic shock."
        )
        result = env.step(action)
        assert result.reward >= 0.75, \
            f"Expert masked deterioration response should score ≥0.75, got {result.reward}"

    def test_multiple_episodes_independent(self):
        """Multiple episodes should not share state between resets."""
        env = MedicalTriageEnvironment()

        env.reset(ResetRequest(task_id="simple_triage", case_index=0))
        env.step(TriageAction(priority="high"))
        ep1_id = env.state.episode_id

        env.reset(ResetRequest(task_id="simple_triage", case_index=1))
        assert env.state.episode_id != ep1_id, "Each episode should have unique ID"
        assert env.state.step_count == 0, "Step count should reset"
        assert env.state.is_done is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
