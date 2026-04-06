"""API contract tests for request parsing and typed models."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_step_preserves_deterioration_action_field():
    reset_resp = client.post(
        "/reset",
        json={"task_id": "deteriorating_patient", "case_index": 0, "seed": 42},
    )
    assert reset_resp.status_code == 200

    step_resp = client.post(
        "/step",
        json={
            "action": {
                "action": "monitor",
                "rationale": "initially stable but continue close monitoring",
                "confidence": 0.8,
            }
        },
    )
    assert step_resp.status_code == 200
    payload = step_resp.json()
    assert payload["info"]["agent_action"] == "monitor"


def test_step_preserves_confidence_field_for_bonus():
    reset_resp = client.post(
        "/reset",
        json={"task_id": "simple_triage", "case_index": 1, "seed": 42},
    )
    assert reset_resp.status_code == 200

    step_resp = client.post(
        "/step",
        json={
            "action": {
                "priority": "low",
                "news2_score": 0,
                "critical_sign": "none",
                "recommended_action": "routine_monitoring",
                "confidence": 0.92,
            }
        },
    )
    assert step_resp.status_code == 200
    payload = step_resp.json()
    breakdown = payload["observation"].get("score_breakdown") or {}
    assert payload["reward"] >= 0.9
    assert "confidence_bonus" in breakdown or payload["reward"] == 1.0


def test_grade_fairness_consistent_responses():
    """Identical responses across all demographic variants should score 1.0."""
    consistent_responses = {
        "FP001_white_male":      {"priority": "high", "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
        "FP001_black_male":      {"priority": "high", "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
        "FP001_hispanic_female": {"priority": "high", "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
        "FP001_unhoused_female": {"priority": "high", "news2_score": 3, "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
    }
    resp = client.post("/grade-fairness", json={"group_id": "FP001", "responses": consistent_responses})
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] == 1.0
    assert "breakdown" in data


def test_grade_fairness_inconsistent_responses():
    """Different priorities across demographic variants should score below 1.0."""
    inconsistent_responses = {
        "FP001_white_male":      {"priority": "high",   "recommended_action": "urgent_review"},
        "FP001_black_male":      {"priority": "medium", "recommended_action": "routine_monitoring"},
        "FP001_hispanic_female": {"priority": "high",   "recommended_action": "urgent_review"},
        "FP001_unhoused_female": {"priority": "low",    "recommended_action": "routine_monitoring"},
    }
    resp = client.post("/grade-fairness", json={"group_id": "FP001", "responses": inconsistent_responses})
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] < 1.0


def test_grade_fairness_unknown_group_returns_404():
    resp = client.post("/grade-fairness", json={"group_id": "UNKNOWN_GROUP_XYZ", "responses": {}})
    assert resp.status_code == 404


def test_session_isolation_concurrent_episodes():
    """Two sessions running different tasks must not corrupt each other."""
    # Session A: simple_triage case 0 — explicit session_id to avoid _default collision
    resp_a = client.post("/reset", json={"task_id": "simple_triage", "case_index": 0,
                                         "seed": 1, "session_id": "test-session-A"})
    assert resp_a.status_code == 200
    sid_a = resp_a.json()["info"]["session_id"]

    # Session B: conflicting_vitals case 0 — separate explicit session
    resp_b = client.post("/reset", json={"task_id": "conflicting_vitals", "case_index": 0,
                                          "seed": 2, "session_id": "test-session-B"})
    assert resp_b.status_code == 200
    sid_b = resp_b.json()["info"]["session_id"]

    # Step session A — correct answer for ST001
    step_a = client.post("/step", json={
        "session_id": sid_a,
        "action": {"priority": "high", "news2_score": 8, "critical_sign": "respiratory_rate",
                   "recommended_action": "urgent_review"}
    })
    assert step_a.status_code == 200
    assert step_a.json()["reward"] >= 0.8   # good score for ST001

    # Step session B — correct answer for CV001
    step_b = client.post("/step", json={
        "session_id": sid_b,
        "action": {"priority": "critical", "critical_sign": "spo2", "recommended_action": "emergency_response",
                   "misleading_signs": ["heart_rate", "systolic_bp"],
                   "rationale": "silent hypoxia masked by normal HR and BP"}
    })
    assert step_b.status_code == 200
    # Sessions are isolated — B's reward must be based on CV001, not ST001
    assert step_b.json()["observation"]["task_id"] == "conflicting_vitals"


def test_metrics_endpoint_structure():
    """/metrics returns expected top-level keys and numeric values."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_episodes" in data
    assert "active_sessions" in data
    assert "by_task" in data
    assert "difficulty_gradient_verified" in data
    assert "cases_covered" in data
    assert isinstance(data["active_sessions"], int)
    assert isinstance(data["total_episodes"], int)


def test_metrics_by_task_has_distribution_fields():
    """/metrics by_task entries have distribution stats when episodes exist."""
    # Run one episode to populate history
    client.post("/reset", json={"task_id": "simple_triage", "case_index": 0})
    client.post("/step", json={"action": {"priority": "high", "news2_score": 8,
                                           "critical_sign": "respiratory_rate",
                                           "recommended_action": "urgent_review"}})

    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    if data["total_episodes"] > 0:
        for task_data in data["by_task"].values():
            for key in ("count", "avg", "min", "max", "p25", "p75"):
                assert key in task_data, f"Missing key '{key}' in by_task entry"


def test_step_info_is_typed():
    """StepResult.info contains session_id after reset."""
    resp = client.post("/reset", json={"task_id": "simple_triage", "case_index": 0})
    assert resp.status_code == 200
    info = resp.json()["info"]
    assert "session_id" in info
    assert isinstance(info["session_id"], str)
    assert "episode_id" in info
    assert "task_id" in info
    assert info["task_id"] == "simple_triage"
