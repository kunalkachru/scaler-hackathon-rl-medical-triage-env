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
