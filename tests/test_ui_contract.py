"""UI contract tests for demo page structure and hooks."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_web_ui_contains_ai_button_and_status_ids():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    # New wired IDs
    assert 'id="ai-btn"' in html
    assert 'id="ai-status"' in html

    # Legacy dead IDs should not exist
    assert 'id="agent-btn"' not in html
    assert 'id="agent-status"' not in html


def test_web_ui_contains_case_select_population_hooks():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'id="task-select"' in html
    assert 'id="case-select"' in html
    assert 'async function populateCaseSelect()' in html
    assert 'onTaskSelectionChange' in html
    assert "fetch('/tasks')" in html


def test_web_ui_uses_session_id_for_isolated_episodes():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'function createSessionId()' in html
    assert 'session_id: createSessionId()' in html
    assert 'if (state.session_id) payload.session_id = state.session_id;' in html


def test_web_ui_placeholder_metrics_match_current_counts():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert '>28<' in html
    assert '>106<' in html


def test_tasks_endpoint_has_case_ids_for_all_tasks():
    resp = client.get('/tasks')
    assert resp.status_code == 200
    payload = resp.json()

    expected_tasks = {
        'simple_triage',
        'conflicting_vitals',
        'masked_deterioration',
        'demographic_fairness',
        'deteriorating_patient',
    }
    assert expected_tasks.issubset(payload.keys())

    for task_id in expected_tasks:
        task_data = payload[task_id]
        assert isinstance(task_data.get('case_ids'), list)
        assert len(task_data['case_ids']) >= 1
