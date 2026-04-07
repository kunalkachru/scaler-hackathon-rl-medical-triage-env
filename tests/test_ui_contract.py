"""UI contract tests for demo page structure and hooks."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    # Fallback for environments where FastAPI's re-export is not discoverable.
    from starlette.testclient import TestClient  # pyright: ignore[reportMissingImports]

from server.app import app


client = TestClient(app)


def test_web_ui_contains_ai_button_and_status_ids():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    # New wired IDs
    assert 'id="ai-btn"' in html
    assert 'id="ai-status"' in html
    assert 'id="submit-btn"' in html
    assert 'id="reset-btn"' in html

    # Legacy dead IDs should not exist
    assert 'id="agent-btn"' not in html
    assert 'id="agent-status"' not in html


def test_web_ui_contains_case_select_population_hooks():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'id="task-select"' in html
    assert 'id="case-select"' in html
    assert 'id="case-select-hint"' in html
    assert 'async function populateCaseSelect()' in html
    assert 'onTaskSelectionChange' in html
    assert "fetch('/tasks')" in html


def test_web_ui_has_episode_history_demarcation_hook():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'function logDivider(label)' in html
    assert 'Episode complete · final reward' in html


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
    assert '>116<' in html


def test_web_ui_task_switch_clears_stale_form_state():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'async function onTaskSelectionChange()' in html
    assert 'document.getElementById("response-form").innerHTML = "";' in html
    assert 'document.getElementById("result-section").innerHTML = "";' in html


def test_web_ui_training_empty_state_hint_present():
    resp = client.get('/web')
    assert resp.status_code == 200
    html = resp.text

    assert 'id="training-empty-hint"' in html
    assert "No completed episodes yet." in html


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
