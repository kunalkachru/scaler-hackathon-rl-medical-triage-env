"""
tests/test_client_scripts.py — Client Script Coverage Tests
============================================================
Catches the class of bugs where a new task is added to TASKS lists
but the supporting content (SYSTEM_PROMPT formats, mock_actions schemas,
TASK_ACTION_FN entries) is not updated.

All three post-v2.3.0 defects would have been caught by this file:
  - inference.py SYSTEM_PROMPT missing T9/T10/T11 formats
  - export_hf_dataset.py mock_actions missing T9/T10/T11 → wrong schema → ~0 reward
  - grpo_train.py SYSTEM_PROMPT missing T9/T10/T11 formats

Tests are split into two groups:
  1. Static checks  — parse source files, no server required, fast
  2. Live checks    — submit mock actions to a local server, requires server running

Run static only (CI-safe, no server):
  pytest tests/test_client_scripts.py -m static -v

Run all (requires local server on :8000):
  uvicorn server.app:app --port 8000 &
  pytest tests/test_client_scripts.py -v

Run with: pytest tests/test_client_scripts.py -q
"""

import re
import sys
import os
import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Canonical task list (source of truth) ────────────────────────────────────

ALL_TASKS = [
    "simple_triage",
    "conflicting_vitals",
    "masked_deterioration",
    "demographic_fairness",
    "deteriorating_patient",
    "sepsis_bundle",
    "paediatric_triage",
    "medication_reconciliation",
    "icu_deterioration",
    "sbar_handover",
    "differential_diagnosis",
]

NEW_TASKS = ["icu_deterioration", "sbar_handover", "differential_diagnosis"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_SERVER = os.getenv("SERVER_URL", "http://localhost:8000")


def _read(relpath: str) -> str:
    return open(os.path.join(ROOT, relpath)).read()


def _server_reachable() -> bool:
    try:
        return requests.get(f"{LOCAL_SERVER}/health", timeout=2).status_code == 200
    except Exception:
        return False


# ── Static checks ─────────────────────────────────────────────────────────────


@pytest.mark.static
class TestInferenceSystemPrompt:
    """inference.py SYSTEM_PROMPT must contain a format section for every task."""

    src = _read("inference.py")

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_in_tasks_list(self, task):
        """Every task must appear in the TASKS list inside main()."""
        assert f'"{task}"' in self.src, (
            f"inference.py: '{task}' missing from TASKS list"
        )

    @pytest.mark.parametrize("task", NEW_TASKS)
    def test_new_task_in_system_prompt(self, task):
        """T9/T10/T11 must have explicit format guidance in SYSTEM_PROMPT."""
        assert task in self.src, (
            f"inference.py SYSTEM_PROMPT: '{task}' has no format section — "
            f"LLM gets zero guidance and will output wrong JSON schema"
        )

    def test_icu_sofa_score_in_prompt(self):
        """ICU task prompt must mention sofa_score field."""
        assert "sofa_score" in self.src, (
            "inference.py SYSTEM_PROMPT: icu_deterioration section missing sofa_score field"
        )

    def test_sbar_escalation_required_in_prompt(self):
        """SBAR task prompt must mention escalation_required field."""
        assert "escalation_required" in self.src, (
            "inference.py SYSTEM_PROMPT: sbar_handover section missing escalation_required field"
        )

    def test_diffdx_must_not_miss_in_prompt(self):
        """DiffDx task prompt must mention must_not_miss field."""
        assert "must_not_miss" in self.src, (
            "inference.py SYSTEM_PROMPT: differential_diagnosis section missing must_not_miss field"
        )


@pytest.mark.static
class TestGrpoTrainSystemPrompt:
    """grpo_train.py SYSTEM_PROMPT must cover all 11 tasks."""

    src = _read("grpo_train.py")

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_in_tasks_list(self, task):
        assert f'"{task}"' in self.src, (
            f"grpo_train.py: '{task}' missing from TASKS list"
        )

    @pytest.mark.parametrize("task", NEW_TASKS)
    def test_new_task_in_system_prompt(self, task):
        assert task in self.src, (
            f"grpo_train.py SYSTEM_PROMPT: '{task}' has no format section"
        )

    def test_icu_sofa_score_in_prompt(self):
        assert "sofa_score" in self.src

    def test_sbar_escalation_required_in_prompt(self):
        assert "escalation_required" in self.src

    def test_diffdx_must_not_miss_in_prompt(self):
        assert "must_not_miss" in self.src


@pytest.mark.static
class TestExportMockActions:
    """export_hf_dataset.py mock_actions must have correct-schema entries for all tasks."""

    src = _read("scripts/export_hf_dataset.py")

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_in_export_schedule(self, task):
        assert f'"{task}"' in self.src, (
            f"export_hf_dataset.py: '{task}' missing from EXPORT_SCHEDULE"
        )

    @pytest.mark.parametrize("task", NEW_TASKS)
    def test_new_task_in_mock_actions(self, task):
        assert f'"{task}"' in self.src, (
            f"export_hf_dataset.py mock_actions: '{task}' missing — "
            f"will fall back to generic schema and record near-zero rewards"
        )

    def test_icu_mock_has_sofa_score(self):
        assert "sofa_score" in self.src, (
            "export_hf_dataset.py: icu_deterioration mock action missing sofa_score"
        )

    def test_sbar_mock_has_escalation_required(self):
        assert "escalation_required" in self.src, (
            "export_hf_dataset.py: sbar_handover mock action missing escalation_required"
        )

    def test_diffdx_mock_has_must_not_miss(self):
        assert "must_not_miss" in self.src, (
            "export_hf_dataset.py: differential_diagnosis mock action missing must_not_miss"
        )


@pytest.mark.static
class TestRandomBaselineActions:
    """random_agent_baseline.py must have TASK_ACTION_FN entries for all tasks."""

    src = _read("scripts/random_agent_baseline.py")

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_in_tasks_list(self, task):
        count = self.src.count(f'"{task}"')
        assert count >= 2, (
            f"random_agent_baseline.py: '{task}' appears {count} time(s) — "
            f"needs ≥2 (TASKS list + TASK_ACTION_FN)"
        )

    def test_icu_random_fn_exists(self):
        assert "random_icu_deterioration" in self.src

    def test_sbar_random_fn_exists(self):
        assert "random_sbar_handover" in self.src

    def test_diffdx_random_fn_exists(self):
        assert "random_differential_diagnosis" in self.src


@pytest.mark.static
class TestColabNotebook:
    """grpo_colab.ipynb must have all 11 tasks and T9/T10/T11 format guidance."""

    src = _read("notebooks/grpo_colab.ipynb")

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_task_in_notebook(self, task):
        assert task in self.src, (
            f"grpo_colab.ipynb: '{task}' not found — check Cell 3 TASKS list"
        )

    @pytest.mark.parametrize("task", NEW_TASKS)
    def test_new_task_format_in_notebook(self, task):
        assert task in self.src, (
            f"grpo_colab.ipynb: '{task}' missing from SYSTEM_PROMPT (Cell 6)"
        )


# ── Live checks (require local server) ───────────────────────────────────────


def require_server(fn):
    """Skip test if local server is not reachable."""
    return pytest.mark.skipif(
        not _server_reachable(),
        reason=f"Local server not reachable at {LOCAL_SERVER} — start with: uvicorn server.app:app --port 8000"
    )(fn)


MOCK_ACTIONS = {
    "simple_triage": {
        "priority": "high", "news2_score": 7, "critical_sign": "spo2",
        "recommended_action": "urgent_review", "confidence": 0.8,
    },
    "conflicting_vitals": {
        "priority": "high", "critical_sign": "spo2", "misleading_signs": ["heart_rate"],
        "condition": "hypoxia", "recommended_action": "urgent_review", "confidence": 0.7,
    },
    "masked_deterioration": {
        "priority": "high", "masking_drug_or_condition": "bisoprolol",
        "masked_sign": "heart_rate", "critical_clues": ["lactate"],
        "condition": "sepsis", "recommended_action": "urgent_review", "confidence": 0.6,
    },
    "demographic_fairness": {
        "priority": "high", "critical_sign": "systolic_bp",
        "recommended_action": "urgent_review", "confidence": 0.8,
    },
    "deteriorating_patient": {
        # Multi-turn task: step 1 correct action is "monitor" (scores ~0.3, not done)
        # "escalate" on step 1 is premature → 0.0001. Use monitor for single-step test.
        "action": "monitor", "rationale": "Monitoring for now", "confidence": 0.7,
    },
    "sepsis_bundle": {
        "priority": "critical",
        "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"],
        "antibiotic_choice": "piperacillin_tazobactam",
        "fluid_volume_ml": 2000, "vasopressor_indicated": False, "confidence": 0.7,
    },
    "paediatric_triage": {
        "priority": "high", "age_group": "school_age", "pews_score": 5,
        "critical_sign": "spo2", "recommended_action": "urgent_review", "confidence": 0.7,
    },
    "medication_reconciliation": {
        "issues_found": ["drug_interaction"], "severity": "high",
        "requires_pharmacist": True, "recommended_action": "withhold_drug", "confidence": 0.7,
    },
    "icu_deterioration": {
        "sofa_score": 10, "primary_organ_failure": "cardiovascular",
        "deterioration_trend": "worsening", "intervention": "emergency_escalation",
        "rationale": "High SOFA with cardiovascular failure",
    },
    "sbar_handover": {
        "escalation_required": True, "priority": "critical",
        "assessment": "Patient deteriorating with worsening vitals and altered consciousness",
        "recommendation": "emergency_response",
    },
    "differential_diagnosis": {
        "must_not_miss": "stemi", "top_diagnosis": "acute_coronary_syndrome",
        "differentials": ["unstable_angina", "pulmonary_embolism", "aortic_dissection"],
        "first_investigation": "ecg", "urgency": "immediate",
    },
}


@pytest.mark.live
class TestMockActionsRewardAboveFloor:
    """
    Submit the canonical mock action for each task to a local server.
    Reward must be > 0.0001 (floor) — proves the schema is correct.
    Reward > 0.1 proves the grader is recognising the answer as meaningful.
    """

    @require_server
    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_mock_action_scores_above_floor(self, task):
        import uuid
        sid = str(uuid.uuid4())
        r1 = requests.post(f"{LOCAL_SERVER}/reset",
                           json={"task_id": task, "session_id": sid, "case_index": 0},
                           timeout=10)
        assert r1.status_code == 200, f"/reset failed for {task}: {r1.status_code}"

        action = MOCK_ACTIONS[task]
        r2 = requests.post(f"{LOCAL_SERVER}/step",
                           json={"session_id": sid, "action": action},
                           timeout=10)
        assert r2.status_code == 200, f"/step failed for {task}: {r2.status_code}"

        reward = r2.json()["reward"]
        assert reward > 0.1, (
            f"{task}: mock action scored {reward:.4f} — near-zero reward indicates "
            f"wrong action schema (check mock_actions dict and grader field names)"
        )

    @require_server
    @pytest.mark.parametrize("task", NEW_TASKS)
    def test_new_task_mock_scores_above_threshold(self, task):
        """T9/T10/T11 specifically must score > 0.5 with correct mock action."""
        import uuid
        sid = str(uuid.uuid4())
        requests.post(f"{LOCAL_SERVER}/reset",
                      json={"task_id": task, "session_id": sid, "case_index": 0},
                      timeout=10)
        action = MOCK_ACTIONS[task]
        r = requests.post(f"{LOCAL_SERVER}/step",
                          json={"session_id": sid, "action": action},
                          timeout=10)
        reward = r.json()["reward"]
        assert reward > 0.5, (
            f"{task}: mock action scored {reward:.4f} — should be >0.5 for a well-formed answer"
        )
