"""
client.py — MedicalTriageEnv Client
=====================================
WHY THIS FILE EXISTS:
  OpenEnv requires a client class that wraps the HTTP server
  so external code can use it as a simple Python API.
  This is what training loops (TRL, torchforge) import.

  Pattern follows openenv EnvClient interface:
  - reset() → StepResult
  - step(action) → StepResult
  - state() → TriageState
  - Supports both sync and async usage
"""

import requests
from models import TriageAction, StepResult, TriageState, ResetRequest


class MedicalTriageEnv:
    """
    HTTP client for the Medical Triage Environment.

    Usage:
        env = MedicalTriageEnv(base_url="http://localhost:8000")
        result = env.reset(task_id="simple_triage")
        print(result.observation.patient_history)

        action = TriageAction(
            priority="critical",
            news2_score=9,
            critical_sign="systolic_bp",
            recommended_action="emergency_response"
        )
        result = env.step(action)
        print(result.reward)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._session_id: str | None = None  # set after reset(), used by step()

    def reset(
        self,
        task_id: str | None = None,
        case_index: int | None = None,
        seed: int | None = None,
        session_id: str | None = None,
    ) -> StepResult:
        """Start a new episode. Returns first observation (patient case)."""
        payload: dict = {}
        if task_id:
            payload["task_id"] = task_id
        if case_index is not None:
            payload["case_index"] = case_index
        if seed is not None:
            payload["seed"] = seed
        if session_id is not None:
            payload["session_id"] = session_id

        resp = self.session.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        result = StepResult(**resp.json())
        # Cache session_id so step() targets the correct episode automatically
        self._session_id = result.info.get("session_id") if result.info else None
        return result

    def step(self, action: TriageAction, session_id: str | None = None) -> StepResult:
        """Submit a triage assessment. Returns scored observation."""
        payload: dict = {"action": action.model_dump(exclude_none=True)}
        sid = session_id or self._session_id
        if sid:
            payload["session_id"] = sid
        resp = self.session.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self, session_id: str | None = None) -> TriageState:
        """Get current episode state."""
        sid = session_id or self._session_id
        params = {"session_id": sid} if sid else {}
        resp = self.session.get(f"{self.base_url}/state", params=params)
        resp.raise_for_status()
        return TriageState(**resp.json())

    def health(self) -> dict:
        """Check if server is running."""
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> dict:
        """List all available tasks."""
        resp = self.session.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
