import importlib.util
import types
from pathlib import Path

_INFERENCE_PATH = Path(__file__).resolve().parents[1] / "inference.py"
_SPEC = importlib.util.spec_from_file_location("inference_module", _INFERENCE_PATH)
assert _SPEC and _SPEC.loader
inference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inference)


def test_validate_required_env_reports_missing_fields(monkeypatch):
    monkeypatch.setattr(inference, "API_KEY", "")

    missing = inference._validate_required_env()

    assert "HF_TOKEN (or OPENAI_API_KEY/API_KEY)" in missing


def test_main_uses_server_url_for_case_lookup(monkeypatch, capsys):
    monkeypatch.setattr(inference, "API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setattr(inference, "MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setattr(inference, "API_KEY", "dummy-token")
    monkeypatch.setattr(inference, "SERVER_URL", "https://triage-env.example.com")

    monkeypatch.setattr(
        inference,
        "OpenAI",
        lambda base_url, api_key: object(),
    )

    monkeypatch.setattr(
        inference,
        "run_episode",
        lambda client, task_id, case_index, server_url: (
            0.8,
            [0.8],
            {"priority": "high", "critical_sign": "heart_rate", "recommended_action": "urgent_review"},
        ),
    )

    captured_server_urls = []

    def fake_get_case_id(server_url, task_id, case_index):
        captured_server_urls.append(server_url)
        return f"{task_id}_{case_index}"

    monkeypatch.setattr(inference, "_get_case_id", fake_get_case_id)

    def fake_post(url, json, timeout):
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {"score": 1.0, "breakdown": {"priority_parity": 1.0}},
        )

    monkeypatch.setattr(inference.req, "post", fake_post)

    inference.main()
    output = capsys.readouterr().out

    assert "[START]" in output
    assert "[END]" in output
    assert len(captured_server_urls) == 22  # 11 tasks × 2 cases each
    assert all(url == inference.SERVER_URL for url in captured_server_urls)


def test_run_episode_step_exception_keeps_step_and_reward_counts_aligned(monkeypatch):
    reset_payload = {
        "done": False,
        "info": {"session_id": "sid-1"},
        "observation": {"patient_history": "h", "task_description": "d"},
    }

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = {"n": 0}

    def fake_post(url, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(reset_payload)
        raise RuntimeError("step failed")

    step_logs = []

    monkeypatch.setattr(inference.req, "post", fake_post)
    monkeypatch.setattr(inference, "call_llm", lambda *args, **kwargs: {"priority": "high"})
    monkeypatch.setattr(
        inference,
        "log_step",
        lambda step, action, reward, done, error: step_logs.append(
            {"step": step, "reward": reward, "done": done, "error": error}
        ),
    )

    last_reward, step_rewards, _ = inference.run_episode(
        client=object(),
        task_id="simple_triage",
        case_index=0,
        server_url="https://env.example.com",
    )

    assert len(step_logs) == 1
    assert len(step_rewards) == 1
    assert step_rewards[0] == 0.0
    assert last_reward == 0.0
