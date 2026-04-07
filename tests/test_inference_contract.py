import types

import inference


def test_validate_required_env_reports_missing_fields(monkeypatch):
    monkeypatch.setattr(inference, "API_BASE_URL", "")
    monkeypatch.setattr(inference, "MODEL_NAME", "")
    monkeypatch.setattr(inference, "API_KEY", "")

    missing = inference._validate_required_env()

    assert "API_BASE_URL" in missing
    assert "MODEL_NAME" in missing
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

    assert "RESULTS SUMMARY" in output
    assert len(captured_server_urls) == 10
    assert all(url == inference.SERVER_URL for url in captured_server_urls)
