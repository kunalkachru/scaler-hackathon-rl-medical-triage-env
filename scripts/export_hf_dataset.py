"""
scripts/export_hf_dataset.py — HF Dataset Export
==================================================
Exports (observation, action, reward) triples from the Medical Triage Environment
to Hugging Face Datasets format (JSONL).

This script:
1. Runs inference across all 8 tasks (all cases) using the configured LLM
2. Records each (patient_history, task_id, action, reward, breakdown) tuple
3. Saves as JSONL to datasets/medical_triage_triples.jsonl
4. Optionally pushes to Hugging Face Hub if HF_DATASET_REPO is set

Usage:
  python scripts/export_hf_dataset.py [--output PATH] [--push-to-hub]

Environment variables:
  API_BASE_URL      — LLM inference endpoint
  MODEL_NAME        — Model identifier
  HF_TOKEN          — Hugging Face API key
  SERVER_URL        — Override env server URL (default http://localhost:8000)
  HF_DATASET_REPO   — HF Hub dataset repo (e.g. kunalkachru23/medical-triage-triples)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests as req

# ── Config ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")

OUTPUT_DIR  = Path("datasets")
OUTPUT_FILE = OUTPUT_DIR / "medical_triage_triples.jsonl"

# All 8 tasks with representative cases per task
EXPORT_SCHEDULE = [
    ("simple_triage",            list(range(10))),   # 10 cases
    ("conflicting_vitals",       list(range(8))),    # 8 cases
    ("masked_deterioration",     list(range(10))),   # 10 cases
    ("demographic_fairness",     list(range(12))),   # 12 cases
    ("deteriorating_patient",    list(range(7))),    # 7 cases
    ("sepsis_bundle",            list(range(4))),    # 4 cases
    ("paediatric_triage",        list(range(6))),    # 6 cases
    ("medication_reconciliation", list(range(6))),   # 6 cases
]


def wait_for_server(url: str, retries: int = 20) -> bool:
    for _ in range(retries):
        try:
            if req.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_mock_episode(task_id: str, case_index: int) -> dict | None:
    """
    Run one episode with a rule-based mock action (no LLM required).
    Used when no LLM is configured.
    """
    try:
        r = req.post(f"{SERVER_URL}/reset",
                     json={"task_id": task_id, "case_index": case_index},
                     timeout=10)
        r.raise_for_status()
        reset_data = r.json()
    except Exception as e:
        print(f"  [Reset error]: {e}")
        return None

    obs = reset_data["observation"]
    session_id = reset_data.get("info", {}).get("session_id")

    # Mock action based on task type
    mock_actions = {
        "simple_triage":           {"priority": "high", "news2_score": 7, "critical_sign": "respiratory_rate", "recommended_action": "urgent_review", "confidence": 0.7},
        "conflicting_vitals":      {"priority": "high", "critical_sign": "spo2", "misleading_signs": ["heart_rate"], "condition": "hypoxia", "recommended_action": "urgent_review", "confidence": 0.6},
        "masked_deterioration":    {"priority": "high", "masking_drug_or_condition": "bisoprolol", "masked_sign": "heart_rate", "critical_clues": ["lactate"], "condition": "sepsis", "recommended_action": "urgent_review", "confidence": 0.5},
        "demographic_fairness":    {"priority": "high", "critical_sign": "systolic_bp", "recommended_action": "urgent_review", "confidence": 0.8},
        "deteriorating_patient":   {"action": "escalate", "confidence": 0.7, "rationale": "Trend worsening"},
        "sepsis_bundle":           {"priority": "critical", "bundle_elements": ["blood_cultures", "broad_spectrum_antibiotics", "iv_fluid_bolus", "lactate_measurement"], "antibiotic_choice": "piperacillin_tazobactam", "fluid_volume_ml": 2000, "vasopressor_indicated": False, "confidence": 0.6},
        "paediatric_triage":       {"priority": "high", "age_group": "school_age", "pews_score": 5, "critical_sign": "spo2", "recommended_action": "urgent_review", "confidence": 0.6},
        "medication_reconciliation": {"issues_found": ["drug_interaction"], "severity": "high", "requires_pharmacist": True, "recommended_action": "withhold_drug", "confidence": 0.5},
    }
    action = mock_actions.get(task_id, {"priority": "medium", "confidence": 0.5})

    payload: dict = {"action": action}
    if session_id:
        payload["session_id"] = session_id

    try:
        r = req.post(f"{SERVER_URL}/step", json=payload, timeout=15)
        r.raise_for_status()
        step_data = r.json()
    except Exception as e:
        print(f"  [Step error]: {e}")
        return None

    step_obs = step_data["observation"]
    return {
        "task_id":         task_id,
        "case_index":      case_index,
        "case_id":         obs.get("case_id", ""),
        "patient_history": obs.get("patient_history", ""),
        "task_description": obs.get("task_description", ""),
        "action":          action,
        "reward":          round(step_data.get("reward", 0.0), 4),
        "score_breakdown": step_obs.get("score_breakdown") or {},
        "feedback":        step_obs.get("feedback") or "",
        "model":           "mock_rule_based",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Medical Triage (obs, action, reward) triples to JSONL")
    parser.add_argument("--output", default=str(OUTPUT_FILE),
                        help=f"Output JSONL path (default: {OUTPUT_FILE})")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HF Hub after export (requires HF_DATASET_REPO env var)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start server if localhost
    server_proc = None
    if "localhost" in SERVER_URL:
        print("  Starting environment server...", end=" ", flush=True)
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not wait_for_server(SERVER_URL):
            print("FAILED — server did not start")
            sys.exit(1)
        print("ready")

    records: list[dict] = []
    total = sum(len(cases) for _, cases in EXPORT_SCHEDULE)
    done = 0

    print(f"\nExporting {total} episodes across 8 tasks...")
    print(f"Output: {output_path}\n")

    for task_id, case_indices in EXPORT_SCHEDULE:
        print(f"  [{task_id}] {len(case_indices)} cases...")
        for ci in case_indices:
            record = run_mock_episode(task_id, ci)
            if record:
                records.append(record)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total} episodes exported")

    # Write JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✓ Exported {len(records)} records to {output_path}")
    print(f"  Tasks: {len(EXPORT_SCHEDULE)}")
    print(f"  Format: JSONL — each line is one (observation, action, reward) triple")

    # Write dataset card
    card_path = output_path.parent / "README.md"
    card_path.write_text(
        "# Medical Triage RL Triples\n\n"
        "Dataset of `(observation, action, reward)` triples from the "
        "Medical Triage OpenEnv environment.\n\n"
        "## Fields\n\n"
        "| Field | Description |\n"
        "|-------|-------------|\n"
        "| `task_id` | One of 8 clinical tasks |\n"
        "| `case_id` | Unique patient case identifier |\n"
        "| `patient_history` | Full clinical presentation text |\n"
        "| `task_description` | Agent instruction prompt |\n"
        "| `action` | Agent response JSON |\n"
        "| `reward` | Scalar reward in (0.0001, 0.9999) |\n"
        "| `score_breakdown` | Per-dimension score breakdown |\n"
        "| `model` | Model or agent type used |\n\n"
        "## Tasks\n\n"
        "simple_triage, conflicting_vitals, masked_deterioration, demographic_fairness, "
        "deteriorating_patient, sepsis_bundle, paediatric_triage, medication_reconciliation\n\n"
        "## Source\n\n"
        "Generated by `scripts/export_hf_dataset.py` from "
        "`kunalkachru23/medical-triage-env`.\n"
    )
    print(f"  Dataset card written to: {card_path}")

    # Optional HF Hub push
    if args.push_to_hub:
        repo = HF_DATASET_REPO or "kunalkachru23/medical-triage-triples"
        if not API_KEY:
            print("\n[!] HF_TOKEN not set — skipping Hub push")
        else:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=API_KEY)
                api.upload_file(
                    path_or_fileobj=str(output_path),
                    path_in_repo="medical_triage_triples.jsonl",
                    repo_id=repo,
                    repo_type="dataset",
                )
                api.upload_file(
                    path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=repo,
                    repo_type="dataset",
                )
                print(f"\n✓ Pushed to HF Hub: https://huggingface.co/datasets/{repo}")
            except ImportError:
                print("\n[!] huggingface_hub not installed — run: pip install huggingface_hub")
            except Exception as e:
                print(f"\n[!] Hub push failed: {e}")

    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
