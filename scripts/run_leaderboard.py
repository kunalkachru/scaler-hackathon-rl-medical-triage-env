#!/usr/bin/env python3
"""
Multi-model leaderboard runner for Medical Triage Environment v2.3.0.

Runs inference across all 11 tasks for each specified model,
then writes a comparison table to docs/LEADERBOARD.md.

Usage:
    python scripts/run_leaderboard.py
    python scripts/run_leaderboard.py --models "Llama-3.3-70B-Instruct,Qwen2.5-72B-Instruct"
    python scripts/run_leaderboard.py --out docs/LEADERBOARD.md --server-url http://localhost:8000
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_API_BASE   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
DEFAULT_HF_TOKEN   = os.getenv("HF_TOKEN", "")
DEFAULT_SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
DEFAULT_OUT        = "docs/LEADERBOARD.md"

MODELS = [
    ("meta-llama/Llama-3.3-70B-Instruct",          "Llama-3.3-70B"),
    ("Qwen/Qwen2.5-72B-Instruct",                   "Qwen2.5-72B"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",    "DeepSeek-R1-32B"),
]

TASKS_ORDER = [
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

TASK_DIFFICULTY = {
    "simple_triage":           "Easy",
    "conflicting_vitals":      "Medium",
    "masked_deterioration":    "Hard",
    "demographic_fairness":    "Medium",
    "deteriorating_patient":   "Hard",
    "sepsis_bundle":           "Hard",
    "paediatric_triage":       "Hard",
    "medication_reconciliation":"Hard",
    "icu_deterioration":       "Hard",
    "sbar_handover":           "Medium",
    "differential_diagnosis":  "Hard",
}

# ── helpers ──────────────────────────────────────────────────────────────────
def parse_inference_output(stdout: str) -> dict[str, list[float]]:
    """
    Parse [END] lines from inference.py stdout.
    Returns {task_id: [score, score, ...]} (one score per case run).
    """
    results: dict[str, list[float]] = {}
    # [END] success=true steps=1 score=0.9999 rewards=0.9999
    end_re = re.compile(r"\[END\].*?score=([\d.]+)")
    # [START] task=simple_triage:ST001 env=... model=...
    start_re = re.compile(r"\[START\] task=([\w]+):")

    current_task = None
    for line in stdout.splitlines():
        sm = start_re.search(line)
        if sm:
            current_task = sm.group(1)
        em = end_re.search(line)
        if em and current_task:
            score = float(em.group(1))
            results.setdefault(current_task, []).append(score)
    return results


def run_model(model_id: str, api_base: str, hf_token: str, server_url: str) -> dict[str, list[float]]:
    """Run inference.py for one model, return per-task scores."""
    print(f"\n{'='*60}")
    print(f"  Running: {model_id}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["MODEL_NAME"]   = model_id
    env["API_BASE_URL"] = api_base
    env["HF_TOKEN"]     = hf_token
    env["OPENAI_API_KEY"] = hf_token
    env["API_KEY"]      = hf_token
    env["SERVER_URL"]   = server_url

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        env=env,
        capture_output=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(f"  [WARN] inference.py exited with code {proc.returncode}")

    return parse_inference_output(proc.stdout)


def run_random_baseline(server_url: str) -> dict[str, list[float]]:
    """Run random_agent_baseline.py, return per-task scores."""
    print(f"\n{'='*60}")
    print(f"  Running: Random Baseline Agent")
    print(f"{'='*60}")

    proc = subprocess.run(
        [sys.executable, "scripts/random_agent_baseline.py",
         "--server-url", server_url, "--runs", "2"],
        capture_output=True, text=True,
    )

    results: dict[str, list[float]] = {}
    # Parse "task_id ... avg=X.XX" lines from random baseline output
    for line in proc.stdout.splitlines():
        m = re.search(r"([\w_]+)\s+.*?(\d+\.\d+)", line)
        if m:
            task = m.group(1)
            if task in TASKS_ORDER:
                score = float(m.group(2))
                results.setdefault(task, []).append(score)
    return results


def avg(scores: list[float]) -> float | None:
    return round(sum(scores) / len(scores), 4) if scores else None


def fmt(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


def write_leaderboard(
    scores_by_model: dict[str, dict[str, list[float]]],
    out_path: str,
    run_date: str,
) -> None:
    lines = [
        "# Multi-Model Leaderboard",
        "",
        f"**Environment:** Medical Triage Environment v2.3.0  ",
        f"**Run date:** {run_date}  ",
        f"**Server:** `http://localhost:8000` (Llama-3.3-70B-Instruct via HF Router)  ",
        "",
        "---",
        "",
        "## Per-Task Scores (avg across 2 cases)",
        "",
    ]

    model_labels = list(scores_by_model.keys())
    header = "| Task | Difficulty | " + " | ".join(model_labels) + " |"
    sep    = "|---|---|" + "|".join(["---"] * len(model_labels)) + "|"
    lines += [header, sep]

    overall: dict[str, list[float]] = {m: [] for m in model_labels}

    for task in TASKS_ORDER:
        diff = TASK_DIFFICULTY.get(task, "—")
        row_vals = []
        for m in model_labels:
            task_scores = scores_by_model[m].get(task, [])
            a = avg(task_scores)
            row_vals.append(fmt(a))
            if a is not None:
                overall[m].append(a)
        lines.append(f"| `{task}` | {diff} | " + " | ".join(row_vals) + " |")

    lines += [
        "",
        "## Overall Average",
        "",
        "| Model | Avg Score | Rank |",
        "|---|---|---|",
    ]

    ranked = sorted(
        [(m, avg(overall[m])) for m in model_labels if overall[m]],
        key=lambda x: x[1] or 0,
        reverse=True,
    )
    for rank, (m, score) in enumerate(ranked, 1):
        lines.append(f"| {m} | **{fmt(score)}** | #{rank} |")

    lines += [
        "",
        "---",
        "",
        "## Key Observations",
        "",
        "- `simple_triage` (Easy) — all models expected to perform well (≥0.80)",
        "- `conflicting_vitals` (Medium) — trap cases; frontier LLMs score ~0.37",
        "- `masked_deterioration` (Hard) — pharmacological masking; requires clinical reasoning",
        "- `sepsis_bundle` (Hard) — most demanding; requires structured Hour-1 Bundle knowledge",
        "- `icu_deterioration` (Hard) — SOFA reasoning; new in v2.3.0",
        "- `differential_diagnosis` (Hard) — must-not-miss weighting; new in v2.3.0",
        "",
        "---",
        "",
        "*Generated by `scripts/run_leaderboard.py` for Team Falcons — Meta × Scaler Hackathon 2026*",
    ]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\n[leaderboard] Written to {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model leaderboard runner")
    parser.add_argument("--models", default="",
                        help="Comma-separated HF model IDs (overrides built-in list)")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--hf-token", default=DEFAULT_HF_TOKEN)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random baseline agent")
    args = parser.parse_args()

    model_list = MODELS
    if args.models:
        model_list = [(m.strip(), m.strip().split("/")[-1]) for m in args.models.split(",")]

    run_date = time.strftime("%Y-%m-%d")
    scores_by_model: dict[str, dict[str, list[float]]] = {}

    for model_id, label in model_list:
        scores = run_model(model_id, args.api_base, args.hf_token, args.server_url)
        scores_by_model[label] = scores
        print(f"  [{label}] tasks scored: {list(scores.keys())}")

    if not args.skip_random:
        random_scores = run_random_baseline(args.server_url)
        scores_by_model["Random Baseline"] = random_scores

    write_leaderboard(scores_by_model, args.out, run_date)

    print("\n[leaderboard] Done.")
    print(f"[leaderboard] Output: {args.out}")


if __name__ == "__main__":
    main()
