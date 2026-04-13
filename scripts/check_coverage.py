#!/usr/bin/env python3
"""
check_coverage.py — Pre-release coverage parity gate.

Verifies that every task defined in openenv.yaml has full coverage in:
  1. full_browser_test.py  — API answer dict + browser UI TASKS list
  2. random_agent_baseline.py — TASKS list + TASK_ACTION_FN
  3. live_verify.sh         — task-specific smoke section
  4. server/app.py          — web UI <select> dropdown option
  5. inference.py           — TASKS list
  6. train.py               — TRAINING_TASKS list

Run this before every staging deploy. Exit 1 if any gap found.

Usage:
  python scripts/check_coverage.py
  python scripts/check_coverage.py --root /path/to/medical_triage_env
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml


def load_tasks(root: Path) -> list[str]:
    """Read canonical task list from openenv.yaml."""
    with open(root / "openenv.yaml") as f:
        data = yaml.safe_load(f)
    return [t["id"] for t in data.get("tasks", [])]


def check_browser_test(root: Path, tasks: list[str]) -> list[str]:
    """Each task must appear in browser UI TASKS list AND have an answer dict or section."""
    gaps = []
    text = (root / "scripts/full_browser_test.py").read_text()

    for task in tasks:
        # Browser UI TASKS list
        if f'"{task}"' not in text and f"'{task}'" not in text:
            gaps.append(f"full_browser_test.py: task '{task}' not found anywhere")
            continue

        # Check for answer dict (e.g. IC_ANSWERS, SH_ANSWERS) or a test_ function
        has_answer_dict = re.search(rf'\b\w+_ANSWERS\s*=.*?{re.escape(task)}|\btest_{re.escape(task)}\b', text, re.DOTALL)
        # Check that it's in the browser UI TASKS list specifically
        ui_tasks_block = re.search(r'TASKS\s*=\s*\[(.*?)\]', text, re.DOTALL)
        in_ui_list = ui_tasks_block and task in ui_tasks_block.group(1)

        if not in_ui_list:
            gaps.append(f"full_browser_test.py: '{task}' missing from browser UI TASKS list")

        # Check for a dedicated API test section (answer dict or test function)
        has_section = (
            f"test_{task}" in text or
            re.search(rf'[A-Z]{{2,}}_ANSWERS\s*=\s*\{{.*?{re.escape(task)}', text, re.DOTALL) or
            re.search(rf'"{re.escape(task)}".*?reset\(', text, re.DOTALL) or
            task in ["demographic_fairness"]  # handled via grade-fairness endpoint
        )
        if not has_section:
            gaps.append(f"full_browser_test.py: '{task}' has no API test section (answer dict or test_* function)")

    return gaps


def check_random_baseline(root: Path, tasks: list[str]) -> list[str]:
    gaps = []
    text = (root / "scripts/random_agent_baseline.py").read_text()

    # Count occurrences: each task must appear at least twice (TASKS list + TASK_ACTION_FN)
    for task in tasks:
        count = text.count(f'"{task}"')
        if count < 2:
            if count == 0:
                gaps.append(f"random_agent_baseline.py: '{task}' missing from TASKS list and TASK_ACTION_FN")
            else:
                # Figure out which one is missing
                fn_block = re.search(r'TASK_ACTION_FN\s*=\s*\{([^}]+)\}', text)
                fn_text = fn_block.group(1) if fn_block else ""
                if task not in fn_text:
                    gaps.append(f"random_agent_baseline.py: '{task}' missing from TASK_ACTION_FN")
                else:
                    gaps.append(f"random_agent_baseline.py: '{task}' missing from TASKS list")

    return gaps


def check_live_verify(root: Path, tasks: list[str]) -> list[str]:
    gaps = []
    text = (root / "scripts/live_verify.sh").read_text()
    # Skip the 8 original tasks that are covered by the generic reset/step checks
    # New tasks (T9+) must have explicit smoke sections
    IMPLICIT_TASKS = {
        "simple_triage", "conflicting_vitals", "masked_deterioration",
        "demographic_fairness", "deteriorating_patient", "sepsis_bundle",
        "paediatric_triage", "medication_reconciliation",
    }
    for task in tasks:
        if task in IMPLICIT_TASKS:
            continue
        if task not in text:
            gaps.append(f"live_verify.sh: '{task}' has no smoke check section")
    return gaps


def check_web_ui_dropdown(root: Path, tasks: list[str]) -> list[str]:
    gaps = []
    text = (root / "server/app.py").read_text()
    dropdown = re.search(r'id="task-select">(.*?)</select>', text, re.DOTALL)
    dropdown_text = dropdown.group(1) if dropdown else ""
    for task in tasks:
        if f'value="{task}"' not in dropdown_text:
            gaps.append(f"server/app.py web UI dropdown: '{task}' option missing")
    return gaps


def check_inference(root: Path, tasks: list[str]) -> list[str]:
    gaps = []
    text = (root / "inference.py").read_text()
    for task in tasks:
        if f'"{task}"' not in text:
            gaps.append(f"inference.py: '{task}' missing from TASKS list")
    return gaps


def check_train(root: Path, tasks: list[str]) -> list[str]:
    gaps = []
    text = (root / "train.py").read_text()
    for task in tasks:
        if f'"{task}"' not in text:
            gaps.append(f"train.py: '{task}' missing from TRAINING_TASKS list")
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-release coverage parity check")
    parser.add_argument("--root", default=".", help="Path to medical_triage_env root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    tasks = load_tasks(root)

    print(f"\n[coverage] Checking {len(tasks)} tasks: {tasks}\n")

    all_gaps: list[str] = []
    checks = [
        ("full_browser_test.py",      check_browser_test),
        ("random_agent_baseline.py",  check_random_baseline),
        ("live_verify.sh",            check_live_verify),
        ("server/app.py UI dropdown", check_web_ui_dropdown),
        ("inference.py",              check_inference),
        ("train.py",                  check_train),
    ]

    for label, fn in checks:
        gaps = fn(root, tasks)
        if gaps:
            print(f"  ❌  {label}:")
            for g in gaps:
                print(f"       - {g}")
            all_gaps.extend(gaps)
        else:
            print(f"  ✅  {label}: all {len(tasks)} tasks covered")

    print()
    if all_gaps:
        print(f"[coverage] FAIL — {len(all_gaps)} gap(s) found. Fix before deploying to staging.\n")
        sys.exit(1)
    else:
        print(f"[coverage] PASS — all {len(tasks)} tasks fully covered across all artifacts.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
