#!/usr/bin/env python3
"""Headless browser smoke test for deployed Medical Triage UI.

Usage:
  python scripts/browser_ui_smoke.py --base-url https://<space>.hf.space

Prerequisites:
  pip install playwright
  playwright install chromium
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://kunalkachru23-scaler-hackathon-rl-medical-triage-env.hf.space"


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ModuleNotFoundError:
        print("[ui-smoke] Playwright is required.")
        print("[ui-smoke] Install with:")
        print("  python -m pip install playwright")
        print("  python -m playwright install chromium")
        raise SystemExit(2)
    return sync_playwright


def run(base_url: str, headless: bool = True) -> int:
    sync_playwright = _require_playwright()
    results: list[StepResult] = []
    base_url = base_url.rstrip("/")
    web_url = f"{base_url}/web"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.set_default_timeout(20_000)

        def check(name: str, fn):
            try:
                fn()
                results.append(StepResult(name=name, ok=True))
            except Exception as exc:  # noqa: BLE001 - test harness
                results.append(StepResult(name=name, ok=False, detail=str(exc)))

        def open_web():
            page.goto(web_url, wait_until="domcontentloaded")
            page.wait_for_selector("#task-select")
            page.wait_for_selector("#reset-btn")
            # Submit button is rendered only after a patient is loaded.
            page.wait_for_selector("#submit-btn", state="attached")
            page.wait_for_selector("#case-select-hint")

        check("load-web-and-core-controls", open_web)

        def simple_triage_flow():
            page.select_option("#task-select", "simple_triage")
            page.click("#reset-btn")
            page.wait_for_function(
                """() => {
                    const t = document.querySelector('#patient-history')?.textContent || '';
                    return t.length > 20 && !t.includes('Loading patient case');
                }"""
            )
            page.click("#ai-btn")
            page.wait_for_function(
                """() => (document.querySelector('#ai-status')?.textContent || '').includes('Filled by:')"""
            )
            page.click("#submit-btn")
            page.wait_for_selector("#result-section .score-big")
            score = (page.text_content("#result-section .score-big") or "").strip()
            if not score.endswith("%"):
                raise AssertionError(f"Missing percentage score in result: {score!r}")

        check("simple-triage-ai-submit", simple_triage_flow)

        def episode_history_demarcation():
            page.wait_for_function(
                """() => {
                    const txt = document.querySelector('#episode-log')?.textContent || '';
                    return txt.includes('Episode') && (txt.includes('started') || txt.includes('final reward'));
                }"""
            )

        check("episode-history-demarcation", episode_history_demarcation)

        def task_switch_clears_form():
            page.select_option("#task-select", "deteriorating_patient")
            page.click("#reset-btn")
            page.wait_for_selector("#btn-monitor")
            # Switch to a non-deteriorating task and verify old action buttons disappear.
            page.select_option("#task-select", "masked_deterioration")
            page.click("#reset-btn")
            page.wait_for_selector("#f-priority")
            if page.locator("#btn-monitor").count() != 0:
                raise AssertionError("Deteriorating action buttons leaked into non-deteriorating form.")

        check("task-switch-clears-stale-ui", task_switch_clears_form)

        def deteriorating_multiturn_flow():
            page.select_option("#task-select", "deteriorating_patient")
            page.click("#reset-btn")
            page.wait_for_selector("#btn-monitor")
            page.click("#btn-monitor")
            page.click("#submit-btn")
            # Either done immediately (some case/action combos) or continue prompt appears.
            continued = page.locator("text=Episode continues").count() > 0
            if continued:
                page.click("#btn-escalate")
                page.click("#submit-btn")
            page.wait_for_selector("#result-section .score-big")

        check("deteriorating-multiturn", deteriorating_multiturn_flow)

        def training_tab_state():
            page.click("#tab-training")
            page.wait_for_selector("#panel-training")
            page.click("text=Refresh")
            # Accept either empty-state or populated stats.
            page.wait_for_function(
                """() => {
                    const hint = document.querySelector('#training-empty-hint');
                    const ep = document.querySelector('#stat-episodes');
                    if (!hint || !ep) return false;
                    return hint.style.display !== 'none' || (ep.textContent || '').trim().length > 0;
                }"""
            )

        check("training-tab-refresh", training_tab_state)

        browser.close()

    print(f"[ui-smoke] URL: {web_url}")
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}{' :: ' + r.detail if r.detail else ''}")

    if failed:
        print(f"[ui-smoke] FAIL: {len(failed)} step(s) failed.")
        return 1
    print(f"[ui-smoke] PASS: {len(results)} checks succeeded.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless browser smoke test for deployed UI.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL, e.g. https://<space>.hf.space")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode for debugging.")
    args = parser.parse_args()
    return run(base_url=args.base_url, headless=not args.headed)


if __name__ == "__main__":
    raise SystemExit(main())
