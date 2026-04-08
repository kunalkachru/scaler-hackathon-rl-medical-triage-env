#!/usr/bin/env python3
"""Capture README-friendly demo assets: full-page screenshot + short screen recording.

Writes to repo-root assets/:
  - github-demo-screenshot.png
  - github-demo.webm
  - github-demo.mp4 (if ffmpeg is on PATH — H.264 for Safari / inline README)

Requires: playwright + chromium (same as browser_ui_smoke.py).

Usage:
  python scripts/capture_github_demo.py --base-url https://<space>.hf.space
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DEFAULT_BASE_URL = "https://kunalkachru23-medical-triage-env.hf.space"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture screenshot + short demo video for GitHub.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="HF Space base URL")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    web_url = f"{base}/web"

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ModuleNotFoundError:
        print("Install Playwright: python -m pip install playwright && python -m playwright install chromium")
        return 2

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shot_path = ASSETS_DIR / "github-demo-screenshot.png"
    video_path = ASSETS_DIR / "github-demo.webm"
    mp4_path = ASSETS_DIR / "github-demo.mp4"

    for p in (shot_path, video_path, mp4_path):
        if p.exists():
            p.unlink()

    before_videos = set(ASSETS_DIR.glob("*.webm"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(ASSETS_DIR),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.set_default_timeout(45_000)

        page.goto(web_url, wait_until="domcontentloaded")
        page.wait_for_selector("#task-select")
        time.sleep(1.5)

        page.select_option("#task-select", "simple_triage")
        page.locator("button[onclick='resetEnv()']").click()
        page.wait_for_function(
            """() => {
                const t = document.querySelector('#patient-history')?.textContent || '';
                return t.length > 20 && !t.includes('Loading patient case');
            }"""
        )
        time.sleep(1.0)

        page.click("#ai-btn")
        page.wait_for_function(
            """() => (document.querySelector('#ai-status')?.textContent || '').includes('Filled by:')"""
        )
        time.sleep(1.0)

        page.locator("button[onclick='submitAction()']").click()
        page.wait_for_selector("#result-section .score-big")
        time.sleep(2.0)

        page.click("#tab-training")
        page.wait_for_selector("#panel-training")
        page.locator("button[onclick='loadTrainingData()']").click()
        time.sleep(2.5)

        page.screenshot(path=str(shot_path), full_page=True)

        context.close()
        browser.close()

    new_videos = set(ASSETS_DIR.glob("*.webm")) - before_videos
    if len(new_videos) != 1:
        print(f"Expected exactly one new .webm in {ASSETS_DIR}, got: {new_videos}")
        return 1
    shutil.move(str(next(iter(new_videos))), str(video_path))

    print(f"Wrote {shot_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {video_path.relative_to(REPO_ROOT)}")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(mp4_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Wrote {mp4_path.relative_to(REPO_ROOT)}")
    except FileNotFoundError:
        print("Optional: install ffmpeg and re-run to also write github-demo.mp4 (better Safari / GitHub viewing).")
    except subprocess.CalledProcessError as exc:
        print("ffmpeg failed; github-demo.mp4 not updated:", exc.stderr[-500:] if exc.stderr else exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
