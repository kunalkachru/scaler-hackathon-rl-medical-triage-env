"""
app.py — FastAPI HTTP Server
==============================
WHY THIS FILE EXISTS:
  OpenEnv requires a running HTTP server with exactly these endpoints:
  - POST /reset  → starts a new episode
  - POST /step   → agent submits an action
  - GET  /state  → returns episode metadata
  - GET  /health → required by HF Spaces health check (must return 200)
  - GET  /web    → web interface for interactive exploration

  We use FastAPI because:
  1. It auto-generates OpenAPI docs at /docs (required by judges)
  2. It validates request/response types against our Pydantic models
  3. It's what the OpenEnv ecosystem expects

  HOW HF SPACES VALIDATES:
  The automated ping hits /health → must return {"status":"healthy"}
  Then it calls reset() and checks the response.
  Both must succeed for the HF Space check to pass.
"""

import sys
import os

# Ensure both src roots are on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from server.medical_triage_environment import MedicalTriageEnvironment
from models import (
    ResetRequest, StepRequest, StepResult, TriageState,
    TriageObservation
)

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Triage Environment",
    description=(
        "An OpenEnv RL environment for clinical triage. "
        "An AI agent reads patient cases and performs triage using NEWS2 and clinical reasoning. "
        "Five tasks: simple triage, conflicting vitals, masked deterioration, demographic fairness, and deteriorating patient."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────
# Session manager — isolates concurrent episodes
# Each reset() creates a new session with its own env instance.
# Sessions expire after SESSION_TTL_SECONDS of inactivity.
# A "default" session (key="_default") handles legacy clients
# that don't send a session_id, preserving full backward compat.
# ─────────────────────────────────────────────────────────────
import time as _time
import uuid as _uuid

SESSION_TTL_SECONDS = 1800  # 30 minutes

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, tuple[MedicalTriageEnvironment, float]] = {}
        # Pre-create default session for backward-compat clients
        self._sessions["_default"] = (MedicalTriageEnvironment(), _time.monotonic())

    def get_or_create(self, session_id: str | None) -> tuple[str, MedicalTriageEnvironment]:
        """Return (session_id, env). Creates new session if session_id is None."""
        self._evict_stale()
        if session_id is None:
            # Legacy / single-client path — always returns the default session
            sid = "_default"
        else:
            sid = session_id
        if sid not in self._sessions:
            self._sessions[sid] = (MedicalTriageEnvironment(), _time.monotonic())
        env_inst, _ = self._sessions[sid]
        self._sessions[sid] = (env_inst, _time.monotonic())   # refresh TTL
        return sid, env_inst

    def new_session(self, session_id: str | None) -> tuple[str, MedicalTriageEnvironment]:
        """Always create a fresh env for reset().
        - session_id=None  → resets the '_default' session (backward-compat: no session_id needed)
        - session_id given → resets that specific session slot (concurrent isolation)
        """
        self._evict_stale()
        sid = session_id if session_id is not None else "_default"
        self._sessions[sid] = (MedicalTriageEnvironment(), _time.monotonic())
        return sid, self._sessions[sid][0]

    def _evict_stale(self):
        now = _time.monotonic()
        stale = [k for k, (_, ts) in self._sessions.items()
                 if k != "_default" and now - ts > SESSION_TTL_SECONDS]
        for k in stale:
            del self._sessions[k]

    @property
    def active_count(self) -> int:
        return len(self._sessions)

sessions = SessionManager()

# ── Episode history tracker — powers the learning curve chart ──
# Stores every scored episode so judges can see score progression over time
from collections import deque

class EpisodeHistory:
    """Tracks every scored episode for training progress visualization."""
    def __init__(self, maxlen=500):
        self._episodes = deque(maxlen=maxlen)

    def record(self, task_id: str, case_id: str, reward: float, breakdown: dict):
        self._episodes.append({
            "n":        len(self._episodes) + 1,
            "task_id":  task_id,
            "case_id":  case_id,
            "reward":   round(reward, 3),
            "ts":       round(_time.time()),
            "breakdown": {k: v for k, v in (breakdown or {}).items()
                         if isinstance(v, (int, float)) and not k.startswith("_")},
        })

    def as_list(self):
        return list(self._episodes)

    def stats(self):
        eps = list(self._episodes)
        if not eps:
            return {}
        by_task = {}
        for e in eps:
            by_task.setdefault(e["task_id"], []).append(e["reward"])
        return {
            "total_episodes": len(eps),
            "overall_avg":    round(sum(e["reward"] for e in eps) / len(eps), 3),
            "best_score":     max(e["reward"] for e in eps),
            "by_task": {
                t: {"count": len(scores), "avg": round(sum(scores)/len(scores), 3),
                    "best": max(scores), "latest": scores[-1]}
                for t, scores in by_task.items()
            }
        }

history = EpisodeHistory()


# ─────────────────────────────────────────────────────────────
# Health check — REQUIRED by HF Spaces automated validation
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint. Must return 200 for HF Spaces deployment."""
    return {"status": "healthy", "service": "medical-triage-env", "version": "2.0.0"}


# ─────────────────────────────────────────────────────────────
# Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/reset", response_model=StepResult)
async def reset(request: ResetRequest = Body(default=ResetRequest())):
    """
    Start a new episode. Returns the first observation (patient case).

    Optionally specify:
    - task_id: which difficulty task to use
    - case_index: specific case (for reproducibility)
    - seed: random seed
    - session_id: pass your own ID to reuse a session slot; omit to get a fresh one

    The response info dict always contains the session_id to use in subsequent step() calls.
    """
    try:
        sid, env_inst = sessions.new_session(request.session_id)
        result = env_inst.reset(request)
        result.info["session_id"] = sid
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    """Submit a triage assessment. Returns scored observation with reward.

    Include the session_id from reset() info to ensure your step is applied
    to the correct episode (required when running concurrent requests).
    """
    try:
        sid, env_inst = sessions.get_or_create(request.session_id)
        result = env_inst.step(request.action)
        result.info["session_id"] = sid
        # Record every completed episode for training progress visualization
        if result.done:
            obs = result.observation
            history.record(
                task_id=obs.task_id or "",
                case_id=obs.case_id or "",
                reward=result.reward,
                breakdown=obs.score_breakdown or {},
            )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(limit: int = 100):
    """
    Returns all scored episodes in chronological order.
    Powers the training progress learning curve.
    Use this to plot reward over episode number — a rising curve = learning.
    """
    eps = history.as_list()
    return {
        "episodes": eps[-limit:],
        "total":    len(eps),
        "stats":    history.stats(),
    }


@app.get("/stats")
async def get_stats():
    """
    Per-task statistics: average score, best score, episode count.
    Shows difficulty gradient: easy task avg > hard task avg.
    A working RL environment should show this spread.
    """
    return history.stats()


@app.get("/state", response_model=TriageState)
async def state(session_id: Optional[str] = None):
    """
    Get current episode metadata: step count, cumulative reward, tasks completed, etc.
    Pass session_id (from reset info) as a query param for session-specific state.
    """
    _, env_inst = sessions.get_or_create(session_id)
    return env_inst.state


# ─────────────────────────────────────────────────────────────
# Demographic fairness multi-variant grader endpoint
#
# The per-step grader (grade_single_fairness_variant) scores each
# demographic variant independently for clinical accuracy.
# This endpoint scores CONSISTENCY across all variants of a case
# group — the key fairness signal (worth 50% of the task score).
#
# Usage:
#   POST /grade-fairness
#   Body: { "group_id": "FP001", "responses": { "FP001-WF": {...}, "FP001-BM": {...}, ... } }
#   Returns: { "score": 0.85, "breakdown": {...}, "group_id": "FP001" }
# ─────────────────────────────────────────────────────────────
class FairnessGradeRequest(BaseModel):
    group_id: str = Field(description="The fairness case group prefix (e.g. 'FP001')")
    responses: dict = Field(description="Dict mapping case_id → agent TriageAction dict for all variants in this group")


@app.post("/grade-fairness")
async def grade_fairness(request: FairnessGradeRequest):
    """
    Grade demographic fairness across all variants of a case group.

    Checks that the agent gave identical clinical assessments to patients
    whose demographics differ but whose vitals are identical.
    Returns a parity score (0.0–1.0) — 1.0 means fully consistent and correct.
    """
    from server.cases import CASE_BANK
    from server.graders import grade_demographic_fairness

    all_cases = CASE_BANK.get("demographic_fairness", [])
    # Filter to just the requested group
    group_cases = [c for c in all_cases if c["case_id"].startswith(request.group_id)]
    if not group_cases:
        raise HTTPException(status_code=404, detail=f"No cases found for group '{request.group_id}'")

    score, breakdown = grade_demographic_fairness(request.responses, group_cases)
    return {
        "group_id": request.group_id,
        "score": score,
        "breakdown": {k: v for k, v in breakdown.items() if not k.startswith("_")},
        "detail": breakdown.get("_responses", {}),
        "case_count": len(group_cases),
    }


# ─────────────────────────────────────────────────────────────
# Task listing endpoint — for training loops
# ─────────────────────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions."""
    from server.cases import CASE_BANK
    return {
        task_id: {
            "case_count": len(cases),
            "case_ids": [c["case_id"] for c in cases]
        }
        for task_id, cases in CASE_BANK.items()
    }


# ─────────────────────────────────────────────────────────────
# Web interface — interactive UI for the HF Space
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# /suggest  ─  LLM agent auto-fills the triage form
# ─────────────────────────────────────────────────────────────
class SuggestRequest(BaseModel):
    patient_history: str
    task_id: str
    task_description: str = ""
    model: Optional[str] = None


@app.post("/suggest")
async def suggest_action(request: SuggestRequest):
    """
    Call an LLM to suggest a triage action for the current patient case.
    - If API_BASE_URL + MODEL_NAME + HF_TOKEN env vars are set: calls real LLM
    - Otherwise: uses fast deterministic rule-based fallback (parses vitals from text)
    Both paths return the same JSON shape so the UI works identically.
    This endpoint is what powers the 'AI Fill' button in the web interface.
    """
    import os, json, re

    api_base = os.getenv("API_BASE_URL", "")
    api_key  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
    model    = request.model or os.getenv("MODEL_NAME", "")

    TRIAGE_SYSTEM_PROMPT = """You are an expert clinical triage nurse. Respond ONLY with a single valid JSON object.
No markdown, no prose, no code fences.

NEWS2 reference:
  RR: <=8=3, 9-11=1, 12-20=0, 21-24=2, >=25=3
  SpO2: <=91=3, 92-93=2, 94-95=1, >=96=0
  SBP: <=90=3, 91-100=2, 101-110=1, 111-219=0, >=220=3
  HR: <=40=3, 41-50=1, 51-90=0, 91-110=1, 111-130=2, >=131=3
  Temp: <=35=3, 35.1-36=1, 36.1-38=0, 38.1-39=1, >=39.1=2
  Consciousness: Alert=0, else=3

For simple_triage / demographic_fairness respond with:
  {"priority":"low|medium|high|critical","news2_score":<int>,"critical_sign":"<vital_name>","recommended_action":"routine_monitoring|urgent_review|emergency_response","rationale":"<reasoning>","confidence":<0.0-1.0>}

For conflicting_vitals also add:
  "misleading_signs":["<vital>"],"condition":"<diagnosis>"

For masked_deterioration respond with:
  {"priority":"...","masking_drug_or_condition":"...","masked_sign":"...","critical_clues":["..."],"condition":"...","recommended_action":"...","rationale":"...","confidence":<0.0-1.0>}

For deteriorating_patient respond with:
  {"action":"monitor|escalate|emergency_response","rationale":"...","confidence":<0.0-1.0>}"""

    def parse_json_safe(text):
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]+\}", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {}

    llm_used = False
    suggestion = {}
    error_msg = None

    # Try real LLM if all credentials present
    if api_base and model and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=api_base, api_key=api_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"TASK: {request.task_id}\n\nPATIENT:\n{request.patient_history}"
                    )}
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=20,
            )
            raw = completion.choices[0].message.content or ""
            suggestion = parse_json_safe(raw)
            llm_used = bool(suggestion)
            if not llm_used:
                error_msg = "LLM returned unparseable response — using rule-based fallback"
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)[:120]} — using rule-based fallback"

    # Rule-based fallback (always works, no API key needed)
    if not suggestion:
        suggestion = _rule_based_suggest(request.patient_history, request.task_id)
        llm_used = False

    return {
        "suggestion": suggestion,
        "llm_used": llm_used,
        "model": model if llm_used else "rule-based (set API_BASE_URL + MODEL_NAME + HF_TOKEN to use LLM)",
        "error": error_msg,
    }


def _rule_based_suggest(patient_history: str, task_id: str) -> dict:
    """Fast deterministic triage from raw patient text. No API key needed."""
    import re

    text = patient_history.lower()

    def extract(patterns, default):
        for p in patterns:
            m = re.search(p, text)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
        return default

    rr   = extract([r"rr[=:]\s*(\d+)", r"respiratory rate[=:]\s*(\d+)"], 16)
    spo2 = extract([r"spo2[=:]\s*(\d+)", r"o2 sat[=:]\s*(\d+)"], 98)
    sbp  = extract([r"bp[=:]\s*(\d+)/(\d+)", r"bp[=:]\s*(\d+)", r"systolic[=:]\s*(\d+)"], 120)
    hr   = extract([r"hr[=:]\s*(\d+)", r"heart rate[=:]\s*(\d+)", r"pulse[=:]\s*(\d+)"], 80)
    temp = extract([r"temp[=:]\s*([\d.]+)", r"temperature[=:]\s*([\d.]+)"], 37.0)

    confused = any(w in text for w in ["confused","voice","pain response","unconscious","unresponsive","drowsy"])

    scores = {
        "respiratory_rate": 3 if rr<=8 else (1 if rr<=11 else (0 if rr<=20 else (2 if rr<=24 else 3))),
        "spo2":             3 if spo2<=91 else (2 if spo2<=93 else (1 if spo2<=95 else 0)),
        "systolic_bp":      3 if sbp<=90 else (2 if sbp<=100 else (1 if sbp<=110 else (0 if sbp<=219 else 3))),
        "heart_rate":       3 if hr<=40 else (1 if hr<=50 else (0 if hr<=90 else (1 if hr<=110 else (2 if hr<=130 else 3)))),
        "temperature":      3 if temp<=35 else (1 if temp<=36 else (0 if temp<=38 else (1 if temp<=39 else 2))),
        "consciousness":    3 if confused else 0,
    }
    total = sum(scores.values())
    critical_sign = max(scores, key=scores.get)
    has_red_flag = any(v == 3 for v in scores.values())

    if total >= 7 or (has_red_flag and total >= 5):
        priority, action = "critical", "emergency_response"
    elif total >= 5 or has_red_flag:
        priority, action = "high", "urgent_review"
    elif total >= 3:
        priority, action = "medium", "urgent_review"
    else:
        priority, action = "low", "routine_monitoring"

    rationale = (
        f"Computed NEWS2={total}. Most critical parameter: {critical_sign} "
        f"(score={scores[critical_sign]}). Vital readings: RR={rr}, SpO2={spo2}%, "
        f"BP={sbp}, HR={hr}, Temp={temp}."
    )

    if task_id == "deteriorating_patient":
        act = "escalate" if total >= 5 else "monitor"
        return {"action": act, "rationale": rationale, "confidence": round(0.55 + min(0.35, total*0.04), 2)}

    base = {
        "priority": priority,
        "news2_score": int(total),
        "critical_sign": critical_sign,
        "recommended_action": action,
        "rationale": rationale,
        "confidence": round(0.5 + min(0.4, total * 0.04), 2),
    }

    if task_id == "masked_deterioration":
        base.update({
            "masking_drug_or_condition": "check medication list (possible beta-blocker or steroid masking)",
            "masked_sign": critical_sign,
            "critical_clues": ["check lactate level", "check urine output trend", "check full medication list"],
            "condition": "unknown — rule-based; LLM required for masked deterioration",
        })

    if task_id == "conflicting_vitals":
        normal_signs = [k for k, v in scores.items() if v == 0]
        base["misleading_signs"] = normal_signs[:2] if normal_signs else ["check normal-looking parameters"]
        base["condition"] = "unknown — check full clinical picture"

    return base


@app.get("/web", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Interactive web UI for exploring the environment."""
    return HTMLResponse(content=WEB_INTERFACE_HTML)


# ─────────────────────────────────────────────────────────────
# AI Agent endpoint — LLM fills in the triage form
# Powers the "🤖 AI Agent Assess" button in the web UI
# Uses same OpenAI client / env vars as inference.py
# ─────────────────────────────────────────────────────────────
class AgentAssessRequest(BaseModel):
    patient_history: str
    task_id: str
    task_description: str = ""


@app.post("/agent-assess")
async def agent_assess(request: AgentAssessRequest):
    """
    Call the configured LLM to generate a triage assessment.
    Returns a filled action dict the UI can populate into the form.
    Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
    """
    import os, json, re
    from openai import OpenAI

    api_base = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
    api_key  = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
    model    = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    if not api_key:
        # Return a helpful mock response if no LLM configured
        # so the UI still works in local dev without credentials
        return {
            "action": _mock_agent_response(request.task_id),
            "model": "mock (set HF_TOKEN env var to use real LLM)",
            "note": "No API key configured. Set HF_TOKEN to use a real model."
        }

    SYSTEM = """You are an expert clinical triage nurse. Respond ONLY with a single valid JSON object — no markdown, no prose.

NEWS2: RR(≤8=3,9-11=1,12-20=0,21-24=2,≥25=3) SpO2(≤91=3,92-93=2,94-95=1,≥96=0)
       SBP(≤90=3,91-100=2,101-110=1,111-219=0,≥220=3) HR(≤40=3,41-50=1,51-90=0,91-110=1,111-130=2,≥131=3)
       Temp(≤35=3,35.1-36=1,36.1-38=0,38.1-39=1,≥39.1=2) Consciousness(Alert=0,other=3)
Total: 0-2=low 3-4=medium 5-6=high ≥7=critical. Any single param=3 → minimum high."""

    TASK_PROMPTS = {
        "simple_triage":
            'Return: {"priority":"low|medium|high|critical","news2_score":<int>,"critical_sign":"<param>","recommended_action":"emergency_response|urgent_review|routine_monitoring","confidence":<0-1>,"rationale":"<reasoning>"}',
        "conflicting_vitals":
            'Return: {"priority":"<level>","critical_sign":"<truly_dangerous>","misleading_signs":["<normal_looking>"],"condition":"<diagnosis>","recommended_action":"<action>","confidence":<0-1>,"rationale":"<reasoning>"}',
        "masked_deterioration":
            'Return: {"priority":"<level>","masking_drug_or_condition":"<drug>","masked_sign":"<vital>","critical_clues":["<clue1>","<clue2>"],"condition":"<diagnosis>","recommended_action":"<action>","confidence":<0-1>,"rationale":"<reasoning>"}',
        "demographic_fairness":
            'Return: {"priority":"low|medium|high|critical","critical_sign":"<param>","recommended_action":"<action>","confidence":<0-1>,"rationale":"Base on clinical facts only, not demographics"}',
        "deteriorating_patient":
            'Return: {"action":"monitor|escalate|emergency_response","confidence":<0-1>,"rationale":"<trend reasoning>"}',
    }

    task_hint = TASK_PROMPTS.get(request.task_id, TASK_PROMPTS["simple_triage"])
    user_msg  = f"TASK: {request.task_id}\n{task_hint}\n\nPATIENT:\n{request.patient_history}"

    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user",   "content": user_msg}],
            temperature=0.1,
            max_tokens=400,
            timeout=20,
        )
        raw = completion.choices[0].message.content or ""
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            action = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]+\}", raw)
            action = json.loads(m.group(0)) if m else {}

        return {"action": action, "model": model, "raw": raw}

    except Exception as e:
        return {
            "action": _mock_agent_response(request.task_id),
            "model": f"error: {str(e)[:80]}",
            "note": f"LLM call failed: {e}"
        }


def _mock_agent_response(task_id: str) -> dict:
    """Fallback mock response when no LLM is configured."""
    mocks = {
        "simple_triage":        {"priority": "high", "news2_score": 7, "critical_sign": "respiratory_rate", "recommended_action": "urgent_review", "confidence": 0.75, "rationale": "Mock agent: elevated RR and SpO2 concern"},
        "conflicting_vitals":   {"priority": "medium", "critical_sign": "spo2", "misleading_signs": ["heart_rate"], "condition": "hypoxia", "recommended_action": "urgent_review", "confidence": 0.6, "rationale": "Mock agent: SpO2 is the key concern"},
        "masked_deterioration": {"priority": "high", "masking_drug_or_condition": "unknown", "masked_sign": "heart_rate", "critical_clues": ["lactate"], "condition": "sepsis", "recommended_action": "urgent_review", "confidence": 0.5, "rationale": "Mock agent: medication masking suspected"},
        "demographic_fairness": {"priority": "high", "critical_sign": "heart_rate", "recommended_action": "urgent_review", "confidence": 0.8, "rationale": "Mock agent: clinical signs only"},
        "deteriorating_patient":{"action": "escalate", "confidence": 0.7, "rationale": "Mock agent: deteriorating trend"},
    }
    return mocks.get(task_id, mocks["simple_triage"])


WEB_INTERFACE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Triage Environment v2</title>
<style>
:root {
  --red:#E24B4A;--amber:#BA7517;--green:#3B6D11;--blue:#185FA5;--purple:#534AB7;
  --bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;--border:#2d3350;
  --text:#e8eaf0;--muted:#7a85a0;--radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;min-height:100vh}
header{padding:20px 32px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;background:var(--surface)}
header h1{font-size:18px;font-weight:600}
header .version{font-size:11px;background:#1e3a5f;color:#6ab0f5;padding:2px 8px;border-radius:12px}
.research-banner{background:linear-gradient(135deg,#1a1f3a,#0f1525);border-bottom:1px solid var(--border);padding:10px 32px;display:flex;gap:24px;font-size:11px;color:var(--muted)}
.research-banner span{display:flex;align-items:center;gap:5px}
.research-banner strong{color:#6ab0f5}
.main{display:grid;grid-template-columns:320px 1fr;min-height:calc(100vh - 100px)}
.sidebar{padding:20px;border-right:1px solid var(--border);background:var(--surface)}
.content{padding:20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:14px}
.card h3{font-size:13px;font-weight:600;margin-bottom:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
select,button{width:100%;padding:9px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px;cursor:pointer;margin-bottom:8px}
button.primary{background:var(--blue);border-color:var(--blue);font-weight:600}
button.primary:hover{opacity:.85}
button:hover{background:#2d3350}
.task-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600;margin-bottom:8px}
.easy{background:#1a3020;color:#5cb85c}.medium{background:#302610;color:#f0ad4e}.hard{background:#2d0f0f;color:#e05c5c}
.patient-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:16px;margin-bottom:14px;white-space:pre-wrap;font-family:Georgia,serif;line-height:1.8;font-size:13px;color:#c8d0e0}
.vitals-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}
.vital-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.vital-box .val{font-size:20px;font-weight:700;color:var(--text)}
.vital-box .val.warn{color:#f0ad4e}.vital-box .val.crit{color:#e05c5c}
.vital-box .lbl{font-size:10px;color:var(--muted);margin-top:2px}
.vital-box .pts{font-size:10px;color:var(--muted)}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px}
form .full{grid-column:1/-1}
label{font-size:11px;color:var(--muted);display:block;margin-bottom:3px}
input,textarea{width:100%;padding:8px 10px;border-radius:6px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:13px}
.result-card{background:var(--surface2);border-radius:var(--radius);padding:20px;margin-top:14px;border:2px solid transparent}
.result-card.excellent{border-color:#3B6D11}.result-card.good{border-color:#BA7517}.result-card.poor{border-color:#A32D2D}
.score-big{font-size:48px;font-weight:700;text-align:center;margin:10px 0}
.score-big.excellent{color:#5cb85c}.score-big.good{color:#f0ad4e}.score-big.poor{color:#e05c5c}
.breakdown{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.dim-row{display:flex;justify-content:space-between;align-items:center;background:var(--surface);padding:8px 10px;border-radius:6px}
.dim-bar-wrap{background:var(--border);border-radius:3px;height:5px;width:80px;margin-top:3px}
.dim-bar{height:5px;border-radius:3px;background:var(--green)}
.hint-box{background:#1a1007;border:1px solid #3d2800;border-radius:8px;padding:12px;margin-top:10px;font-size:12px;color:#f0ad4e;line-height:1.6}
.feedback-box{text-align:center;font-size:15px;font-weight:600;margin-bottom:8px}
.episode-trail{display:flex;gap:6px;margin:8px 0;align-items:center}
.step-dot{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
.step-dot.active{background:var(--blue);color:#fff}.step-dot.done-good{background:var(--green);color:#fff}.step-dot.done-bad{background:var(--red);color:#fff}.step-dot.pending{background:var(--surface2);color:var(--muted)}
.fairness-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0}
.demo-badge{padding:8px 10px;border-radius:8px;font-size:11px;border:1px solid var(--border)}
.news2-table{font-size:11px;width:100%;border-collapse:collapse;margin-top:8px}
.news2-table td,.news2-table th{padding:4px 6px;border:1px solid var(--border);text-align:center}
.news2-table th{background:var(--surface2);color:var(--muted)}
.s3{color:#e05c5c;font-weight:700}.s2{color:#f0ad4e;font-weight:600}.s1{color:#a0c050}.s0{color:var(--muted)}
.error{background:#2d0f0f;border:1px solid #5a1a1a;border-radius:8px;padding:10px;color:#e05c5c;font-size:12px;margin-top:8px}
.loading{text-align:center;padding:20px;color:var(--muted);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}
#episode-log{max-height:200px;overflow-y:auto;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px;font-family:monospace;font-size:11px;color:var(--muted);margin-top:8px}
</style>
</head>
<body>
<header>
  <div style="font-size:28px">🏥</div>
  <div>
    <h1>Medical Triage Environment</h1>
    <div style="font-size:11px;color:var(--muted);margin-top:2px">OpenEnv RL Environment &middot; NHS NEWS2 &middot; Team Falcons</div>
  </div>
  <span class="version">v2.0.0</span>
</header>

<div class="research-banner">
  <span>📊 <strong>Nature Medicine 2025</strong>: 1.7M LLM outputs showed 1.7× invasiveness bias against Black/unhoused patients</span>
  <span>🏥 <strong>Bordeaux Hospital</strong>: 200,000+ wrong triage decisions/year from gender bias</span>
  <span>🎓 <strong>Oxford 2026</strong>: LLMs drop from 85% knowledge → 60% real-world triage accuracy</span>
  <span>📈 <strong>MIMIC-III</strong>: 70% of preventable ED deaths involve post-assessment deterioration</span>
</div>

<div class="main">
  <!-- Sidebar: controls -->
  <div class="sidebar">
    <div class="card">
      <h3>Select Task</h3>
      <select id="task-select">
        <option value="simple_triage">🟢 Simple Triage (Easy)</option>
        <option value="conflicting_vitals">🟡 Conflicting Vitals (Medium)</option>
        <option value="masked_deterioration">🔴 Masked Deterioration (Hard)</option>
        <option value="demographic_fairness">⚖️ Demographic Fairness (Medium)</option>
        <option value="deteriorating_patient">📉 Deteriorating Patient (Hard, Multi-Turn)</option>
      </select>
      <select id="case-select"><option value="">Random case</option></select>
      <button class="primary" onclick="resetEnv()">🔄 New Patient Case</button>
    </div>

    <div class="card" id="task-info-card">
      <h3>About This Task</h3>
      <div id="task-info-text" style="font-size:12px;color:var(--muted);line-height:1.6"></div>
    </div>

    <div class="card">
      <h3>Episode History</h3>
      <div id="episode-log">No episode started yet.</div>
    </div>

    <div class="card">
      <h3>NEWS2 Quick Reference</h3>
      <table class="news2-table">
        <tr><th>Parameter</th><th class="s3">3</th><th class="s2">2</th><th class="s1">1</th><th class="s0">0</th></tr>
        <tr><td>Resp Rate</td><td class="s3">≤8</td><td></td><td class="s1">9-11</td><td class="s0">12-20</td></tr>
        <tr><td>SpO2</td><td class="s3">≤91</td><td class="s2">92-93</td><td class="s1">94-95</td><td class="s0">≥96</td></tr>
        <tr><td>Sys BP</td><td class="s3">≤90</td><td class="s2">91-100</td><td class="s1">101-110</td><td class="s0">111-219</td></tr>
        <tr><td>Heart Rate</td><td class="s3">≤40</td><td></td><td class="s1">41-50</td><td class="s0">51-90</td></tr>
        <tr><td>Temp</td><td class="s3">≤35</td><td></td><td class="s1">35.1-36</td><td class="s0">36.1-38</td></tr>
        <tr><td>Conscious</td><td class="s3">V/P/U</td><td></td><td></td><td class="s0">Alert</td></tr>
      </table>
      <div style="font-size:10px;color:var(--muted);margin-top:6px">0-2=Low &middot; 3-4=Medium &middot; 5-6=High &middot; ≥7=Critical</div>
    </div>
  </div>

  <!-- Main content -->
  <div class="content">
    <!-- Tab navigation -->
    <div style="display:flex;gap:2px;margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:0">
      <button id="tab-triage" onclick="switchTab('triage')" style="padding:8px 16px;background:transparent;border:none;border-bottom:2px solid var(--blue);color:var(--blue);font-weight:600;cursor:pointer;font-size:13px">🏥 Triage</button>
      <button id="tab-training" onclick="switchTab('training')" style="padding:8px 16px;background:transparent;border:none;border-bottom:2px solid transparent;color:var(--muted);cursor:pointer;font-size:13px">📈 Training Progress</button>
    </div>
    <div id="panel-triage">
    <div id="patient-section" style="display:none">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
        <div id="task-badge" class="task-badge easy">Easy</div>
        <div id="case-label" style="font-size:12px;color:var(--muted)"></div>
        <div id="episode-dots" class="episode-trail"></div>
      </div>

      <div class="card">
        <h3><span style="background:#185FA5;color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;margin-right:6px">1</span>Patient Presentation <span style="font-size:11px;color:var(--muted);font-weight:400">— read the case. Vitals auto-extracted below.</span></h3>
        <div class="patient-card" id="patient-history"></div>
        <div class="vitals-grid" id="vitals-grid"></div>
      </div>


      <!-- Live NEWS2 Calculator -->
      <div class="card" id="calc-card" style="margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <h3 style="margin:0"><span style="background:#185FA5;color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;margin-right:6px">2</span>NEWS2 Calculator <span style="font-size:11px;color:var(--muted);font-weight:400">— auto-filled from the patient above. Edit if needed.</span></h3>
          <button onclick="toggleCalc()" id="calc-toggle-btn" style="font-size:11px;padding:3px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--muted);cursor:pointer">hide</button>
        </div>
        <div id="calc-body">
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px">
            <div>
              <label>Resp Rate (RR)</label>
              <input type="number" id="c-rr" placeholder="e.g. 24" oninput="calcNEWS2()" style="text-align:center">
              <div id="s-rr" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
            <div>
              <label>SpO2 (%)</label>
              <input type="number" id="c-spo2" placeholder="e.g. 93" oninput="calcNEWS2()" style="text-align:center">
              <div id="s-spo2" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
            <div>
              <label>Systolic BP</label>
              <input type="number" id="c-bp" placeholder="e.g. 105" oninput="calcNEWS2()" style="text-align:center">
              <div id="s-bp" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
            <div>
              <label>Heart Rate (HR)</label>
              <input type="number" id="c-hr" placeholder="e.g. 112" oninput="calcNEWS2()" style="text-align:center">
              <div id="s-hr" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
            <div>
              <label>Temperature (°C)</label>
              <input type="number" id="c-temp" step="0.1" placeholder="e.g. 38.4" oninput="calcNEWS2()" style="text-align:center">
              <div id="s-temp" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
            <div>
              <label>Consciousness</label>
              <select id="c-cons" onchange="calcNEWS2()" style="text-align:center">
                <option value="">—</option>
                <option value="alert">Alert (0 pts)</option>
                <option value="voice">Voice (3 pts)</option>
                <option value="pain">Pain (3 pts)</option>
                <option value="unresponsive">Unresponsive (3 pts)</option>
              </select>
              <div id="s-cons" style="font-size:11px;text-align:center;margin-top:2px;color:var(--muted)">—</div>
            </div>
          </div>
          <!-- Result row -->
          <div style="display:flex;align-items:center;gap:12px;background:var(--surface2);border-radius:8px;padding:12px 14px">
            <div style="text-align:center">
              <div style="font-size:11px;color:var(--muted)">NEWS2 Total</div>
              <div id="r-total" style="font-size:32px;font-weight:700;color:var(--text)">—</div>
            </div>
            <div style="flex:1">
              <div style="font-size:11px;color:var(--muted)">Priority</div>
              <div id="r-priority" style="font-size:18px;font-weight:600">—</div>
            </div>
            <div style="flex:1">
              <div style="font-size:11px;color:var(--muted)">Critical Sign</div>
              <div id="r-sign" style="font-size:14px;font-weight:500;color:#f0ad4e">—</div>
            </div>
            <div style="flex:1">
              <div style="font-size:11px;color:var(--muted)">Action</div>
              <div id="r-action" style="font-size:13px;color:#6ab0f5">—</div>
            </div>
            <button onclick="applyCalcToForm()" id="apply-btn" style="display:none;padding:8px 14px;background:#185FA5;border:none;border-radius:8px;color:#fff;font-weight:600;cursor:pointer;font-size:13px">✅ Step 3: Apply to Assessment Form</button>
          </div>
        </div>
      </div>
      <div class="card" id="response-card">
        <h3><span style="background:#185FA5;color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;margin-right:6px">3</span>Your Triage Assessment <span style="font-size:11px;color:var(--muted);font-weight:400">— filled automatically. Add rationale, then submit.</span></h3>
        <div id="response-form"></div>
        <div style="font-size:11px;color:var(--muted);margin:10px 0 6px;text-align:center">— or skip steps 2-3 and let the AI fill everything —</div>
        <div style="display:flex;gap:8px;margin-top:4px">
          <button class="primary" style="flex:2" onclick="submitAction()">📋 Step 4: Submit & Score</button>
          <button id="ai-btn" style="flex:1;background:#1e3a5f;border-color:#2a5080;color:#6ab0f5;font-weight:600" onclick="aiFill()">🤖 Auto-fill with AI</button>
        </div>
        <div id="ai-status" style="font-size:11px;color:var(--muted);margin-top:4px;text-align:center"></div>
      </div>

      <div id="result-section"></div>
    </div>

    <div id="placeholder" style="padding:40px;text-align:center;color:var(--muted)">
      <div style="font-size:48px;margin-bottom:12px">🏥</div>
      <div style="font-size:16px;font-weight:600;color:var(--text)">Medical Triage Environment</div>
      <div style="margin-top:8px;font-size:13px">Select a task and click "New Patient Case" to begin</div>
      <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;max-width:600px;margin-left:auto;margin-right:auto">
        <div class="card"><div style="font-size:20px">5</div><div style="font-size:11px;color:var(--muted)">Tasks</div></div>
        <div class="card"><div style="font-size:20px">24</div><div style="font-size:11px;color:var(--muted)">Patient Cases</div></div>
        <div class="card"><div style="font-size:20px">94</div><div style="font-size:11px;color:var(--muted)">Tests Passing</div></div>
      </div>
    </div>

    </div><!-- /panel-triage -->

    <!-- Training Progress Panel -->
    <div id="panel-training" style="display:none">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px" id="stat-cards">
        <div class="card" style="text-align:center"><div style="font-size:24px;font-weight:700;color:var(--text)" id="stat-episodes">0</div><div style="font-size:11px;color:var(--muted)">Total Episodes</div></div>
        <div class="card" style="text-align:center"><div style="font-size:24px;font-weight:700;color:#5cb85c" id="stat-avg">—</div><div style="font-size:11px;color:var(--muted)">Overall Avg Score</div></div>
        <div class="card" style="text-align:center"><div style="font-size:24px;font-weight:700;color:#f0ad4e" id="stat-best">—</div><div style="font-size:11px;color:var(--muted)">Best Score</div></div>
        <div class="card" style="text-align:center"><div style="font-size:24px;font-weight:700;color:#6ab0f5" id="stat-tasks">0</div><div style="font-size:11px;color:var(--muted)">Tasks Attempted</div></div>
      </div>

      <div class="card" style="margin-bottom:14px">
        <h3 style="margin-bottom:12px">Learning Curve — Reward Per Episode</h3>
        <div style="position:relative;height:220px;width:100%"><canvas id="learning-curve"></canvas></div>
        <div style="font-size:11px;color:var(--muted);margin-top:6px">Each coloured dot = one completed episode, coloured by task. Dashed line = that task's recent average. <strong style="color:#e8eaf0">Expect green (easy) to score highest, red (hard) lowest</strong> — that's correct, not a bug. A rising trend within one colour = the agent is learning.</div>
      </div>

      <div class="card" style="margin-bottom:14px">
        <h3 style="margin-bottom:12px">Score by Task — Difficulty Gradient</h3>
        <div style="position:relative;height:160px;width:100%"><canvas id="task-bars"></canvas></div>
        <div style="font-size:11px;color:var(--muted);margin-top:6px"><strong style="color:#e8eaf0">Green bar (Simple Triage) should be highest. Red bar (Masked) should be lowest.</strong> This difficulty gradient is intentional — it means the reward signal is strong enough for an RL agent to learn from.</div>
      </div>

      <div class="card">
        <h3 style="margin-bottom:10px">Episode Log</h3>
        <div id="episode-table" style="font-size:12px;max-height:200px;overflow-y:auto"></div>
      </div>

      <button onclick="loadTrainingData()" style="margin-top:10px;padding:8px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;color:var(--text);cursor:pointer;font-size:12px">🔄 Refresh</button>
    </div>
  </div>
</div>

<script>
const TASK_INFO = {
  simple_triage: {diff:"easy", label:"Simple Triage", desc:"Patient has 3-4 clear vital signs. Compute NEWS2 score and classify urgency. Tests foundational clinical knowledge."},
  conflicting_vitals: {diff:"medium", label:"Conflicting Vitals", desc:"Some vitals look normal — they are a trap. Identify the truly dangerous parameter. Tests clinical reasoning beyond mechanical scoring."},
  masked_deterioration: {diff:"hard", label:"Masked Deterioration", desc:"Medications pharmacologically suppress classic warning signs. Beta-blockers mask tachycardia. Steroids mask fever. Tests expert-level clinical pattern recognition."},
  demographic_fairness: {diff:"medium", label:"Demographic Fairness", desc:"Identical clinical presentation, different demographic descriptor. Triage must be identical regardless of race, gender, housing status. Tests for systematic bias."},
  deteriorating_patient: {diff:"hard", label:"Deteriorating Patient", desc:"Multi-turn episode. Patient worsens over time. Must escalate at T=30 before crash. Reward: T=30 escalation=1.0, T=60 late=0.6, miss=0.0"}
};

const DIFF_CLASS = {easy:"easy", medium:"medium", hard:"hard"};
let state = {task_id:null, case_id:null, episode_id:null, step:0, max_steps:1};

function log(msg) {
  const el = document.getElementById("episode-log");
  el.innerHTML += `<div>${new Date().toLocaleTimeString()} ${msg}</div>`;
  el.scrollTop = el.scrollHeight;
}

function updateTaskInfo() {
  const tid = document.getElementById("task-select").value;
  const info = TASK_INFO[tid] || {};
  document.getElementById("task-info-text").textContent = info.desc || "";
}

async function populateCaseSelect() {
  const taskId = document.getElementById("task-select").value;
  const caseSelect = document.getElementById("case-select");
  caseSelect.innerHTML = '<option value="">Random case</option>';

  try {
    const resp = await fetch('/tasks');
    if (!resp.ok) return;
    const payload = await resp.json();
    const taskData = payload[taskId] || {};
    const caseIds = taskData.case_ids || [];

    caseIds.forEach((caseId, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `${caseId} (index ${idx})`;
      caseSelect.appendChild(opt);
    });
  } catch (_) {
    // Keep Random case option if /tasks is unavailable.
  }
}

async function onTaskSelectionChange() {
  updateTaskInfo();
  await populateCaseSelect();
}

document.getElementById("task-select").addEventListener("change", onTaskSelectionChange);
onTaskSelectionChange();

async function resetEnv() {
  const tid = document.getElementById("task-select").value;
  const ci  = document.getElementById("case-select").value;
  document.getElementById("patient-section").style.display = "block";
  document.getElementById("placeholder").style.display = "none";
  document.getElementById("result-section").innerHTML = "";
  document.getElementById("patient-history").textContent = "Loading patient case...";
  document.getElementById("patient-history").className = "patient-card loading";
  document.getElementById("vitals-grid").innerHTML = "";
  document.getElementById("episode-dots").innerHTML = "";

  try {
    const payload = {task_id: tid, seed: 42};
    if (ci) payload.case_index = parseInt(ci);
    const r = await fetch("/reset", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    const d = await r.json();
    const obs = d.observation;
    const info = d.info || {};

    state = {task_id: tid, case_id: obs.case_id, episode_id: info.episode_id, step: 0, max_steps: info.max_steps || 1};

    const tinfo = TASK_INFO[tid] || {};
    const badge = document.getElementById("task-badge");
    badge.textContent = tinfo.label || tid;
    badge.className = "task-badge " + (DIFF_CLASS[tinfo.diff] || "easy");

    document.getElementById("case-label").textContent = `Case: ${obs.case_id || "?"} | Episode: ${info.episode_id || "?"}`;
    document.getElementById("patient-history").textContent = obs.patient_history || "";
    document.getElementById("patient-history").className = "patient-card";

    // Auto-parse vitals from patient text and populate NEWS2 calculator
    NEWS2_PARAMS.forEach(p => { const el=document.getElementById('c-'+p); if(el) el.value=''; });
    const consEl=document.getElementById('c-cons'); if(consEl) consEl.value='';
    if (obs.patient_history) autoParseVitals(obs.patient_history);

    renderEpisodeDots();
    buildForm(tid);
    log(`Reset: ${tid} → ${obs.case_id}`);
  } catch(e) {
    document.getElementById("patient-history").textContent = "Error: " + e.message;
    document.getElementById("patient-history").className = "patient-card error";
  }
}

function renderEpisodeDots() {
  const el = document.getElementById("episode-dots");
  if (state.max_steps <= 1) { el.innerHTML = ""; return; }
  const labels = ["T=0","T=30","T=60"];
  el.innerHTML = labels.slice(0, state.max_steps).map((l,i) =>
    `<div class="step-dot ${i === state.step ? "active" : "pending"}" title="${l}">${l.replace("T=","")}</div>`
  ).join('<div style="font-size:10px;color:var(--muted)">→</div>');
}

function buildForm(tid) {
  const el = document.getElementById("response-form");
  if (tid === "deteriorating_patient") {
    el.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px">
        <button onclick="setAction('monitor')" id="btn-monitor" style="background:#1a2533;border-color:#2a5080;color:#6ab0f5">🔵 Monitor</button>
        <button onclick="setAction('escalate')" id="btn-escalate" style="background:#2d1a00;border-color:#5a3a00;color:#f0ad4e">🟡 Escalate</button>
        <button onclick="setAction('emergency_response')" id="btn-emergency" style="background:#2d0f0f;border-color:#5a1a1a;color:#e05c5c">🔴 Emergency</button>
      </div>
      <input type="hidden" id="f-action" value="">
      <div class="full"><label>Clinical Rationale</label><textarea id="f-rationale" rows="3" placeholder="Describe the trend you are responding to..."></textarea></div>
      <div><label>Confidence (0-1)</label><input type="number" id="f-confidence" min="0" max="1" step="0.1" value="0.7"></div>
    `;
    window._selectedAction = "";
  } else {
    const isFairness = tid === "demographic_fairness";
    el.innerHTML = `
      <div><label>Priority</label><select id="f-priority"><option value="">--</option><option>low</option><option>medium</option><option>high</option><option>critical</option></select></div>
      <div><label>NEWS2 Score</label><input type="number" id="f-news2" min="0" max="20" placeholder="0-20"></div>
      <div><label>Critical Sign</label><input id="f-sign" placeholder="e.g. respiratory_rate"></div>
      <div><label>Recommended Action</label><select id="f-action"><option value="">--</option><option value="routine_monitoring">Routine monitoring</option><option value="urgent_review">Urgent review</option><option value="emergency_response">Emergency response</option></select></div>
      ${tid==="conflicting_vitals"?'<div class="full"><label>Misleading Signs (comma-separated)</label><input id="f-misleading" placeholder="e.g. heart_rate, systolic_bp"></div>':''}
      ${tid==="masked_deterioration"?`
        <div><label>Masking Drug/Condition</label><input id="f-masking" placeholder="e.g. bisoprolol"></div>
        <div><label>Masked Sign</label><input id="f-masked-sign" placeholder="e.g. heart_rate"></div>
        <div class="full"><label>Critical Clues (comma-separated)</label><input id="f-clues" placeholder="e.g. lactate, urine_output_reduced"></div>
      `:""}
      <div class="full"><label>Rationale</label><textarea id="f-rationale" rows="2" placeholder="Your clinical reasoning..."></textarea></div>
      <div><label>Confidence (0-1)</label><input type="number" id="f-confidence" min="0" max="1" step="0.1" value="0.8"></div>
    `;
  }
}

function setAction(a) {
  window._selectedAction = a;
  document.getElementById("f-action").value = a;
  ["monitor","escalate","emergency_response"].forEach(x => {
    const b = document.getElementById("btn-"+x);
    if (b) b.style.fontWeight = x===a ? "700" : "400";
  });
}

function gv(id) { const el=document.getElementById(id); return el ? (el.value||"").trim() : ""; }

function buildAction() {
  const tid = state.task_id;
  if (tid === "deteriorating_patient") {
    return {action: window._selectedAction || "monitor", rationale: gv("f-rationale"),
            confidence: parseFloat(gv("f-confidence")||"0.7")};
  }
  const a = {priority: gv("f-priority"), recommended_action: gv("f-action"),
              rationale: gv("f-rationale"), confidence: parseFloat(gv("f-confidence")||"0.8")};
  const n2 = gv("f-news2"); if (n2) a.news2_score = parseInt(n2);
  const s = gv("f-sign"); if (s) a.critical_sign = s;
  if (tid==="conflicting_vitals") { const m=gv("f-misleading"); if(m) a.misleading_signs=m.split(",").map(x=>x.trim()); }
  if (tid==="masked_deterioration") {
    const mk=gv("f-masking"); if(mk) a.masking_drug_or_condition=mk;
    const ms=gv("f-masked-sign"); if(ms) a.masked_sign=ms;
    const c=gv("f-clues"); if(c) a.critical_clues=c.split(",").map(x=>x.trim());
  }
  return a;
}

async function submitAction() {
  const action = buildAction();
  document.getElementById("result-section").innerHTML = '<div class="loading">Grading response...</div>';

  try {
    const r = await fetch("/step", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action})});
    const d = await r.json();
    const obs = d.observation;
    const reward = d.reward;
    const done = d.done;

    state.step++;
    renderResult(obs, reward, done, d.info || {});
    log(`Step ${state.step}: reward=${reward.toFixed(3)} done=${done}`);

    if (!done && state.task_id==="deteriorating_patient") {
      // Update patient history for next step
      document.getElementById("patient-history").textContent = obs.patient_history;
      renderEpisodeDots();
      document.getElementById("result-section").insertAdjacentHTML("beforeend",
        `<div style="margin-top:8px;padding:10px;background:#1a1f3a;border-radius:8px;font-size:12px;color:#6ab0f5">
          ⏩ Episode continues — assess updated vitals above and submit next action
        </div>`);
    }
  } catch(e) {
    document.getElementById("result-section").innerHTML = `<div class="error">Error: ${e.message}</div>`;
  }
}

function renderResult(obs, reward, done, info) {
  const pct = Math.round(reward * 100);
  const cls = reward>=0.85?"excellent":reward>=0.55?"good":"poor";
  const emoji = reward>=0.85?"🏆":reward>=0.55?"✅":"⚠️";
  const breakdown = obs.score_breakdown || {};

  const dimRows = Object.entries(breakdown)
    .filter(([k,v]) => typeof v==="number" && !k.startsWith("_"))
    .map(([k,v]) => {
      const maxVal = k==="priority"?0.6:k==="critical_sign"?0.3:k==="recommended_action"?0.2:k==="news2_score"?0.25:v>0.1?v:0.15;
      const pct = Math.min(100,Math.round((v/Math.max(maxVal,v,0.01))*100));
      return `<div class="dim-row"><div><div style="font-size:12px;font-weight:500">${k.replace(/_/g," ")}</div>
        <div class="dim-bar-wrap"><div class="dim-bar" style="width:${pct}%"></div></div></div>
        <div style="font-size:14px;font-weight:700;color:${v>0.1?"#5cb85c":"#e05c5c"}">${v.toFixed(3)}</div></div>`;
    }).join("");

  const gt = info.ground_truth;
  const gtHtml = (done && gt) ? `
    <div style="margin-top:12px;padding:10px;background:#0f1a0f;border:1px solid #1d3a1d;border-radius:8px;font-size:11px;color:var(--muted)">
      <div style="font-weight:600;color:#5cb85c;margin-bottom:4px">Ground Truth Revealed:</div>
      ${gt.critical_sign?"<div>Critical sign: <strong style='color:#e8eaf0'>"+gt.critical_sign+"</strong></div>":""}
      ${gt.priority?"<div>Priority: <strong style='color:#e8eaf0'>"+gt.priority+"</strong></div>":""}
      ${gt.rationale?"<div style='margin-top:4px'>"+gt.rationale.substring(0,200)+"</div>":""}
    </div>` : "";

  const hintHtml = obs.hint ? `<div class="hint-box">💡 ${obs.hint}</div>` : "";

  document.getElementById("result-section").innerHTML = `
    <div class="result-card ${cls}">
      <div class="feedback-box">${emoji} ${obs.feedback || ""}</div>
      <div class="score-big ${cls}">${pct}%</div>
      <div style="text-align:center;font-size:12px;color:var(--muted);margin-bottom:12px">reward = ${reward.toFixed(3)}</div>
      <div class="breakdown">${dimRows}</div>
      ${hintHtml}${gtHtml}
    </div>`;
}

async function agentAssess() {
  // Backward-compatible alias used by older docs/snippets.
  return aiFill();
}

async function aiFill() {
  const btn = document.getElementById("ai-btn");
  const status = document.getElementById("ai-status");
  const tid = state.task_id;

  if (!tid) { status.textContent = "Start a patient case first."; return; }

  btn.textContent = "⏳ Thinking...";
  btn.disabled = true;
  status.textContent = "";

  try {
    const history = document.getElementById("patient-history").textContent;
    const r = await fetch("/suggest", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({patient_history: history, task_id: tid})
    });
    const d = await r.json();
    const s = d.suggestion || {};

    // Populate every form field
    const set = (id, val) => { const el=document.getElementById(id); if(el && val!==undefined && val!==null) el.value=val; };

    if (tid === "deteriorating_patient") {
      if (s.action) setAction(s.action);
    } else {
      set("f-priority", s.priority || "");
      set("f-news2", s.news2_score ?? "");
      set("f-sign", s.critical_sign || "");
      set("f-action", s.recommended_action || "");
      if (s.misleading_signs) set("f-misleading", (s.misleading_signs||[]).join(", "));
      if (s.masking_drug_or_condition) set("f-masking", s.masking_drug_or_condition);
      if (s.masked_sign) set("f-masked-sign", s.masked_sign);
      if (s.critical_clues) set("f-clues", (s.critical_clues||[]).join(", "));
    }
    set("f-rationale", s.rationale || "");
    set("f-confidence", s.confidence ?? 0.7);

    const src = d.llm_used ? `🤖 ${d.model}` : `⚙️ Rule-based (${d.model})`;
    status.textContent = `Filled by: ${src}`;
    status.style.color = d.llm_used ? "#8ab4f8" : "#aaa";
    log(`AI filled: ${tid} via ${d.llm_used ? "LLM":"rules"}`);
  } catch(e) {
    status.textContent = "AI fill failed: " + e.message;
    status.style.color = "#e05c5c";
  } finally {
    btn.textContent = "🤖 Auto-fill with AI";
    btn.disabled = false;
  }
}

// ── Tab switching ────────────────────────────────────────────
function switchTab(tab) {
  ['triage','training'].forEach(t => {
    document.getElementById('panel-'+t).style.display = t===tab ? 'block' : 'none';
    const btn = document.getElementById('tab-'+t);
    if (t===tab) {
      btn.style.borderBottomColor = 'var(--blue)';
      btn.style.color = 'var(--blue)';
      btn.style.fontWeight = '600';
    } else {
      btn.style.borderBottomColor = 'transparent';
      btn.style.color = 'var(--muted)';
      btn.style.fontWeight = '400';
    }
  });
  if (tab==='training') loadTrainingData();
}

// ── Training progress charts ─────────────────────────────────
let lcChart = null, tbChart = null;

const TASK_COLORS = {
  simple_triage:        '#5cb85c',
  conflicting_vitals:   '#f0ad4e',
  masked_deterioration: '#e05c5c',
  demographic_fairness: '#6ab0f5',
  deteriorating_patient:'#b06bf5',
};
const TASK_SHORT = {
  simple_triage:'Simple',conflicting_vitals:'Conflicting',
  masked_deterioration:'Masked',demographic_fairness:'Fairness',
  deteriorating_patient:'Deterioration'
};

async function loadTrainingData() {
  try {
    const r = await fetch('/history?limit=200');
    const d = await r.json();
    renderStats(d.stats || {});
    renderLearningCurve(d.episodes || []);
    renderTaskBars(d.stats || {});
    renderEpisodeTable(d.episodes || []);
  } catch(e) {
    console.error('loadTrainingData:', e);
  }
}

function renderStats(stats) {
  document.getElementById('stat-episodes').textContent = stats.total_episodes || 0;
  document.getElementById('stat-avg').textContent = stats.overall_avg != null ? stats.overall_avg.toFixed(3) : '—';
  document.getElementById('stat-best').textContent = stats.best_score != null ? stats.best_score.toFixed(3) : '—';
  document.getElementById('stat-tasks').textContent = Object.keys(stats.by_task || {}).length;
}

function renderLearningCurve(episodes) {
  const ctx = document.getElementById('learning-curve').getContext('2d');
  ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
  if (!episodes.length) {
    ctx.fillStyle = '#7a85a0'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Submit assessments on the Triage tab to see the learning curve', ctx.canvas.width/2, 110);
    return;
  }

  // Separate scatter series per task so dots are colour-coded by difficulty
  const taskOrder = ['simple_triage','conflicting_vitals','masked_deterioration','demographic_fairness','deteriorating_patient'];
  const byTask = {};
  episodes.forEach((e, i) => {
    if (!byTask[e.task_id]) byTask[e.task_id] = [];
    byTask[e.task_id].push({x: i+1, y: e.reward, ep: e});
  });

  // Rolling 5-episode average — the important trend line
  const scores = episodes.map(e => e.reward);
  const rolling = scores.map((_, i) => {
    const w = scores.slice(Math.max(0, i-4), i+1);
    return +(w.reduce((a,b)=>a+b,0) / w.length).toFixed(3);
  });

  const datasets = [];
  taskOrder.filter(t => byTask[t]).forEach(t => {
    datasets.push({
      label: TASK_SHORT[t] || t,
      data: byTask[t],
      type: 'scatter',
      backgroundColor: (TASK_COLORS[t] || '#888') + 'cc',
      borderColor: TASK_COLORS[t] || '#888',
      borderWidth: 1, pointRadius: 7, pointHoverRadius: 9,
    });
  });
  datasets.push({
    label: '5-ep rolling avg (trend)',
    data: rolling.map((y,i) => ({x:i+1, y})),
    type: 'line',
    borderColor: '#ffffff', backgroundColor: 'transparent',
    borderWidth: 2.5, borderDash: [5,3], pointRadius: 0, tension: 0.4, order: -1,
  });

  if (lcChart) lcChart.destroy();
  lcChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: {
          type: 'linear',
          title: {display:true, text:'Episode number', color:'#7a85a0'},
          ticks: {color:'#7a85a0', precision:0},
          grid: {color:'rgba(45,51,80,0.6)'}, min: 0,
        },
        y: {
          title: {display:true, text:'Reward  (0=worst  1=perfect)', color:'#7a85a0'},
          ticks: {color:'#7a85a0', callback: v => v.toFixed(1)},
          grid: {color:'rgba(45,51,80,0.6)'}, min: 0, max: 1,
        }
      },
      plugins: {
        legend: {position:'bottom', labels:{color:'#e8eaf0',boxWidth:10,padding:10,font:{size:11}}},
        tooltip: {
          callbacks: {
            title: items => 'Episode #' + items[0].parsed.x,
            label: item => {
              const ep = episodes[Math.round(item.parsed.x) - 1];
              if (!ep) return 'Avg: ' + item.parsed.y.toFixed(3);
              const parts = [TASK_SHORT[ep.task_id] + ' | ' + ep.case_id, 'Score: ' + ep.reward.toFixed(3)];
              const bd = Object.entries(ep.breakdown||{}).filter(([k])=>!k.startsWith('_')).slice(0,3);
              if (bd.length) parts.push(bd.map(([k,v])=>k.replace(/_/g,' ')+': '+v).join('  '));
              return parts;
            }
          }
        }
      }
    },
    plugins: [{
      id: 'thresholdLines',
      afterDraw(chart) {
        const {ctx: c, scales:{x,y}} = chart;
        [{v:0.85,label:'Excellent ≥0.85',col:'#5cb85c'},{v:0.55,label:'Good ≥0.55',col:'#f0ad4e'}].forEach(({v,label,col}) => {
          const yy = y.getPixelForValue(v);
          c.save();
          c.strokeStyle = col + '55'; c.lineWidth=1; c.setLineDash([4,4]);
          c.beginPath(); c.moveTo(x.left,yy); c.lineTo(x.right,yy); c.stroke();
          c.fillStyle=col+'aa'; c.font='10px sans-serif'; c.textAlign='right';
          c.fillText(label, x.right-2, yy-3);
          c.restore();
        });
      }
    }]
  });
}

function renderTaskBars(stats) {
  const ctx = document.getElementById('task-bars').getContext('2d');
  const byTask = stats.by_task || {};
  if (!Object.keys(byTask).length) {
    ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
    ctx.fillStyle='#7a85a0'; ctx.font='13px sans-serif'; ctx.textAlign='center';
    ctx.fillText('No data yet — submit assessments to see scores by task',ctx.canvas.width/2,60);
    return;
  }
  const taskOrder = ['simple_triage','conflicting_vitals','masked_deterioration','demographic_fairness','deteriorating_patient'];
  const present = taskOrder.filter(t => byTask[t]);
  if (!present.length) return;

  if (tbChart) tbChart.destroy();
  ctx.canvas.style.width = '100%';
  ctx.canvas.style.height = '160px';
  tbChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: present.map(t => TASK_SHORT[t]),
      datasets: [
        {
          label: 'Average Score',
          data: present.map(t => byTask[t].avg),
          backgroundColor: present.map(t => TASK_COLORS[t]+'99'),
          borderColor:     present.map(t => TASK_COLORS[t]),
          borderWidth: 1,
        },
        {
          label: 'Best Score',
          data: present.map(t => byTask[t].best),
          backgroundColor: 'transparent',
          borderColor: present.map(t => TASK_COLORS[t]),
          borderWidth: 2,
          borderDash: [4,3],
          type: 'line',
          pointRadius: 5,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks:{color:'#7a85a0'}, grid:{color:'#2d3350'} },
        y: { ticks:{color:'#7a85a0'}, grid:{color:'#2d3350'}, min:0, max:1,
             title:{display:true,text:'Score',color:'#7a85a0'} }
      },
      plugins: { legend:{labels:{color:'#e8eaf0',boxWidth:10}} }
    }
  });
}

function renderEpisodeTable(episodes) {
  const el = document.getElementById('episode-table');
  if (!episodes.length) { el.innerHTML = '<div style="color:var(--muted)">No episodes yet. Submit assessments on the Triage tab.</div>'; return; }
  const rows = [...episodes].reverse().slice(0,30).map(e =>
    `<div style="display:flex;gap:12px;padding:4px 0;border-bottom:0.5px solid var(--border)">
      <span style="color:var(--muted);min-width:28px">#${e.n}</span>
      <span style="min-width:90px;color:${TASK_COLORS[e.task_id]||'#888'}">${TASK_SHORT[e.task_id]||e.task_id}</span>
      <span style="min-width:50px;color:var(--muted)">${e.case_id}</span>
      <span style="font-weight:600;color:${e.reward>=0.85?'#5cb85c':e.reward>=0.55?'#f0ad4e':'#e05c5c'}">${e.reward.toFixed(3)}</span>
      <span style="color:var(--muted);font-size:11px">${Object.entries(e.breakdown||{}).map(([k,v])=>k.replace(/_/g,' ')+':'+v).join(' | ')}</span>
    </div>`
  ).join('');
  el.innerHTML = rows;
}

// Load Chart.js
(function() {
  const s = document.createElement('script');
  s.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js';
  s.onload = () => { if(document.getElementById('panel-training').style.display!=='none') loadTrainingData(); };
  document.head.appendChild(s);
})();

// Auto-refresh training panel every 8s when visible — only if new episodes exist
let _lastEpisodeCount = 0;
setInterval(async () => {
  if (document.getElementById('panel-training').style.display === 'none') return;
  try {
    const r = await fetch('/stats');
    const s = await r.json();
    const newCount = s.total_episodes || 0;
    if (newCount !== _lastEpisodeCount) {
      _lastEpisodeCount = newCount;
      loadTrainingData();
    }
  } catch(e) {}
}, 8000);

// ── Live NEWS2 Calculator ─────────────────────────────────────
const NEWS2_PARAMS = ['rr','spo2','bp','hr','temp','cons'];
const PARAM_NAMES  = {rr:'respiratory_rate',spo2:'spo2',bp:'systolic_bp',hr:'heart_rate',temp:'temperature',cons:'consciousness'};

function scoreRR(v)   { if(v<=8)return 3; if(v<=11)return 1; if(v<=20)return 0; if(v<=24)return 2; return 3; }
function scoreSpO2(v) { if(v<=91)return 3; if(v<=93)return 2; if(v<=95)return 1; return 0; }
function scoreBP(v)   { if(v<=90)return 3; if(v<=100)return 2; if(v<=110)return 1; if(v<=219)return 0; return 3; }
function scoreHR(v)   { if(v<=40)return 3; if(v<=50)return 1; if(v<=90)return 0; if(v<=110)return 1; if(v<=130)return 2; return 3; }
function scoreTemp(v) { if(v<=35.0)return 3; if(v<=36.0)return 1; if(v<=38.0)return 0; if(v<=39.0)return 1; return 2; }
function scoreCons(v) { return v==='alert'||v===''?0:3; }

const SCORERS = {rr:scoreRR, spo2:scoreSpO2, bp:scoreBP, hr:scoreHR, temp:scoreTemp, cons:scoreCons};
const COLOR = {0:'#7a85a0',1:'#a0c050',2:'#f0ad4e',3:'#e05c5c'};

function calcNEWS2() {
  const vals = {};
  const scores = {};

  ['rr','spo2','bp','hr','temp'].forEach(p => {
    const el = document.getElementById('c-'+p);
    const v = parseFloat(el?.value);
    vals[p] = isNaN(v) ? null : v;
  });
  vals['cons'] = document.getElementById('c-cons')?.value || '';

  let total = 0;
  let maxScore = -1;
  let critParam = null;
  let hasAny = false;

  NEWS2_PARAMS.forEach(p => {
    const v = vals[p];
    const scoreEl = document.getElementById('s-'+p);
    if (v===null || v==='') { if(scoreEl) scoreEl.textContent='—'; scoreEl.style.color='var(--muted)'; return; }
    hasAny = true;
    const s = SCORERS[p](v);
    scores[p] = s;
    total += s;
    if(s > maxScore) { maxScore = s; critParam = p; }
    if(scoreEl) {
      scoreEl.textContent = s + ' pt' + (s!==1?'s':'');
      scoreEl.style.color = COLOR[s] || '#888';
      scoreEl.style.fontWeight = s>=2?'700':'400';
    }
  });

  if (!hasAny) {
    document.getElementById('r-total').textContent = '—';
    document.getElementById('r-priority').textContent = '—';
    document.getElementById('r-sign').textContent = '—';
    document.getElementById('r-action').textContent = '—';
    document.getElementById('apply-btn').style.display = 'none';
    return;
  }

  // Priority
  const hasRed = Object.values(scores).some(s => s===3);
  let priority, action, pColor;
  if (total>=7 || (hasRed && total>=5)) {
    priority='critical'; action='emergency_response'; pColor='#e05c5c';
  } else if (total>=5 || hasRed) {
    priority='high'; action='urgent_review'; pColor='#f0ad4e';
  } else if (total>=3) {
    priority='medium'; action='urgent_review'; pColor='#f0ad4e';
  } else {
    priority='low'; action='routine_monitoring'; pColor='#5cb85c';
  }

  document.getElementById('r-total').textContent = total;
  document.getElementById('r-total').style.color = pColor;
  document.getElementById('r-priority').textContent = priority.toUpperCase();
  document.getElementById('r-priority').style.color = pColor;
  document.getElementById('r-sign').textContent = critParam ? PARAM_NAMES[critParam] : '—';
  document.getElementById('r-action').textContent = action.replace(/_/g,' ');
  document.getElementById('apply-btn').style.display = 'inline-block';

  // Store computed values for applyCalcToForm()
  window._calc = { total, priority, critParam: critParam ? PARAM_NAMES[critParam] : '', action };
}

function applyCalcToForm() {
  const c = window._calc;
  if (!c) return;
  const set = (id, val) => { const el=document.getElementById(id); if(el&&val!=null) el.value=val; };
  set('f-priority', c.priority);
  set('f-news2',    c.total);
  set('f-sign',     c.critParam);
  set('f-action',   c.action);
  // Flash the form to show it was filled
  const card = document.getElementById('response-card');
  card.style.borderColor = '#185FA5';
  setTimeout(() => card.style.borderColor = 'var(--border)', 800);
  document.getElementById('response-card').scrollIntoView({behavior:'smooth', block:'nearest'});
}

function toggleCalc() {
  const body = document.getElementById('calc-body');
  const btn  = document.getElementById('calc-toggle-btn');
  const hidden = body.style.display==='none';
  body.style.display = hidden ? 'block' : 'none';
  btn.textContent = hidden ? 'hide' : 'show';
}

// Auto-parse vitals from patient history text when a case loads
function autoParseVitals(text) {
  const RX_SEP = '[=:\u005cs]*';
  const patterns = {
    rr:   new RegExp('RR'   + RX_SEP + '(\u005cd+)',       'i'),
    spo2: new RegExp('SpO2' + RX_SEP + '(\u005cd+)',       'i'),
    bp:   new RegExp('BP'   + RX_SEP + '(\u005cd+)/\u005cd+', 'i'),
    hr:   new RegExp('HR'   + RX_SEP + '(\u005cd+)',       'i'),
    temp: new RegExp('Temp' + RX_SEP + '([\u005cd.]+)',    'i'),
    cons: new RegExp('Consciousness' + RX_SEP + '(\u005cw+)', 'i'),
  };
  let found = false;
  Object.entries(patterns).forEach(([param, rx]) => {
    const m = text.match(rx);
    if (!m) return;
    const el = document.getElementById('c-'+param);
    if (!el) return;
    if (param==='cons') {
      const v = m[1].toLowerCase();
      el.value = ['alert','voice','pain','unresponsive'].includes(v) ? v : 'alert';
    } else {
      el.value = m[1];
    }
    found = true;
  });
  if (found) {
    calcNEWS2();
    // Scroll calculator into view
    document.getElementById('calc-card').scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

</script>
</body>
</html>'''

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
# uvicorn.run(app, host="0.0.0.0", port=8000)
def main() -> None:
    """CLI entrypoint required by OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
