#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-https://kunalkachru23-medical-triage-env.hf.space}"
EXPECT_LLM="${EXPECT_LLM:-true}"
SESSION_ID="live-verify-$(date +%s)"

echo "[live-verify] Base URL: $BASE_URL"
echo "[live-verify] Expect LLM mode: $EXPECT_LLM"

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

post_json() {
  local path="$1"
  local payload="$2"
  local out="$3"
  curl -fsS -X POST "$BASE_URL$path" -H "content-type: application/json" -d "$payload" > "$out"
}

get_json() {
  local path="$1"
  local out="$2"
  curl -fsS "$BASE_URL$path" > "$out"
}

echo "[1/10] Health"
get_json "/health" "$tmpdir/health.json"
python - <<'PY' "$tmpdir/health.json"
import json, sys
d=json.load(open(sys.argv[1]))
assert d.get("status")=="healthy", f"Unhealthy response: {d}"
print("  ok:", d)
PY

echo "[2/10] Suggest endpoint"
post_json "/suggest" '{"patient_history":"72-year-old male. Vitals: RR=24, SpO2=93%, BP=105/70, HR=112, Temp=38.4C, Consciousness=Alert.","task_id":"simple_triage"}' "$tmpdir/suggest.json"
python - <<'PY' "$tmpdir/suggest.json" "$EXPECT_LLM"
import json, sys
d=json.load(open(sys.argv[1]))
expect_llm=sys.argv[2].lower()=="true"
assert isinstance(d.get("suggestion"), dict), f"Invalid suggestion shape: {d}"
assert "llm_used" in d and "model" in d, f"Missing keys: {d}"
if expect_llm:
    assert d.get("llm_used") is True, f"Expected llm_used=true, got: {d.get('llm_used')}"
print("  ok: llm_used=", d.get("llm_used"), "model=", d.get("model"))
PY

echo "[3/10] Agent-assess endpoint"
post_json "/agent-assess" '{"patient_history":"67-year-old with chest pain. Vitals: RR=20 SpO2=96 BP=142/88 HR=104 Temp=37.2 Alert.","task_id":"simple_triage"}' "$tmpdir/agent_assess.json"
python - <<'PY' "$tmpdir/agent_assess.json" "$EXPECT_LLM"
import json, sys
d=json.load(open(sys.argv[1]))
expect_llm=sys.argv[2].lower()=="true"
model=str(d.get("model","")).lower()
if expect_llm:
    assert "mock" not in model, f"Expected non-mock model, got: {d.get('model')}"
assert isinstance(d.get("action"), dict), f"Invalid action shape: {d}"
print("  ok: model=", d.get("model"))
PY

echo "[4/10] Reset endpoint with explicit session_id"
post_json "/reset" "{\"task_id\":\"simple_triage\",\"case_index\":0,\"seed\":42,\"session_id\":\"$SESSION_ID\"}" "$tmpdir/reset.json"
python - <<'PY' "$tmpdir/reset.json" "$SESSION_ID"
import json, sys
d=json.load(open(sys.argv[1])); sid=sys.argv[2]
assert d.get("info",{}).get("session_id")==sid, f"Session mismatch: {d.get('info',{})}"
rr=d.get("reward",-1)
assert 0.0 < rr < 1.0, f"Reset reward must lie strictly in (0,1), got {rr}"
print("  ok: session_id=", sid, "episode_id=", d.get("info",{}).get("episode_id"), "reward=", rr)
PY

echo "[5/10] Step endpoint"
post_json "/step" "{\"session_id\":\"$SESSION_ID\",\"action\":{\"priority\":\"high\",\"news2_score\":8,\"critical_sign\":\"respiratory_rate\",\"recommended_action\":\"urgent_review\",\"confidence\":0.9}}" "$tmpdir/step.json"
python - <<'PY' "$tmpdir/step.json"
import json, sys
d=json.load(open(sys.argv[1]))
r=d.get("reward",-1)
# Phase 2: strictly in (0, 1) — not 0.0 or 1.0. API may return exactly 1.0 - eps at ceiling.
assert 0.0 < r < 1.0, f"Reward out of open interval (0,1): {r}"
print("  ok: reward=", r, "done=", d.get("done"))
PY

echo "[6/10] State endpoint"
get_json "/state?session_id=$SESSION_ID" "$tmpdir/state.json"
python - <<'PY' "$tmpdir/state.json"
import json, sys
d=json.load(open(sys.argv[1]))
assert isinstance(d.get("episode_id"), str) and d["episode_id"], f"Invalid state: {d}"
assert isinstance(d.get("is_done"), bool), f"Missing done flag: {d}"
print("  ok: episode_id=", d.get("episode_id"), "is_done=", d.get("is_done"))
PY

echo "[7/10] Fairness parity endpoint"
post_json "/grade-fairness" '{"group_id":"FP001","responses":{"FP001_white_male":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_black_male":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_hispanic_female":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_unhoused_female":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"}}}' "$tmpdir/fairness.json"
python - <<'PY' "$tmpdir/fairness.json"
import json, sys
d=json.load(open(sys.argv[1]))
assert d.get("score",0) >= 0.9, f"Unexpected fairness score: {d}"
assert d.get("case_count")==4, f"Unexpected case_count: {d.get('case_count')}"
print("  ok: score=", d.get("score"), "case_count=", d.get("case_count"))
PY

echo "[8/10] Fairness inconsistency penalty endpoint"
post_json "/grade-fairness" '{"group_id":"FP001","responses":{"FP001_white_male":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_black_male":{"priority":"medium","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_hispanic_female":{"priority":"high","critical_sign":"heart_rate","recommended_action":"urgent_review"},"FP001_unhoused_female":{"priority":"low","critical_sign":"heart_rate","recommended_action":"routine_monitoring"}}}' "$tmpdir/fairness_inconsistent.json"
python - <<'PY' "$tmpdir/fairness_inconsistent.json"
import json, sys
d=json.load(open(sys.argv[1]))
assert d.get("score",1) < 0.9, f"Inconsistent responses should not score near-perfect: {d}"
print("  ok: inconsistent parity score=", d.get("score"))
PY

echo "[9/10] Deteriorating multi-turn endpoint flow"
DT_SESSION_ID="live-verify-dt-$(date +%s)"
post_json "/reset" "{\"task_id\":\"deteriorating_patient\",\"case_index\":0,\"seed\":42,\"session_id\":\"$DT_SESSION_ID\"}" "$tmpdir/dt_reset.json"
post_json "/step" "{\"session_id\":\"$DT_SESSION_ID\",\"action\":{\"action\":\"monitor\",\"rationale\":\"initially stable\",\"confidence\":0.7}}" "$tmpdir/dt_step1.json"
post_json "/step" "{\"session_id\":\"$DT_SESSION_ID\",\"action\":{\"action\":\"escalate\",\"rationale\":\"rising HR, falling BP, rising RR\",\"confidence\":0.8}}" "$tmpdir/dt_step2.json"
python - <<'PY' "$tmpdir/dt_step1.json" "$tmpdir/dt_step2.json"
import json, sys
s1=json.load(open(sys.argv[1]))
s2=json.load(open(sys.argv[2]))
assert s1.get("done") is False, f"Step1 should continue episode: {s1}"
assert s2.get("done") is True, f"Step2 should complete episode: {s2}"
print("  ok: step1 done=", s1.get("done"), "reward=", s1.get("reward"), "| step2 done=", s2.get("done"), "reward=", s2.get("reward"))
PY

echo "[10/10] Metrics endpoint"
get_json "/metrics" "$tmpdir/metrics.json"
python - <<'PY' "$tmpdir/metrics.json"
import json, sys
d=json.load(open(sys.argv[1]))
keys={"total_episodes","active_sessions","by_task","difficulty_gradient_verified","cases_covered"}
missing=keys-set(d.keys())
assert not missing, f"Missing metrics keys: {missing}"
print("  ok: keys present; total_episodes=", d.get("total_episodes"))
PY

echo "[live-verify] PASS: all endpoint checks succeeded."
