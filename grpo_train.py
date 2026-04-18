"""
grpo_train.py — GRPO Fine-tuning on the Medical Triage Environment
===================================================================
Fine-tunes Qwen2.5-1.5B-Instruct (or any causal LM) using Group Relative
Policy Optimization (GRPO) via TRL. The live HF Space is used as the
reward oracle — no local server needed.

Covers all 11 clinical tasks (v2.3.0):
  simple_triage, conflicting_vitals, masked_deterioration, demographic_fairness,
  deteriorating_patient, sepsis_bundle, paediatric_triage, medication_reconciliation,
  icu_deterioration, sbar_handover, differential_diagnosis

Designed to run on:
  - Google Colab free tier (T4 GPU, 15GB VRAM) — default
  - Apple M4 Pro (MPS backend) — pass --device mps --no-quantize
  - Any CUDA GPU with >=12GB VRAM

Usage (Colab):
  !pip install trl>=0.12.0 transformers accelerate bitsandbytes peft datasets requests -q
  !python grpo_train.py --push-to-hub kunalkachru23/grpo-medical-triage-qwen1.5b

Usage (M4 Pro — conda grpo-train env with Python 3.12):
  python grpo_train.py --device mps --no-quantize

Resume after an interrupted run:
  python grpo_train.py --output-dir grpo-medical-triage --resume-latest
  python grpo_train.py --output-dir grpo-medical-triage --resume-from-checkpoint grpo-medical-triage/checkpoint-50

Env vars (optional overrides):
  SERVER_URL  — default: https://kunalkachru23-medical-triage-env.hf.space
  HF_TOKEN    — for pushing adapter to Hub (optional)
  .env        — HF_TOKEN is auto-loaded from current working directory if present
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from typing import List, Optional, Union

import requests
from requests.adapters import HTTPAdapter, Retry
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = [
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

SYSTEM_PROMPT = """\
You are a clinical triage AI assistant.
The user message begins with [TASK: <name>]. Respond with ONE single flat JSON object for that task only.
No nested objects. No markdown. No code fences. No extra text.
Always include every required key for that task; if uncertain, emit best-guess placeholder values instead of omitting keys.

If [TASK: simple_triage] or [TASK: conflicting_vitals] or [TASK: masked_deterioration] or [TASK: demographic_fairness]:
{"priority":"low|medium|high|critical","news2_score":<integer 0-20>,\
"critical_sign":"<most abnormal vital sign>","recommended_action":"emergency_response|urgent_review|routine_monitoring",\
"rationale":"<str>","confidence":<0.0-1.0>}

If [TASK: deteriorating_patient]:
{"action":"monitor|escalate|emergency_response","rationale":"<str>","confidence":<0.0-1.0>}

If [TASK: sepsis_bundle]:
{"priority":"low|medium|high|critical",\
"bundle_elements":["blood_cultures","broad_spectrum_antibiotics","iv_fluid_bolus","lactate_measurement","vasopressors"],\
"antibiotic_choice":"<str>","fluid_volume_ml":<integer>,"vasopressor_indicated":<true|false>,\
"rationale":"<str>","confidence":<0.0-1.0>}

If [TASK: paediatric_triage]:
{"priority":"low|medium|high|critical",\
"age_group":"infant|toddler|preschool|school_age|adolescent","pews_score":<integer 0-13>,\
"critical_sign":"<most abnormal vital sign>","recommended_action":"emergency_response|urgent_review|routine_monitoring",\
"rationale":"<str>","confidence":<0.0-1.0>}

If [TASK: medication_reconciliation]:
{"issues_found":["<str>"],"severity":"critical|high|medium|low","requires_pharmacist":<true|false>,\
"recommended_action":"safe_to_prescribe|modify_dose|withhold_drug|emergency_review",\
"drug_to_withhold":"<drug name or none>","rationale":"<str>","confidence":<0.0-1.0>}

If [TASK: icu_deterioration]:
{"sofa_score":<integer 0-24>,"primary_organ_failure":"cardiovascular|respiratory|renal|hepatic|neurological|coagulation",\
"deterioration_trend":"improving|stable|worsening",\
"intervention":"maintain_current|increase_support|emergency_escalation|prepare_palliation",\
"rationale":"<str>"}

If [TASK: sbar_handover]:
{"escalation_required":<true|false>,"priority":"low|medium|high|critical",\
"assessment":"<string describing clinical situation>",\
"recommendation":"routine_monitoring|urgent_review|emergency_response"}

If [TASK: differential_diagnosis]:
{"must_not_miss":"<life-threatening diagnosis to exclude>","top_diagnosis":"<most likely diagnosis>",\
"differentials":["<str>","<str>","<str>"],"first_investigation":"<most important first test>",\
"urgency":"immediate|urgent|routine"}"""

DEFAULT_SERVER_URL = "https://kunalkachru23-medical-triage-env.hf.space"
PRIORITY_MAP = {
    "immediate": "critical",
    "urgent": "high",
    "standard": "medium",
    "non_urgent": "low",
}
SEPSIS_BUNDLE_TOKEN_MAP = {
    "iv_antibiotics": "broad_spectrum_antibiotics",
    "iv_fluids": "iv_fluid_bolus",
    "lactate": "lactate_measurement",
}
RECOMMENDED_ACTION_TO_DETERIORATING = {
    "emergency_response": "emergency_response",
    "urgent_review": "escalate",
    "routine_monitoring": "monitor",
}
DETERIORATING_ACTION_MAP = {
    "monitor": "monitor",
    "observe": "monitor",
    "routine_monitoring": "monitor",
    "escalate": "escalate",
    "escalate_care": "escalate",
    "urgent_review": "escalate",
    "emergency_response": "emergency_response",
    "emergency": "emergency_response",
    "code_blue": "emergency_response",
}
URGENCY_MAP = {
    "critical": "immediate",
    "high": "urgent",
    "medium": "routine",
    "low": "routine",
}

# ── Env loading ────────────────────────────────────────────────────────────────


def load_hf_token_from_env(env_path: str = ".env") -> None:
    """
    Load only HF_TOKEN from a local .env file into process environment.
    Existing environment variables are preserved (no override).
    """
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == "HF_TOKEN":
                    os.environ.setdefault(key, value)
    except Exception as exc:
        print(f"[WARN] Could not parse {env_path}: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="GRPO fine-tuning on Medical Triage Env")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    p.add_argument(
        "--server-url",
        default=os.environ.get("SERVER_URL", DEFAULT_SERVER_URL),
        help="Medical Triage Environment server URL",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to load model on (auto = CUDA if available)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization (required for MPS/CPU — bitsandbytes not supported)",
    )
    p.add_argument(
        "--prompts-per-task",
        type=int,
        default=4,
        help="Number of environment resets per task for the training dataset (default: 4)",
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO group size G — completions generated per prompt (default: 4)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs (default: 1)",
    )
    p.add_argument(
        "--output-dir",
        default="grpo-medical-triage",
        help="Directory to save model adapter (default: grpo-medical-triage)",
    )
    p.add_argument(
        "--push-to-hub",
        default="",
        help="HF repo ID to push trained adapter (e.g. username/model-name). "
        "Requires HF_TOKEN env var.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Resume from a specific checkpoint directory path (e.g. output/checkpoint-50).",
    )
    p.add_argument(
        "--resume-latest",
        action="store_true",
        help="Auto-resume from the highest checkpoint-* inside --output-dir.",
    )
    return p.parse_args()


# ── Environment helpers ───────────────────────────────────────────────────────


def health_check(server_url: str) -> None:
    try:
        r = requests.get(f"{server_url}/health", timeout=15)
        r.raise_for_status()
        version = r.json().get("version", "unknown")
        print(f"  Server: {server_url}  (v{version})")
    except Exception as exc:
        print(f"\n[ERROR] Cannot reach environment server at {server_url}")
        print(f"        {exc}")
        sys.exit(1)


def build_dataset(server_url: str, prompts_per_task: int) -> Dataset:
    """Call /reset for each task to collect (prompt, task) pairs."""
    records = []
    for task in TASKS:
        successes = 0
        for _ in range(prompts_per_task):
            try:
                r = requests.post(
                    f"{server_url}/reset",
                    json={"task_id": task},
                    timeout=30,
                )
                r.raise_for_status()
                obs_data = r.json()["observation"]
                obs = obs_data.get("patient_history") or obs_data.get("patient_presentation", "")
                records.append({"prompt": f"[TASK: {task}]\n\n{obs}", "task": task})
                successes += 1
            except Exception as exc:
                print(f"    Warning: /reset failed ({task}): {exc}")
        print(f"    {task}: {successes}/{prompts_per_task} prompts collected")

    if not records:
        print("[ERROR] No prompts collected — check SERVER_URL and environment health.")
        sys.exit(1)

    return Dataset.from_list(records)


# ── Reward function ───────────────────────────────────────────────────────────


def make_reward_fn(server_url: str):
    """
    Returns a GRPO-compatible reward function with persistent HTTP session,
    automatic retries, and per-batch observability logging.
    Each completion gets a unique UUID session to avoid shared _default state.
    """
    _session = requests.Session()
    try:
        _retry = Retry(
            total=3, backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"], raise_on_status=False,
        )
    except TypeError:
        _retry = Retry(
            total=3, backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST"], raise_on_status=False,
        )
    _session.mount("https://", HTTPAdapter(max_retries=_retry))
    _session.mount("http://", HTTPAdapter(max_retries=_retry))

    def _parse_action_dict(completion: str) -> dict:
        """
        Parse model output into a flat action dict expected by /step.
        We strip markdown fences and parse JSON directly so rewards reflect
        the model's structured prediction rather than a wrapped free-text blob.
        """
        txt = re.sub(r"```(?:json)?\s*|\s*```", "", completion or "").strip()
        if not txt:
            return {}
        # Best-effort extraction if the model emits wrapper text around JSON.
        if not txt.startswith("{"):
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                txt = m.group(0)
        try:
            parsed = json.loads(txt)
            # Some models return {"response":"{...json...}"}; unwrap once.
            if isinstance(parsed, dict) and isinstance(parsed.get("response"), str):
                inner = parsed["response"].strip()
                if inner.startswith("{") and inner.endswith("}"):
                    try:
                        parsed = json.loads(inner)
                    except Exception:
                        pass
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _normalize_action_dict(task_id: str, action: dict) -> dict:
        """
        Light schema normalization to reduce avoidable floor rewards when the
        model uses semantically-correct but outdated enum tokens.
        """
        if not isinstance(action, dict):
            return {}
        out = dict(action)
        p = out.get("priority")
        if isinstance(p, str):
            out["priority"] = PRIORITY_MAP.get(p.strip().lower(), p)

        if task_id == "sepsis_bundle":
            elems = out.get("bundle_elements")
            if isinstance(elems, list):
                out["bundle_elements"] = [
                    SEPSIS_BUNDLE_TOKEN_MAP.get(str(e).strip(), str(e).strip())
                    for e in elems
                ]

        if task_id == "medication_reconciliation":
            sev = out.get("severity")
            if isinstance(sev, str) and sev.strip().lower() == "moderate":
                out["severity"] = "medium"

        if task_id == "deteriorating_patient":
            act = out.get("action")
            if isinstance(act, str):
                out["action"] = DETERIORATING_ACTION_MAP.get(act.strip().lower(), act.strip().lower())
            if "action" not in out and isinstance(out.get("recommended_action"), str):
                rec = out["recommended_action"].strip().lower()
                mapped = RECOMMENDED_ACTION_TO_DETERIORATING.get(rec)
                if mapped:
                    out["action"] = mapped
            if "rationale" not in out and isinstance(out.get("assessment"), str):
                out["rationale"] = out["assessment"]
            # Keep only relevant fields for this task to avoid schema confusion.
            out = {
                "action": (
                    out["action"]
                    if out.get("action") in {"monitor", "escalate", "emergency_response"}
                    else "escalate"
                ),
                "rationale": out.get("rationale", "Clinical deterioration risk present."),
                "confidence": out.get("confidence", 0.5),
            }

        if task_id == "differential_diagnosis":
            def _norm_text(v: object) -> str:
                s = str(v or "").strip().lower()
                return s.replace(" ", "_").replace("-", "_")

            # Normalize diagnosis-like fields to snake_case tokens the grader expects.
            for k in ("must_not_miss", "top_diagnosis", "first_investigation"):
                if k in out and isinstance(out[k], str):
                    out[k] = _norm_text(out[k])
            diffs = out.get("differentials")
            if isinstance(diffs, list):
                out["differentials"] = [_norm_text(d) for d in diffs if str(d).strip()]
            urg = out.get("urgency")
            if isinstance(urg, str):
                u = urg.strip().lower()
                out["urgency"] = URGENCY_MAP.get(u, u)

        return out

    def reward_fn(
        completions: List[str],
        prompts: Optional[List[str]] = None,
        task: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        batch_size = len(completions)
        if isinstance(task, str):
            task_ids = [task] * batch_size
        elif task is None:
            task_ids = ["simple_triage"] * batch_size
        else:
            task_ids = list(task)

        rewards: List[float] = []
        success_count = 0
        start_time = time.perf_counter()

        for idx, (completion, task_id) in enumerate(zip(completions, task_ids)):
            try:
                action_dict = _normalize_action_dict(task_id, _parse_action_dict(completion))
                session_id = str(uuid.uuid4())
                reset_r = _session.post(
                    f"{server_url}/reset",
                    json={"task_id": task_id, "session_id": session_id},
                    timeout=25,
                )
                reset_r.raise_for_status()
                step_r = _session.post(
                    f"{server_url}/step",
                    json={"session_id": session_id, "action": action_dict},
                    timeout=25,
                )
                step_r.raise_for_status()
                step_json = step_r.json()
                reward = float(step_json["reward"])
                success_count += 1
                print(f"    [{task_id}] reward={reward:.4f} ({idx+1}/{batch_size})")
                if reward <= 0.0001:
                    info = step_json.get("info", {})
                    feedback = ""
                    if isinstance(info, dict):
                        fb = info.get("feedback")
                        if isinstance(fb, str):
                            feedback = fb[:120]
                    keys = sorted(action_dict.keys()) if isinstance(action_dict, dict) else []
                    print(f"      floor-dbg keys={keys} feedback={feedback or 'n/a'}")
            except Exception as exc:
                reward = 0.0001  # open-interval floor
                print(f"    [reward error] {task_id}: {str(exc)[:80]}")
            rewards.append(reward)

        elapsed = time.perf_counter() - start_time
        success_rate = success_count / batch_size * 100
        print(f"  Batch: {success_count}/{batch_size} OK ({success_rate:.1f}%) | {elapsed:.2f}s")
        if success_rate < 70.0:
            print("  WARNING: Low reward success rate — training signal may be unstable.")
        return rewards

    return reward_fn


# ── Model loader ──────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_id: str, device: str, quantize: bool):
    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for decoder-only GRPO

    cuda_available = torch.cuda.is_available()

    if quantize and cuda_available:
        print("  Loading model with 4-bit NF4 quantization (CUDA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        print(f"  Loading model in {dtype} on {device} (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.to("cuda")

    if cuda_available:
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {used_gb:.1f} GB used / {total_gb:.0f} GB total")

    return model, tokenizer


def latest_checkpoint_path(output_dir: str) -> Optional[str]:
    """
    Return the highest checkpoint-* directory under output_dir, else None.
    Useful when a Colab run disconnects and you want to continue quickly.
    """
    if not os.path.isdir(output_dir):
        return None
    names = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not names:
        return None
    with_steps = []
    for n in names:
        m = re.search(r"checkpoint-(\d+)$", n)
        if m:
            with_steps.append((int(m.group(1)), n))
    if not with_steps:
        return None
    with_steps.sort(key=lambda x: x[0])
    return os.path.join(output_dir, with_steps[-1][1])


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # Support local .env workflows for HF_TOKEN in project root.
    load_hf_token_from_env(".env")
    args = parse_args()

    print("=" * 60)
    print("  Medical Triage — GRPO Fine-tuning")
    print(f"  Model:           {args.model}")
    print(f"  Server:          {args.server_url}")
    print(f"  Device:          {args.device}")
    print(f"  Quantize:        {not args.no_quantize}")
    print(f"  Prompts/task:    {args.prompts_per_task}")
    print(f"  Generations (G): {args.num_generations}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Resume latest:   {args.resume_latest}")
    print(f"  Resume path:     {args.resume_from_checkpoint or 'none'}")
    print("=" * 60)

    # 1. Health check
    health_check(args.server_url)

    # 2. Dataset
    print(f"\nBuilding dataset ({args.prompts_per_task} prompts × {len(TASKS)} tasks)...")
    dataset = build_dataset(args.server_url, args.prompts_per_task)
    print(f"  Total prompts: {len(dataset)}")

    # 3. Model + tokenizer
    print("\nLoading model...")
    quantize = not args.no_quantize
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, quantize)

    # 4. Apply chat template to prompts
    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        example["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return example

    dataset = dataset.map(format_prompt)

    # 5. GRPO config + trainer
    # Import here so Colab installs can run cell-by-cell if needed
    from trl import GRPOConfig, GRPOTrainer

    cuda_available = torch.cuda.is_available()

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        # 260 keeps hard-task JSON (e.g. differential_diagnosis) from truncating.
        max_completion_length=260,
        max_prompt_seq_length=512,
        truncation_prompt=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        fp16=cuda_available,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,  # preserves 'task' column for reward_fn
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[make_reward_fn(args.server_url)],
    )

    # 6. Train
    # Fresh run (default):
    #   python grpo_train.py --output-dir <dir>
    # Resume after interruption:
    #   python grpo_train.py --output-dir <dir> --resume-latest
    #   python grpo_train.py --resume-from-checkpoint <dir/checkpoint-N>
    print("\nStarting GRPO training...")
    print(f"  Steps: {len(dataset)} prompts × {args.epochs} epoch(s)")
    print("  (Each step calls the live environment for reward signals)\n")
    resume_path = ""
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
    elif args.resume_latest:
        found = latest_checkpoint_path(args.output_dir)
        if found:
            resume_path = found
            print(f"  Auto-detected latest checkpoint: {resume_path}")
        else:
            print("  No checkpoint-* found; starting fresh training.")

    if resume_path:
        if not os.path.isdir(resume_path):
            print(f"[ERROR] Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"  Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    # 7. Save
    output_path = f"{args.output_dir}/final"
    print(f"\nSaving adapter to {output_path}...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # 8. Optional Hub push
    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("  Warning: --push-to-hub set but HF_TOKEN not in environment. Skipping.")
        else:
            print(f"  Pushing adapter to Hub: {args.push_to_hub}")
            model.push_to_hub(args.push_to_hub, token=hf_token)
            tokenizer.push_to_hub(args.push_to_hub, token=hf_token)
            print("  Done — adapter live on HF Hub.")

    print("\n" + "=" * 60)
    print("  GRPO training complete.")
    print(f"  Adapter saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
