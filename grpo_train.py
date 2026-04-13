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

Env vars (optional overrides):
  SERVER_URL  — default: https://kunalkachru23-medical-triage-env.hf.space
  HF_TOKEN    — for pushing adapter to Hub (optional)
"""

import argparse
import os
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
You are a clinical triage AI assistant. Analyse the patient case and respond \
with a valid JSON object matching the task format below.

simple_triage / conflicting_vitals / masked_deterioration / demographic_fairness:
{"priority":"immediate|urgent|standard|non_urgent","news2_score":<int>,\
"critical_sign":"<string>","recommended_action":"<string>",\
"rationale":"<string>","confidence":<0.0-1.0>}

deteriorating_patient:
{"action":"monitor|escalate|emergency_response|comfort_care",\
"rationale":"<string>","confidence":<0.0-1.0>}

sepsis_bundle:
{"priority":"immediate|urgent|standard|non_urgent",\
"bundle_elements":["blood_cultures","iv_antibiotics","iv_fluids","lactate","vasopressors"],\
"antibiotic_choice":"<string>","fluid_volume_ml":<int>,\
"vasopressor_indicated":<bool>,"rationale":"<string>","confidence":<0.0-1.0>}

paediatric_triage:
{"priority":"immediate|urgent|standard|non_urgent",\
"age_group":"infant|toddler|preschool|school_age|adolescent","pews_score":<int>,\
"critical_sign":"<string>","recommended_action":"<string>",\
"rationale":"<string>","confidence":<0.0-1.0>}

medication_reconciliation:
{"issues_found":["<string>"],"severity":"critical|high|moderate|low",\
"requires_pharmacist":<bool>,\
"recommended_action":"safe_to_prescribe|modify_dose|withhold_drug|emergency_review",\
"drug_to_withhold":"<string>","rationale":"<string>","confidence":<0.0-1.0>}

icu_deterioration:
{"sofa_score":<int 0-24>,"primary_organ_failure":"cardiovascular|respiratory|renal|hepatic|neurological|coagulation",\
"deterioration_trend":"improving|stable|worsening",\
"intervention":"maintain_current|increase_support|emergency_escalation|prepare_palliation",\
"rationale":"<string>"}

sbar_handover:
{"escalation_required":<bool>,"priority":"low|medium|high|critical",\
"assessment":"<string describing clinical situation>",\
"recommendation":"routine_monitoring|urgent_review|emergency_response"}

differential_diagnosis:
{"must_not_miss":"<string — life-threatening diagnosis to exclude>",\
"top_diagnosis":"<string — most likely diagnosis>",\
"differentials":["<string>","<string>","<string>"],\
"first_investigation":"<string — most important first test>",\
"urgency":"immediate|urgent|routine"}

Respond with JSON only. No preamble, no markdown code fences."""

DEFAULT_SERVER_URL = "https://kunalkachru23-medical-triage-env.hf.space"

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
                session_id = str(uuid.uuid4())
                reset_r = _session.post(
                    f"{server_url}/reset",
                    json={"task_id": task_id, "session_id": session_id},
                    timeout=25,
                )
                reset_r.raise_for_status()
                step_r = _session.post(
                    f"{server_url}/step",
                    json={"session_id": session_id, "action": {"response": completion}},
                    timeout=25,
                )
                step_r.raise_for_status()
                reward = float(step_r.json()["reward"])
                success_count += 1
                print(f"    [{task_id}] reward={reward:.4f} ({idx+1}/{batch_size})")
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


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
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
        max_completion_length=300,
        max_prompt_seq_length=512,
        truncation_prompt=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
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
    print("\nStarting GRPO training...")
    print(f"  Steps: {len(dataset)} prompts × {args.epochs} epoch(s)")
    print("  (Each step calls the live environment for reward signals)\n")
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
