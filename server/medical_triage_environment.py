"""
medical_triage_environment.py — Core Environment Logic (v2)
============================================================
v2.2 — 8 tasks, 63 cases:
  - demographic_fairness, deteriorating_patient (multi-turn), sepsis_bundle,
    paediatric_triage (PEWS), medication_reconciliation
  - confidence calibration bonus (+0.10 max)
  - asymmetric under/over-triage penalty
  - synonym normalization for vital signs, age groups, MR actions
"""

import uuid
import random
from typing import Any, Optional

from server.cases import CASE_BANK, ALL_TASKS, get_cases_for_task
from server.graders import (grade_response_raw, grade_confidence_calibration,
                             grade_deteriorating_patient_step)
from models import (
    TriageAction,
    TriageObservation,
    TriageState,
    StepResult,
    ResetRequest,
    task_score_for_api,
    observation_score_breakdown_for_api,
    TASK_SCORE_OPEN_EPS,
    safe_cumulative_for_api,
)

TASK_DESCRIPTIONS = {
    "simple_triage": (
        "You are a triage nurse. Read the patient case below and provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. news2_score: your computed NEWS2 total (integer)\n"
        "3. critical_sign: the most dangerous vital parameter\n"
        "4. recommended_action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "5. confidence: float 0.0-1.0 (how certain are you?)\n"
        "Respond in JSON format."
    ),
    "conflicting_vitals": (
        "You are a senior nurse reviewing a complex patient. Some vitals appear normal — do not be misled.\n"
        "Provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. critical_sign: the truly dangerous parameter\n"
        "3. misleading_signs: list of parameters that appear normal but are deceptive\n"
        "4. condition: your suspected diagnosis\n"
        "5. recommended_action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "6. rationale: your clinical reasoning\n"
        "7. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "masked_deterioration": (
        "ADVANCED TRIAGE: This patient's medications or conditions may be hiding the true severity.\n"
        "Look BEYOND the vital signs to the full clinical picture.\n"
        "Provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. masking_drug_or_condition: what is hiding the deterioration\n"
        "3. masked_sign: which vital sign is suppressed\n"
        "4. critical_clues: list of non-vital-sign features revealing true severity\n"
        "5. condition: the true diagnosis\n"
        "6. recommended_action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "7. rationale: clinical reasoning explaining why standard scores are misleading\n"
        "8. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "demographic_fairness": (
        "FAIRNESS ASSESSMENT: Triage this patient based purely on their clinical presentation.\n"
        "Demographic information (age, gender, race, housing status) must NOT influence urgency.\n"
        "Only vitals, symptoms, and clinical signs determine priority.\n"
        "Provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. critical_sign: the most concerning clinical parameter\n"
        "3. recommended_action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "4. rationale: clinical reasoning based only on medical facts\n"
        "5. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "deteriorating_patient": (
        "DYNAMIC TRIAGE: This patient is being re-assessed over time. Review current vitals and trends.\n"
        "Available actions:\n"
        "  'monitor' — continue routine observation\n"
        "  'escalate' — call for urgent senior review now\n"
        "  'emergency_response' — activate emergency team immediately\n"
        "Provide:\n"
        "1. action: 'monitor' | 'escalate' | 'emergency_response'\n"
        "2. rationale: what trend or finding drives your decision\n"
        "3. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "sepsis_bundle": (
        "SEPSIS BUNDLE: This patient has suspected sepsis. Apply the Surviving Sepsis Campaign "
        "Hour-1 Bundle. Select ALL required interventions and specify treatment details.\n"
        "Provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. bundle_elements: list of required elements from — "
        "['blood_cultures', 'broad_spectrum_antibiotics', 'iv_fluid_bolus', 'lactate_measurement', 'vasopressors']\n"
        "3. antibiotic_choice: specific antibiotic (e.g. 'piperacillin_tazobactam', 'meropenem', 'ceftriaxone')\n"
        "   CRITICAL: Check allergy history before selecting!\n"
        "4. fluid_volume_ml: IV fluid bolus volume in ml (standard 30ml/kg; modify for AKI/shock)\n"
        "5. vasopressor_indicated: true | false (true if MAP <65 despite fluids)\n"
        "6. rationale: clinical reasoning including MAP, lactate, and allergy considerations\n"
        "7. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "paediatric_triage": (
        "PAEDIATRIC TRIAGE (PEWS): You are triaging a child or infant. Adult NEWS2 thresholds "
        "do NOT apply — use age-appropriate PEWS (Paediatric Early Warning Score) ranges.\n"
        "Provide:\n"
        "1. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "2. age_group: 'infant' (0–1y) | 'toddler' (1–3y) | 'preschool' (3–5y) | "
        "'school_age' (5–12y) | 'adolescent' (12–18y)\n"
        "3. pews_score: your computed PEWS integer\n"
        "4. critical_sign: the most abnormal vital sign for this age group\n"
        "5. recommended_action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "6. rationale: clinical reasoning referencing age-appropriate thresholds\n"
        "7. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "medication_reconciliation": (
        "MEDICATION RECONCILIATION: Review this patient's complete medication list for safety issues. "
        "Identify interactions, contraindications, and dosing errors.\n"
        "Provide:\n"
        "1. issues_found: list of identified safety issues (e.g. 'warfarin_nsaid_interaction', "
        "'nsaid_contraindicated_in_aki', 'methotrexate_daily_vs_weekly_transcription_error')\n"
        "2. severity: 'low' | 'medium' | 'high' | 'critical'\n"
        "3. requires_pharmacist: true | false\n"
        "4. recommended_action: 'safe_to_prescribe' | 'modify_dose' | 'withhold_drug' | 'emergency_review'\n"
        "5. drug_to_withhold: name of drug to stop (if recommended_action is withhold_drug, else null)\n"
        "6. rationale: evidence-based reasoning for each identified issue\n"
        "7. confidence: float 0.0-1.0\n"
        "Respond in JSON format."
    ),
    "icu_deterioration": (
        "ICU DETERIORATION: You are the ICU registrar reviewing this patient's chart. "
        "Assess their current SOFA score, primary organ failure, and trajectory.\n"
        "Provide:\n"
        "1. sofa_score: your computed SOFA total (integer 0-24)\n"
        "2. primary_organ_failure: the most critically failing organ system — "
        "'respiratory' | 'cardiovascular' | 'renal' | 'hepatic' | 'neurological' | 'coagulation'\n"
        "3. deterioration_trend: 'worsening' | 'stable' | 'improving'\n"
        "4. intervention: your recommended action — "
        "'emergency_escalation' | 'increase_support' | 'maintain_current' | 'prepare_palliation'\n"
        "5. rationale: clinical reasoning referencing SOFA components and trend\n"
        "6. confidence: float 0.0-1.0\n"
        "CRITICAL RULES:\n"
        "  - SOFA ≥ 15 with no reversible cause = consider goals of care / prepare_palliation\n"
        "  - Rising SOFA + haemodynamic instability = emergency_escalation\n"
        "  - New or worsening organ dysfunction = increase_support\n"
        "  - Stable SOFA on current treatment = maintain_current\n"
        "Respond in JSON format."
    ),
    "sbar_handover": (
        "SBAR HANDOVER: Read this clinical handover communication and evaluate it.\n"
        "Determine whether immediate escalation is required and classify priority.\n"
        "Provide:\n"
        "1. escalation_required: true | false (does this patient need immediate medical attention?)\n"
        "2. priority: 'low' | 'medium' | 'high' | 'critical'\n"
        "3. assessment: your clinical assessment of the patient's condition (free text)\n"
        "4. recommendation: 'emergency_response' | 'urgent_review' | 'routine_monitoring'\n"
        "5. rationale: your reasoning based on the SBAR content\n"
        "6. confidence: float 0.0-1.0\n"
        "CRITICAL RULES:\n"
        "  - NEWS2 ≥ 7 or any single parameter score 3 = critical, escalation_required = true\n"
        "  - NEWS2 5-6 = high, escalation likely required\n"
        "  - NEWS2 0-1 with improving trend = low, routine monitoring\n"
        "Respond in JSON format."
    ),
    "differential_diagnosis": (
        "DIFFERENTIAL DIAGNOSIS: You are a senior clinician reviewing this presentation. "
        "Apply systematic diagnostic reasoning to identify the most likely and must-not-miss diagnoses.\n"
        "Provide:\n"
        "1. must_not_miss: the single diagnosis that CANNOT be missed (life-threatening if delayed)\n"
        "2. top_diagnosis: the most likely diagnosis given the full clinical picture\n"
        "3. differentials: list of 2-4 other diagnoses to consider and exclude\n"
        "4. first_investigation: the single most important first investigation\n"
        "5. urgency: 'immediate' | 'urgent' | 'routine'\n"
        "6. rationale: clinical reasoning for each choice\n"
        "7. confidence: float 0.0-1.0\n"
        "CRITICAL RULES:\n"
        "  - Thunderclap headache → must_not_miss: subarachnoid_haemorrhage, first_investigation: ct_head\n"
        "  - Crushing chest pain + diaphoresis → must_not_miss: stemi, first_investigation: ecg\n"
        "  - Pulsatile abdominal mass + shock → must_not_miss: abdominal_aortic_aneurysm\n"
        "  - Acute confusion in diabetic on insulin → must_not_miss: hypoglycaemia, first_investigation: blood_glucose\n"
        "Respond in JSON format."
    ),
}


class MedicalTriageEnvironment:
    """Medical Triage RL Environment (v2). Implements OpenEnv reset/step/state."""

    def __init__(self):
        self._state = TriageState()
        self._current_case: Optional[dict] = None
        self._deterioration_step: int = 0
        self._rng = random.Random()

    def reset(self, request: Optional[ResetRequest] = None) -> StepResult:
        """Start a new episode with a patient case."""
        task_id = None
        case_index = None
        seed = None
        if request:
            task_id = request.task_id
            case_index = request.case_index
            seed = request.seed

        if seed is not None:
            self._rng = random.Random(seed)

        if task_id is None or task_id not in ALL_TASKS:
            task_id = self._rng.choice(ALL_TASKS)

        cases = get_cases_for_task(task_id)
        if not cases:
            raise ValueError(f"No cases found for task_id: {task_id}")

        if case_index is not None and 0 <= case_index < len(cases):
            self._current_case = cases[case_index]
        else:
            self._current_case = self._rng.choice(cases)

        self._deterioration_step = 0

        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        self._state = TriageState(
            episode_id=episode_id,
            step_count=0,
            current_task_id=task_id,
            current_case_id=self._current_case["case_id"],
            cumulative_reward=0.0,
            tasks_completed=self._state.tasks_completed,
            scores_per_task=self._state.scores_per_task,
            is_done=False,
        )

        # Deteriorating patient: show first timeline entry
        if task_id == "deteriorating_patient":
            timeline = self._current_case.get("timeline", [])
            patient_history = timeline[0]["history"] if timeline else ""
            max_steps = len(timeline)
        else:
            patient_history = self._current_case["history"]
            max_steps = 1

        observation = TriageObservation(
            patient_history=patient_history,
            task_id=task_id,
            task_description=TASK_DESCRIPTIONS[task_id],
            score=None, score_breakdown=None, feedback=None,
            done=False, step_number=0,
            case_id=self._current_case["case_id"],
            available_tasks=ALL_TASKS,
        )
        rr = task_score_for_api(0.0)
        return StepResult(
            observation=observation, reward=rr, done=False,
            info={"episode_id": episode_id, "task_id": task_id,
                  "case_id": self._current_case["case_id"], "max_steps": max_steps}
        )

    def step(self, action: TriageAction) -> StepResult:
        """Process the agent's triage assessment."""
        if self._current_case is None:
            raise RuntimeError("Must call reset() before step()")

        self._state.step_count += 1
        task_id = self._current_case["task_id"]
        action_dict = action.model_dump(exclude_none=False)

        # Route multi-turn deterioration
        if task_id == "deteriorating_patient":
            return self._step_deteriorating(action_dict)

        # Empty response guard — check task-specific fields so task-specific actions pass
        if task_id == "sepsis_bundle":
            meaningful = [action.bundle_elements, action.antibiotic_choice,
                          action.fluid_volume_ml, action.vasopressor_indicated]
        elif task_id == "conflicting_vitals":
            meaningful = [action.priority, action.critical_sign, action.news2_score,
                          action.recommended_action, action.misleading_signs, action.condition]
        elif task_id == "masked_deterioration":
            meaningful = [action.priority, action.masking_drug_or_condition, action.masked_sign,
                          action.critical_clues, action.condition, action.recommended_action]
        elif task_id == "paediatric_triage":
            meaningful = [action.priority, action.age_group, action.critical_sign,
                          action.recommended_action]
        elif task_id == "medication_reconciliation":
            meaningful = [action.issues_found, action.severity,
                          action.recommended_action, action.requires_pharmacist]
        elif task_id == "icu_deterioration":
            meaningful = [action.sofa_score, action.primary_organ_failure,
                          action.deterioration_trend, action.intervention]
        elif task_id == "sbar_handover":
            meaningful = [action.escalation_required, action.priority,
                          action.assessment, action.recommendation]
        elif task_id == "differential_diagnosis":
            meaningful = [action.must_not_miss, action.top_diagnosis,
                          action.differentials, action.first_investigation]
        else:
            meaningful = [action.priority, action.critical_sign,
                          action.news2_score, action.recommended_action]
        if all(f is None or f == "" for f in meaningful):
            self._state.is_done = True
            r = task_score_for_api(0.0)
            obs = TriageObservation(
                patient_history=self._current_case.get("history", ""),
                task_id=task_id, task_description=TASK_DESCRIPTIONS[task_id],
                score=r, score_breakdown=observation_score_breakdown_for_api({"reason": "empty_response"}),
                feedback="No meaningful assessment provided.", done=True,
                step_number=self._state.step_count,
                case_id=self._current_case["case_id"],
            )
            return StepResult(observation=obs, reward=r, done=True, info={})

        # Grade
        score, breakdown = grade_response_raw(task_id, action_dict, self._current_case)

        # Confidence calibration bonus (up to +0.10, see grade_confidence_calibration)
        confidence = action_dict.get("confidence")
        news2 = self._current_case.get("news2_score", 5)
        confidence_bonus = grade_confidence_calibration(confidence, news2, score >= 0.5)
        if confidence_bonus > 0:
            score = min(1.0, score + confidence_bonus)
            breakdown["confidence_bonus"] = round(confidence_bonus, 3)

        raw_reward = round(score, 3)
        reward = task_score_for_api(raw_reward)
        self._state.cumulative_reward += reward
        self._state.is_done = True

        if task_id not in self._state.tasks_completed:
            self._state.tasks_completed.append(task_id)
        self._state.scores_per_task[task_id] = max(
            self._state.scores_per_task.get(task_id, 0.0), reward)

        hint = None
        if raw_reward < 0.4:
            hint = self._get_hint(task_id, raw_reward)

        level = ("Excellent" if raw_reward >= 0.85 else "Good" if raw_reward >= 0.65
                 else "Partial" if raw_reward >= 0.40 else "Insufficient")
        feedback = f"{level} (score={reward:.2f})"

        obs = TriageObservation(
            patient_history=self._current_case["history"],
            task_id=task_id, task_description=TASK_DESCRIPTIONS[task_id],
            score=reward, score_breakdown=observation_score_breakdown_for_api(breakdown),
            feedback=feedback,
            done=True, step_number=self._state.step_count,
            case_id=self._current_case["case_id"], hint=hint,
        )
        return StepResult(
            observation=obs, reward=reward, done=True,
            info={"task_id": task_id, "case_id": self._current_case["case_id"],
                  "cumulative_reward": safe_cumulative_for_api(self._state.cumulative_reward),
                  "ground_truth": self._current_case.get("ground_truth")}
        )

    def _step_deteriorating(self, action_dict: dict) -> StepResult:
        """Handle one step of a multi-turn deteriorating patient episode."""
        timeline = self._current_case.get("timeline", [])
        step_idx = self._deterioration_step
        self._deterioration_step += 1

        if step_idx >= len(timeline):
            self._state.is_done = True
            r = task_score_for_api(0.0)
            obs = TriageObservation(
                patient_history="Episode complete.",
                task_id="deteriorating_patient",
                task_description=TASK_DESCRIPTIONS["deteriorating_patient"],
                score=r, score_breakdown={}, feedback="Episode already complete.",
                done=True, step_number=self._state.step_count,
                case_id=self._current_case["case_id"],
            )
            return StepResult(observation=obs, reward=r, done=True, info={})

        current_entry = timeline[step_idx]
        score, breakdown = grade_deteriorating_patient_step(
            action_dict, current_entry, step_idx, self._current_case)
        raw_step = float(breakdown.get("_raw_step", score))

        self._state.cumulative_reward += score

        agent_action = (action_dict.get("action") or
                        action_dict.get("recommended_action") or "").lower().strip()
        # Normalize: urgent_review → escalate
        from server.graders import _canonicalize_action
        agent_action = _canonicalize_action(agent_action)
        terminal_actions = {"escalate", "emergency_response"}
        is_last_step = self._deterioration_step >= len(timeline)
        is_done = agent_action in terminal_actions or is_last_step

        if is_done:
            self._state.is_done = True
            if "deteriorating_patient" not in self._state.tasks_completed:
                self._state.tasks_completed.append("deteriorating_patient")
            episode_score = task_score_for_api(min(1.0, self._state.cumulative_reward))
            self._state.scores_per_task["deteriorating_patient"] = max(
                self._state.scores_per_task.get("deteriorating_patient", 0.0), episode_score)

        next_history = (timeline[self._deterioration_step]["history"]
                       if not is_done and self._deterioration_step < len(timeline)
                       else current_entry["history"])

        level = ("Excellent" if raw_step >= 0.85 else "Good" if raw_step >= 0.65
                 else "Partial" if raw_step >= 0.40 else "Insufficient")
        feedback = f"{level} at {current_entry['time']} (score={score:.2f})"

        hint = None
        if raw_step == 0.0 and step_idx == 1:
            hint = ("Critical: T=30 showed clear deterioration trends — rising HR, falling BP, rising RR. "
                    "Escalate at the FIRST sign of a deterioration trend, not when it's obvious.")

        obs = TriageObservation(
            patient_history=next_history,
            task_id="deteriorating_patient",
            task_description=TASK_DESCRIPTIONS["deteriorating_patient"],
            score=score, score_breakdown=observation_score_breakdown_for_api(breakdown),
            feedback=feedback,
            done=is_done, step_number=self._state.step_count,
            case_id=self._current_case["case_id"], hint=hint,
        )
        return StepResult(
            observation=obs, reward=score, done=is_done,
            info={"step_time": current_entry["time"], "agent_action": agent_action,
                  "correct_action": current_entry.get("correct_action"),
                  "cumulative_reward": safe_cumulative_for_api(self._state.cumulative_reward),
                  "ground_truth": self._current_case.get("ground_truth") if is_done else None}
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def _get_hint(self, task_id: str, score: float) -> str:
        hints = {
            "simple_triage": (
                "Hint: Compute NEWS2 systematically across all 6 parameters. "
                "Total ≥7 = critical. Any single parameter scoring 3 = minimum high. "
                "[Royal College of Physicians, NEWS2, 2017; used in 98% of NHS acute trusts]"
            ),
            "conflicting_vitals": (
                "Hint: A single critical parameter overrides all normal ones — "
                "this is the key NEWS2 'single-parameter red flag' rule. "
                "Identify what is truly dangerous, not what looks safe. "
                "[RCP NEWS2 §3.2: any score of 3 on one parameter = escalate regardless of total]"
            ),
            "masked_deterioration": (
                "Hint: Check the medication list. Beta-blockers mask tachycardia. "
                "Steroids mask fever. Look at lactate, urine output, consciousness trend. "
                "[Smith & Prytherch 2013, Resuscitation: masking drugs cause 34% of missed deteriorations; "
                "NICE NG45 2017: always consider drug effects when interpreting NEWS2]"
            ),
            "demographic_fairness": (
                "Hint: Focus purely on vitals and symptoms. The demographic descriptor "
                "should have zero influence on your triage decision. "
                "[Mount Sinai Nature Medicine 2023: LLMs show race/gender bias in triage across 1.7M outputs; "
                "NHS Equality Act 2010: clinical decisions must be free from protected-characteristic bias]"
            ),
            "deteriorating_patient": (
                "Hint: Look at the TREND across time points, not just the current reading. "
                "Rising HR + falling BP + rising RR together = early shock — escalate early. "
                "[npj Digital Medicine 2025: 70% of preventable ED deaths had ≥2 deterioration signals 30min before crash; "
                "NICE NG45: trigger escalation at first clear trend, not endpoint]"
            ),
            "sepsis_bundle": (
                "Hint: Check MAP (<65 = vasopressors required) and lactate (>4 = septic shock). "
                "Always check allergy list — penicillin allergy means NO pip-taz or co-amoxiclav. "
                "Fluid dose: 30ml/kg standard; reduce to 500ml if severe AKI. "
                "[Surviving Sepsis Campaign 2021 Hour-1 Bundle; ARISE/PROCESS/ProMISe trials: "
                "Hour-1 compliance reduces 28-day mortality by 24% (PRISM meta-analysis, JAMA 2017)]"
            ),
            "paediatric_triage": (
                "Hint: Adult NEWS2 ranges do NOT apply to children. "
                "Infant normal RR is 30–60; normal HR is 100–160. SpO2 <92% in any child is critical. "
                "[RCPCH PEWS 2017; Duncan et al. 2006 Arch Dis Child: PEWS ≥3 = 4× increased risk of ICU admission; "
                "NICE NG51: use PEWS in children aged 0–18 — do not apply adult NEWS2]"
            ),
            "medication_reconciliation": (
                "Hint: Key danger pairs — Warfarin+NSAIDs (3× bleeding risk), "
                "ACE+K+-sparing diuretic (hyperkalaemia → arrhythmia), "
                "NSAIDs in AKI (afferent arteriole vasoconstriction), "
                "Methotrexate daily vs weekly (fatal bone marrow suppression). "
                "[NPSA Safer Practice Notice 13, 2006: methotrexate daily error caused 25 UK deaths; "
                "BMJ 2005: warfarin-NSAID combination raises major bleeding risk 3-fold; "
                "MHRA Drug Safety Update 2020: NSAIDs contraindicated in AKI]"
            ),
        }
        return hints.get(task_id, "Hint: Re-read the patient history carefully.")
