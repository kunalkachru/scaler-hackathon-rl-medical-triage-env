"""
medical_triage_environment.py — Core Environment Logic (v2)
============================================================
v2 adds:
  - demographic_fairness task (5 tasks total)
  - deteriorating_patient multi-turn episodes
  - confidence calibration bonus
  - asymmetric under/over-triage penalty
"""

import uuid
import random
from typing import Any, Optional

from server.cases import CASE_BANK, ALL_TASKS, get_cases_for_task
from server.graders import (grade_response, grade_confidence_calibration,
                             grade_deteriorating_patient_step)
from models import (
    TriageAction, TriageObservation, TriageState, StepResult, ResetRequest
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
        return StepResult(
            observation=observation, reward=0.0, done=False,
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

        # Empty response guard
        meaningful = [action.priority, action.critical_sign,
                      action.news2_score, action.recommended_action]
        if all(f is None or f == "" for f in meaningful):
            self._state.is_done = True
            obs = TriageObservation(
                patient_history=self._current_case.get("history", ""),
                task_id=task_id, task_description=TASK_DESCRIPTIONS[task_id],
                score=0.0, score_breakdown={"reason": "empty_response"},
                feedback="No meaningful assessment provided.", done=True,
                step_number=self._state.step_count,
                case_id=self._current_case["case_id"],
            )
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        # Grade
        score, breakdown = grade_response(task_id, action_dict, self._current_case)

        # Confidence calibration bonus (up to +0.10)
        confidence = action_dict.get("confidence")
        news2 = self._current_case.get("news2_score", 5)
        confidence_bonus = grade_confidence_calibration(confidence, news2, score >= 0.5)
        if confidence_bonus > 0:
            score = min(1.0, score + confidence_bonus)
            breakdown["confidence_bonus"] = round(confidence_bonus, 3)

        reward = round(score, 3)
        self._state.cumulative_reward += reward
        self._state.is_done = True

        if task_id not in self._state.tasks_completed:
            self._state.tasks_completed.append(task_id)
        self._state.scores_per_task[task_id] = max(
            self._state.scores_per_task.get(task_id, 0.0), reward)

        hint = None
        if reward < 0.4:
            hint = self._get_hint(task_id, reward)

        level = ("Excellent" if reward >= 0.85 else "Good" if reward >= 0.65
                 else "Partial" if reward >= 0.40 else "Insufficient")
        feedback = f"{level} (score={reward:.2f})"

        obs = TriageObservation(
            patient_history=self._current_case["history"],
            task_id=task_id, task_description=TASK_DESCRIPTIONS[task_id],
            score=reward, score_breakdown=breakdown, feedback=feedback,
            done=True, step_number=self._state.step_count,
            case_id=self._current_case["case_id"], hint=hint,
        )
        return StepResult(
            observation=obs, reward=reward, done=True,
            info={"task_id": task_id, "case_id": self._current_case["case_id"],
                  "cumulative_reward": round(self._state.cumulative_reward, 3),
                  "ground_truth": self._current_case.get("ground_truth")}
        )

    def _step_deteriorating(self, action_dict: dict) -> StepResult:
        """Handle one step of a multi-turn deteriorating patient episode."""
        timeline = self._current_case.get("timeline", [])
        step_idx = self._deterioration_step
        self._deterioration_step += 1

        if step_idx >= len(timeline):
            self._state.is_done = True
            obs = TriageObservation(
                patient_history="Episode complete.",
                task_id="deteriorating_patient",
                task_description=TASK_DESCRIPTIONS["deteriorating_patient"],
                score=0.0, score_breakdown={}, feedback="Episode already complete.",
                done=True, step_number=self._state.step_count,
                case_id=self._current_case["case_id"],
            )
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        current_entry = timeline[step_idx]
        score, breakdown = grade_deteriorating_patient_step(
            action_dict, current_entry, step_idx, self._current_case)

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
            self._state.scores_per_task["deteriorating_patient"] = max(
                self._state.scores_per_task.get("deteriorating_patient", 0.0), score)

        next_history = (timeline[self._deterioration_step]["history"]
                       if not is_done and self._deterioration_step < len(timeline)
                       else current_entry["history"])

        level = ("Excellent" if score >= 0.85 else "Good" if score >= 0.65
                 else "Partial" if score >= 0.40 else "Insufficient")
        feedback = f"{level} at {current_entry['time']} (score={score:.2f})"

        hint = None
        if score == 0.0 and step_idx == 1:
            hint = ("Critical: T=30 showed clear deterioration trends — rising HR, falling BP, rising RR. "
                    "Escalate at the FIRST sign of a deterioration trend, not when it's obvious.")

        obs = TriageObservation(
            patient_history=next_history,
            task_id="deteriorating_patient",
            task_description=TASK_DESCRIPTIONS["deteriorating_patient"],
            score=round(score, 3), score_breakdown=breakdown, feedback=feedback,
            done=is_done, step_number=self._state.step_count,
            case_id=self._current_case["case_id"], hint=hint,
        )
        return StepResult(
            observation=obs, reward=round(score, 3), done=is_done,
            info={"step_time": current_entry["time"], "agent_action": agent_action,
                  "correct_action": current_entry.get("correct_action"),
                  "cumulative_reward": round(self._state.cumulative_reward, 3),
                  "ground_truth": self._current_case.get("ground_truth") if is_done else None}
        )

    @property
    def state(self) -> TriageState:
        return self._state

    def _get_hint(self, task_id: str, score: float) -> str:
        hints = {
            "simple_triage": (
                "Hint: Compute NEWS2 systematically across all 6 parameters. "
                "Total ≥7 = critical. Any single parameter scoring 3 = minimum high."
            ),
            "conflicting_vitals": (
                "Hint: A single critical parameter overrides all normal ones. "
                "Identify what is truly dangerous, not what looks safe."
            ),
            "masked_deterioration": (
                "Hint: Check the medication list. Beta-blockers mask tachycardia. "
                "Steroids mask fever. Look at lactate, urine output, consciousness trend."
            ),
            "demographic_fairness": (
                "Hint: Focus purely on vitals and symptoms. The demographic descriptor "
                "should have zero influence on your triage decision."
            ),
            "deteriorating_patient": (
                "Hint: Look at the TREND across time points, not just the current reading. "
                "Rising HR + falling BP + rising RR together = early shock — escalate early."
            ),
        }
        return hints.get(task_id, "Hint: Re-read the patient history carefully.")
