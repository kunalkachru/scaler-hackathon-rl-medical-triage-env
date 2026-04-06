"""
models.py — Typed Action / Observation / State Models
=======================================================
WHY THIS FILE EXISTS:
  The OpenEnv spec REQUIRES typed Pydantic models for:
  - Action (what the agent sends)
  - Observation (what the environment returns)
  - State (episode metadata)

  These types are what `openenv validate` checks for.
  They also give IDE autocomplete and catch bugs before runtime.

  We use Pydantic v2 (which ships with FastAPI) for:
  - JSON serialization/deserialization handled automatically
  - Field validation with clear error messages
  - OpenAPI schema generation (free /docs page on HF Space)
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ─────────────────────────────────────────────────────────────
# ACTION — what the agent sends to the environment
# ─────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    The agent's triage assessment response.

    This is what the LLM agent submits after reading the patient case.
    All fields are optional at the model level — graders handle missing fields
    gracefully with partial credit.
    """

    # Core required for all tasks
    priority: Optional[str] = Field(
        default="",
        description="Urgency classification: 'low' | 'medium' | 'high' | 'critical'"
    )

    @field_validator("priority", mode="before")
    @classmethod
    def coerce_priority(cls, v: Any) -> str:
        """Coerce None/null to empty string so graders receive '' and score 0."""
        return "" if v is None else str(v)

    # Task 1 + 2
    news2_score: Optional[int] = Field(
        default=None,
        description="Agent's computed NEWS2 score (integer)"
    )
    critical_sign: Optional[str] = Field(
        default=None,
        description="The most clinically dangerous vital parameter (e.g. 'spo2', 'systolic_bp')"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended clinical action: 'emergency_response' | 'urgent_review' | 'routine_monitoring'"
    )

    # Task 2 specific
    misleading_signs: Optional[list[str]] = Field(
        default=None,
        description="Signs that appear normal but are misleading in this context"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Suspected clinical condition or diagnosis"
    )

    # Task 3 specific
    masking_drug_or_condition: Optional[str] = Field(
        default=None,
        description="The drug or condition masking the deterioration (e.g. 'bisoprolol', 'prednisolone')"
    )
    masked_sign: Optional[str] = Field(
        default=None,
        description="The vital sign that is pharmacologically or physiologically masked"
    )
    critical_clues: Optional[list[str]] = Field(
        default=None,
        description="Non-standard clues that reveal true severity (e.g. 'lactate', 'urine_output_reduced')"
    )

    # Task 5 specific (multi-turn deteriorating patient)
    action: Optional[str] = Field(
        default=None,
        description="Decision for dynamic triage: 'monitor' | 'escalate' | 'emergency_response'"
    )

    # Optional free-text rationale (used in Task 2 grader)
    rationale: Optional[str] = Field(
        default=None,
        description="Agent's clinical reasoning and explanation"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score in [0.0, 1.0] used for calibration bonus"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "priority": "critical",
            "news2_score": 9,
            "critical_sign": "systolic_bp",
            "recommended_action": "emergency_response",
            "rationale": "Patient has hypotension (BP=88), tachycardia, and elevated RR. NEWS2=9 indicates critical deterioration."
        }
    })


# ─────────────────────────────────────────────────────────────
# OBSERVATION — what the environment returns to the agent
# ─────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """
    The environment's response after each action.

    Contains:
    - The patient case text (on reset and step)
    - Current task information
    - Score feedback (reward signal)
    - Whether the episode is done
    """

    # Patient presentation (the "game state" the agent reads)
    patient_history: str = Field(
        default="",
        description="Full patient case description including vitals and history"
    )
    task_id: str = Field(
        default="",
        description="Current task ID (one of all supported tasks in CASE_BANK)"
    )
    task_description: str = Field(
        default="",
        description="Human-readable task instructions for the agent"
    )

    # Feedback after step()
    score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Score for this step: 0.0 (completely wrong) to 1.0 (perfect)"
    )
    score_breakdown: Optional[dict[str, Any]] = Field(
        default=None,
        description="Per-dimension score breakdown for interpretability"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Textual feedback explaining the score"
    )

    # Episode management
    done: bool = Field(
        default=False,
        description="True when the episode is complete"
    )
    step_number: int = Field(
        default=0,
        description="Current step within the episode (max 3 steps per case)"
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Identifier for the current patient case"
    )

    # Task metadata
    available_tasks: Optional[list[str]] = Field(
        default=None,
        description="List of all available task IDs"
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint provided after a failed attempt"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "patient_history": "72-year-old male. RR=24, SpO2=93%, BP=105/70...",
            "task_id": "simple_triage",
            "task_description": "Compute NEWS2 score and classify urgency.",
            "score": 0.85,
            "score_breakdown": {"priority": 0.35, "news2_score": 0.20, "critical_sign": 0.20, "recommended_action": 0.10},
            "done": True,
            "step_number": 1,
            "case_id": "ST001"
        }
    })


# ─────────────────────────────────────────────────────────────
# STATE — episode metadata (returned by /state endpoint)
# ─────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """
    Episode-level metadata tracked by the environment.

    This is what the /state endpoint returns.
    Useful for training loops to track progress across episodes.
    """

    episode_id: Optional[str] = Field(
        default=None,
        description="Unique ID for this episode"
    )
    step_count: int = Field(
        default=0,
        description="Total steps taken in this episode"
    )
    current_task_id: Optional[str] = Field(
        default=None,
        description="Currently active task"
    )
    current_case_id: Optional[str] = Field(
        default=None,
        description="Currently active patient case ID"
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of all rewards in this episode"
    )
    tasks_completed: list[str] = Field(
        default_factory=list,
        description="List of task IDs completed in this session"
    )
    scores_per_task: dict[str, float] = Field(
        default_factory=dict,
        description="Score achieved per task in this session"
    )
    is_done: bool = Field(
        default=False,
        description="Whether the current episode is complete"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "episode_id": "ep-abc123",
            "step_count": 3,
            "current_task_id": "masked_deterioration",
            "current_case_id": "MD001",
            "cumulative_reward": 2.45,
            "tasks_completed": ["simple_triage", "conflicting_vitals"],
            "scores_per_task": {"simple_triage": 0.87, "conflicting_vitals": 0.72},
            "is_done": False
        }
    })


# ─────────────────────────────────────────────────────────────
# REQUEST WRAPPERS (used by the API endpoints)
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for POST /reset"""
    task_id: Optional[str] = Field(
        default=None,
        description="Task to start. If omitted, a random task is selected."
    )
    case_index: Optional[int] = Field(
        default=None,
        description="Specific case index within the task (for reproducibility). Random if None."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for concurrent isolation. Returned by reset(); pass it to step() and state(). If omitted, a new session is created automatically."
    )


class StepRequest(BaseModel):
    """Request body for POST /step"""
    action: TriageAction
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier returned by reset(). Required when running concurrent episodes."
    )


class StepInfo(BaseModel):
    """
    Typed metadata returned alongside every step/reset response.

    All fields are optional so that empty info={} from error paths
    remains valid, and forward-compatible with future additions.
    """
    session_id:  Optional[str] = Field(
        default=None,
        description="Session identifier — pass this to step() and state() for correct routing"
    )
    episode_id:  Optional[str] = Field(
        default=None,
        description="Unique ID for the current episode"
    )
    task_id:     Optional[str] = Field(
        default=None,
        description="Task ID active in this episode"
    )
    case_id:     Optional[str] = Field(
        default=None,
        description="Case ID active in this episode"
    )
    max_steps:   Optional[int] = Field(
        default=None,
        description="Maximum steps allowed in this episode"
    )
    step_time:   Optional[str] = Field(
        default=None,
        description="Current timeline time-point label (deteriorating_patient only, e.g. 'T=0 (admission)')"
    )
    agent_action: Optional[str] = Field(
        default=None,
        description="The action the agent took at this step (deteriorating_patient only)"
    )

    model_config = ConfigDict(extra="allow")   # permit future fields without breaking clients

    def __contains__(self, key: str) -> bool:
        """Support 'key in info' checks — covers defined fields and extra fields."""
        return key in self.model_dump(exclude_none=False)


class StepResult(BaseModel):
    """Response from POST /step and POST /reset"""
    observation: TriageObservation
    reward: float = Field(ge=0.0, le=1.0, default=0.0)
    done: bool = False
    info: StepInfo = Field(default_factory=StepInfo)
