"""
Pydantic contracts for every agent hand-off in the ModelForge pipeline.

Design rules (from the research report):
  - Schema-valid ≠ semantically correct — every model adds semantic field_validators
  - Validators CORRECT where safe (e.g. missing lora_r → default 16) rather than
    always raising, to maximise pipeline completion rate
  - All models serialise back to plain dicts via .model_dump() so AgentContext
    (which stores dicts) needs zero changes
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── 1. Intent ──────────────────────────────────────────────────────────────

TaskType = Literal[
    "text_classification",
    "token_classification",
    "text_generation",
    "llm_finetune",
    "embedding",
    "image_classification",
    "audio",
]


class TaskSpec(BaseModel):
    task_type: TaskType
    num_labels: int | None = None
    label_names: list[str] | None = None
    input_column: str = "text"
    label_column: str = "label"
    base_model_hint: str = "distilbert-base-uncased"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    clarification_needed: str | None = None

    @field_validator("input_column", "label_column", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("base_model_hint", mode="before")
    @classmethod
    def non_empty_hint(cls, v: Any) -> Any:
        if not v or not str(v).strip():
            return "distilbert-base-uncased"
        return v


# ── 2. Data profile ────────────────────────────────────────────────────────

class DataProfile(BaseModel):
    num_rows: int
    num_cols: int
    columns: list[str] = Field(default_factory=list)
    input_col: str
    label_col: str | None = None
    avg_input_len: float = 0.0
    max_input_len: int = 0
    num_classes: int | None = None
    label_distribution: dict[str, int] = Field(default_factory=dict)
    # min_class_count / max_class_count — 1.0 = perfectly balanced
    class_balance_ratio: float | None = None
    duplicate_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Estimated fraction of mislabelled rows (populated by CleanAgent / cleanlab)
    label_noise_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


# ── 3. Clean result ────────────────────────────────────────────────────────

class CleanResult(BaseModel):
    original_rows: int
    cleaned_rows: int
    removed_nulls: int = 0
    removed_dups: int = 0
    cleaned_path: str
    status: Literal["cleaned", "skipped"] = "cleaned"


# ── 4. Model recipe ────────────────────────────────────────────────────────

TrainingApproach = Literal["full_finetune", "lora", "qlora", "embed_classify"]


class ModelRecipe(BaseModel):
    base_model: str
    training_approach: TrainingApproach
    lora_r: int | None = Field(None, ge=4, le=128)
    lora_alpha: int | None = Field(None, ge=1, le=512)
    lora_target_modules: list[str] | None = None
    learning_rate: float = Field(ge=1e-7, le=1e-2)
    num_epochs: int = Field(ge=1, le=20)
    batch_size: int = Field(ge=1, le=256)
    max_length: int = Field(default=128, ge=16, le=4096)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5)
    reasoning: str = ""

    @model_validator(mode="after")
    def fill_lora_defaults(self) -> "ModelRecipe":
        """Auto-fill LoRA fields rather than hard-failing — maximises pipeline completion."""
        if self.training_approach in ("lora", "qlora"):
            if self.lora_r is None:
                self.lora_r = 16  # research-backed sweet spot
            if self.lora_alpha is None:
                self.lora_alpha = self.lora_r * 2  # standard: alpha = 2r
        return self

    @field_validator("base_model", mode="before")
    @classmethod
    def non_empty_model(cls, v: Any) -> Any:
        if not v or not str(v).strip():
            raise ValueError("base_model cannot be empty")
        return str(v).strip()

    @field_validator("learning_rate")
    @classmethod
    def sane_lr(cls, v: float) -> float:
        # Values outside this range are almost certainly hallucinations
        if not (1e-7 <= v <= 1e-2):
            raise ValueError(
                f"learning_rate {v:.2e} is outside safe range [1e-7, 1e-2] — "
                "this is likely a hallucination"
            )
        return v

    @field_validator("batch_size")
    @classmethod
    def power_of_two_batch(cls, v: int) -> int:
        # Clamp to nearest valid value rather than reject
        valid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        if v not in valid:
            # Round to nearest power-of-two
            closest = min(valid, key=lambda x: abs(x - v))
            return closest
        return v


# ── 5. Eval report ─────────────────────────────────────────────────────────

EvalGrade = Literal["A", "B", "C", "D", "F"]


class EvalReport(BaseModel):
    evaluation_grade: EvalGrade
    summary: str
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def non_empty_summary(cls, v: Any) -> Any:
        if not v or not str(v).strip():
            return "Evaluation completed."
        return v

    @field_validator("strengths", "concerns", "next_steps", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list:
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []


# ── 6. Deploy result ───────────────────────────────────────────────────────

class DeployResult(BaseModel):
    status: Literal["deployed", "skipped", "failed"]
    repo_id: str | None = None
    model_card: str | None = None
    python_snippet: str | None = None
    pipeline_snippet: str | None = None
    reason: str = ""
