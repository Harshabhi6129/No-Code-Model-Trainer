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

# ── Shared type aliases ────────────────────────────────────────────────────

TaskType = Literal[
    "text_classification",
    "token_classification",
    "text_generation",
    "llm_finetune",
    "embedding",
    "image_classification",
    "audio",
]

TrainingApproach = Literal["full_finetune", "lora", "qlora", "embed_classify"]


# ── 1. Intent ──────────────────────────────────────────────────────────────

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
    # Populated by CleanAgent via cleanlab Confident Learning (Phase B2)
    label_noise_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    label_noise_count: int = Field(default=0, ge=0)   # absolute row count
    issues: list[str] = Field(default_factory=list)


# ── 3. Clean result ────────────────────────────────────────────────────────

class CleanResult(BaseModel):
    original_rows: int
    cleaned_rows: int
    removed_nulls: int = 0
    removed_dups: int = 0
    cleaned_path: str
    status: Literal["cleaned", "skipped"] = "cleaned"


# ── 4a. HPO search space ───────────────────────────────────────────────────
# Emitted by ModelAgent for large datasets (≥200 rows) instead of a fixed recipe.
# TrainAgent runs Optuna/TPE within this space, then converts to a ModelRecipe.

class HPOConfig(BaseModel):
    """The search space Claude defines; Optuna/TPE explores within it."""
    lr_min:           float = Field(ge=1e-7, le=1e-2)
    lr_max:           float = Field(ge=1e-7, le=1e-2)
    lora_r_choices:   list[int] = Field(default_factory=lambda: [8, 16, 32])
    n_trials:         int = Field(default=5, ge=2, le=10)
    epochs_per_trial: int = Field(default=1, ge=1, le=3)

    @model_validator(mode="after")
    def lr_range_valid(self) -> HPOConfig:
        if self.lr_min >= self.lr_max:
            # Swap rather than fail — LLM occasionally inverts min/max
            self.lr_min, self.lr_max = self.lr_max, self.lr_min
        return self


class HPOSearchSpace(BaseModel):
    """
    Emitted by ModelAgent when HPO is warranted (dataset ≥ 200 rows,
    no user overrides, approach is gradient-based).

    Fixed fields (batch_size, max_length, epochs) are determined by the
    LLM from data priors. The hpo_config is what Optuna searches.
    """
    base_model:        str
    training_approach: TrainingApproach = "lora"
    hpo_config:        HPOConfig
    batch_size:        int   = Field(default=16,  ge=1,   le=256)
    max_length:        int   = Field(default=128, ge=16,  le=4096)
    warmup_ratio:      float = Field(default=0.1, ge=0.0, le=0.5)
    weight_decay:      float = Field(default=0.01, ge=0.0, le=0.5)
    num_epochs:        int   = Field(default=3,   ge=1,   le=20)
    reasoning:         str   = ""

    @field_validator("base_model", mode="before")
    @classmethod
    def non_empty(cls, v: Any) -> Any:
        if not v or not str(v).strip():
            raise ValueError("base_model cannot be empty")
        return str(v).strip()

    def to_recipe(self, best_lr: float, best_lora_r: int | None) -> ModelRecipe:
        """Convert search space + Optuna winner into a final ModelRecipe."""
        is_lora    = self.training_approach in ("lora", "qlora")
        lora_r     = best_lora_r if is_lora else None
        lora_alpha = lora_r * 2  if lora_r  else None
        return ModelRecipe(
            base_model=self.base_model,
            training_approach=self.training_approach,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=best_lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            max_length=self.max_length,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            reasoning=self.reasoning,
        )


# ── 4b. Model recipe ───────────────────────────────────────────────────────

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
    # Populated after HPO search; None when HPO was not run
    hpo_best: dict[str, Any] | None = None

    @model_validator(mode="after")
    def fill_lora_defaults(self) -> ModelRecipe:
        """Auto-fill LoRA fields rather than hard-failing — maximises pipeline completion."""
        if self.training_approach in ("lora", "qlora"):
            if self.lora_r is None:
                self.lora_r = 16
            if self.lora_alpha is None:
                self.lora_alpha = self.lora_r * 2
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
        if not (1e-7 <= v <= 1e-2):
            raise ValueError(
                f"learning_rate {v:.2e} outside safe range [1e-7, 1e-2]"
            )
        return v

    @field_validator("batch_size")
    @classmethod
    def power_of_two_batch(cls, v: int) -> int:
        valid   = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        return v if v in valid else min(valid, key=lambda x: abs(x - v))


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
        return v if (v and str(v).strip()) else "Evaluation completed."

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
