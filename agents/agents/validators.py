"""
Semantic field validators for ModelForge model recipes.

Pydantic schemas guarantee shape-correctness (required fields, value ranges).
This module adds the next layer: semantic correctness — values that are
individually in-range but are logically wrong together.

Design rules:
  - Zero LLM calls — all checks are deterministic
  - ERRORS block training (incompatible config → broken run)
  - WARNINGS surface to user (suboptimal config → poor results)
  - Unknown model families → skip model-specific checks (log, don't error)
  - None / missing recipe → return valid immediately (training may be skipped)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Model family detection ────────────────────────────────────────────────────
# Encoder-only families: small, don't need QLoRA (designed for large decoder LLMs).
# These also have a hard 512-token positional embedding limit.
_ENCODER_FAMILIES: frozenset[str] = frozenset({
    "bert", "distilbert", "roberta", "deberta", "electra", "albert",
})
# Extended-context models that legitimately support max_length > 512
_LONG_CONTEXT_MODELS: frozenset[str] = frozenset({
    "longformer", "bigbird", "reformer", "led", "bigbird-roberta",
})

# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.warning("SemanticValidator ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.info("SemanticValidator WARNING: %s", msg)


# ── Public entry point ────────────────────────────────────────────────────────

def validate_recipe_semantics(
    recipe: dict,
    data_profile: dict,
) -> ValidationResult:
    """
    Validate a model recipe dict against the data profile it will train on.

    Args:
        recipe:       model_recipe dict from ModelAgent (must be non-None)
        data_profile: data_profile dict from DataAgent

    Returns:
        ValidationResult — callers check is_valid; warnings are advisory only.
    """
    result = ValidationResult()

    if not recipe:
        return result  # training is being skipped; nothing to validate

    model_id          = str(recipe.get("base_model", "")).lower()
    training_approach = str(recipe.get("training_approach", "")).lower()
    learning_rate     = recipe.get("learning_rate")
    lora_r            = recipe.get("lora_r")
    lora_alpha        = recipe.get("lora_alpha")
    batch_size        = recipe.get("batch_size")
    num_epochs        = recipe.get("num_epochs")
    max_length        = recipe.get("max_length")

    num_rows          = int(data_profile.get("num_rows", 0))
    label_dist        = data_profile.get("label_distribution", {})
    num_labels_profile = len(label_dist)
    num_labels_recipe  = int(data_profile.get("num_classes") or num_labels_profile or 0)

    family = _detect_family(model_id)

    _check_qlora_encoder(training_approach, family, model_id, result)
    _check_learning_rate(learning_rate, result)
    _check_lora_params(lora_r, lora_alpha, training_approach, result)
    _check_max_length(max_length, model_id, family, result)
    _check_batch_vs_dataset(batch_size, num_rows, result)
    _check_epochs_vs_dataset(num_epochs, num_rows, result)
    _check_label_count(num_labels_recipe, num_labels_profile, result)

    return result


# ── Individual checks ─────────────────────────────────────────────────────────

def _detect_family(model_id: str) -> str | None:
    """Return the matched encoder family slug, or None if unknown."""
    for family in _ENCODER_FAMILIES:
        if family in model_id:
            return family
    return None


def _check_qlora_encoder(
    approach: str, family: str | None, model_id: str, result: ValidationResult
) -> None:
    """QLoRA requires 4-bit BitsAndBytes quantization — only beneficial for large LLMs.
    Encoder-only models (BERT family) are small enough that LoRA is the right choice;
    QLoRA on them wastes memory and can cause training instability."""
    if approach != "qlora":
        return
    if family is None:
        # Unknown model — can't rule it out; skip check
        logger.debug("SemanticValidator: unknown model family for QLoRA check (%s)", model_id)
        return
    result.add_error(
        f"QLoRA is designed for large decoder-only LLMs (LLaMA, Mistral, Qwen). "
        f"`{model_id}` is an encoder-only model — use `lora` instead. "
        f"QLoRA on encoder models typically causes training instability."
    )


def _check_learning_rate(lr: object, result: ValidationResult) -> None:
    if lr is None:
        return
    try:
        lr = float(lr)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        result.add_error(f"learning_rate must be a number, got: {lr!r}")
        return

    # Pydantic already enforces [1e-7, 1e-2]; these catch values that slip through
    # a raw dict (HPO path or user override) bypassing Pydantic.
    if lr >= 1e-3:
        result.add_error(
            f"learning_rate={lr:.2e} is dangerously high. "
            f"Values at or above 1e-3 almost always cause training divergence. "
            f"Typical ranges: 2e-5 – 5e-5 (full fine-tune), 1e-4 – 3e-4 (LoRA)."
        )
    elif lr < 1e-6:
        result.add_warning(
            f"learning_rate={lr:.2e} may be too low — the model may barely learn. "
            f"Consider starting at 2e-5 (full fine-tune) or 2e-4 (LoRA)."
        )


def _check_lora_params(
    lora_r: object, lora_alpha: object, approach: str, result: ValidationResult
) -> None:
    if approach not in ("lora", "qlora"):
        return  # full_finetune / embed_classify don't use LoRA params

    if lora_r is not None:
        try:
            r = int(lora_r)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            result.add_error(f"lora_r must be an integer, got: {lora_r!r}")
            return
        if r > 64:
            result.add_warning(
                f"lora_r={r} gives diminishing quality gains over r=32 "
                f"and significantly increases VRAM usage. Consider r=16 or r=32."
            )

    if lora_r is not None and lora_alpha is not None:
        try:
            r = int(lora_r)       # type: ignore[arg-type]
            a = int(lora_alpha)   # type: ignore[arg-type]
            if a < r:
                result.add_warning(
                    f"lora_alpha={a} < lora_r={r}: effective scaling factor (α/r) is "
                    f"{a/r:.2f} < 1.0. Standard practice is alpha = r or alpha = 2×r."
                )
        except (TypeError, ValueError):
            pass  # already handled above


def _check_max_length(
    max_length: object, model_id: str, family: str | None, result: ValidationResult
) -> None:
    if max_length is None:
        return
    try:
        ml = int(max_length)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        result.add_error(f"max_length must be an integer, got: {max_length!r}")
        return

    # BERT-family hard limit: positional embeddings cap out at 512
    if ml > 512 and family in _ENCODER_FAMILIES:
        # Allow long-context variants that override the BERT architecture
        is_long_ctx = any(lc in model_id for lc in _LONG_CONTEXT_MODELS)
        if not is_long_ctx:
            result.add_error(
                f"max_length={ml} exceeds the 512-token limit of `{model_id}`. "
                f"BERT-family models have fixed positional embeddings up to 512. "
                f"Set max_length ≤ 512 or use a long-context model (e.g. Longformer)."
            )


def _check_batch_vs_dataset(
    batch_size: object, num_rows: int, result: ValidationResult
) -> None:
    if batch_size is None or num_rows <= 0:
        return
    try:
        bs = int(batch_size)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    threshold = num_rows / 4
    if bs > threshold:
        result.add_warning(
            f"batch_size={bs} is larger than 25% of the dataset ({num_rows} rows). "
            f"With such large batches, gradient updates will be rare and learning may stall. "
            f"Consider batch_size ≤ {max(1, int(threshold))}."
        )


def _check_epochs_vs_dataset(
    num_epochs: object, num_rows: int, result: ValidationResult
) -> None:
    if num_epochs is None or num_rows <= 0:
        return
    try:
        ne = int(num_epochs)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if ne > 10 and num_rows < 500:
        result.add_warning(
            f"num_epochs={ne} with only {num_rows} training rows is a high overfitting risk. "
            f"Consider num_epochs ≤ 5 and monitor eval_loss carefully."
        )


def _check_label_count(
    num_labels_recipe: int, num_labels_profile: int, result: ValidationResult
) -> None:
    """If the recipe's label count disagrees with the dataset's class count, the
    model head will have the wrong output dimension → training error at first batch."""
    if num_labels_recipe <= 0 or num_labels_profile <= 0:
        return  # one side is unknown — can't compare
    if num_labels_recipe != num_labels_profile:
        result.add_error(
            f"Label count mismatch: recipe expects {num_labels_recipe} classes "
            f"but the dataset has {num_labels_profile} distinct labels. "
            f"The model's output head will have the wrong dimension and training will fail."
        )
