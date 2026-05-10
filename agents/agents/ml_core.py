"""
ML training core — self-contained engine with lazy imports.
torch/transformers/sklearn are only imported inside _blocking_train() so the
agent package stays importable even without GPU libs installed.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Resolved from env so the path is configurable in prod (e.g. ephemeral storage).
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(Path(__file__).parent.parent.parent / "backend" / "runs")))

SUPPORTED_TASK_TYPES = {"text_classification"}

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def has_training_libs() -> bool:
    """True when torch + transformers + sklearn + datasets are all importable."""
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        import sklearn        # noqa: F401
        import datasets       # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Pre-flight validation (pure Python — no ML libs needed)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def validate_training_inputs(
    *,
    dataset_path: str | None,
    task_type: str,
    text_col: str,
    label_col: str,
    model_id: str,
) -> ValidationResult:
    """
    Comprehensive pre-flight checks before touching any GPU libs.
    Returns ValidationResult.ok=False + .error for the first hard blocker.
    Soft issues are collected into .warnings.
    """
    import pandas as pd

    warnings: list[str] = []

    # ── 1. Dataset path ───────────────────────────────────────────────────────
    if not dataset_path:
        return ValidationResult(ok=False, error="No dataset uploaded. Please upload a CSV or JSON file.")

    path = Path(dataset_path)
    if not path.exists():
        return ValidationResult(ok=False, error="Dataset file not found — please re-upload your file.")
    if path.stat().st_size == 0:
        return ValidationResult(ok=False, error="The uploaded file is empty.")

    # ── 2. Task type ──────────────────────────────────────────────────────────
    if task_type not in SUPPORTED_TASK_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TASK_TYPES))
        return ValidationResult(
            ok=False,
            error=(
                f"Task type '{task_type}' is not yet supported in v0. "
                f"Supported: {supported}. More task types are on the roadmap!"
            ),
        )

    # ── 3. Model ID sanity ────────────────────────────────────────────────────
    if not model_id or not model_id.strip():
        return ValidationResult(ok=False, error="No base model ID was selected.")
    # Allow "org/repo" or "model-name" but block path traversal
    if ".." in model_id or model_id.startswith("/"):
        return ValidationResult(ok=False, error=f"Invalid model ID: '{model_id}'.")
    if not re.match(r"^[\w\-\.]+(/[\w\-\.]+)?$", model_id.strip()):
        warnings.append(
            f"Model ID '{model_id}' looks unusual — verify it exists on HuggingFace Hub."
        )

    # ── 4. Parse dataset ──────────────────────────────────────────────────────
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            return ValidationResult(ok=False, error=f"Unsupported file format: {suffix}.")
    except Exception as exc:
        return ValidationResult(ok=False, error=f"Could not read dataset: {exc}")

    if df.empty:
        return ValidationResult(ok=False, error="Dataset is empty after parsing.")

    # ── 5. Column existence ───────────────────────────────────────────────────
    missing: list[str] = []
    if text_col not in df.columns:
        missing.append(f"text column '{text_col}'")
    if label_col not in df.columns:
        missing.append(f"label column '{label_col}'")
    if missing:
        available = ", ".join(f"'{c}'" for c in df.columns)
        return ValidationResult(
            ok=False,
            error=f"Missing {' and '.join(missing)}. Available columns: {available}.",
        )

    # ── 6. Valid row count ────────────────────────────────────────────────────
    df_clean = df[[text_col, label_col]].dropna()
    df_clean = df_clean[df_clean[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
    n = len(df_clean)

    if n < 10:
        return ValidationResult(
            ok=False,
            error=f"Dataset has only {n} valid samples after removing empty/null rows. Need at least 10.",
        )
    if n < 50:
        warnings.append(
            f"Very small dataset ({n} samples) — metrics will be noisy and the model may not generalise well."
        )
    elif n < 200:
        warnings.append(f"Small dataset ({n} samples) — consider collecting more data for better accuracy.")

    # ── 7. Label checks ───────────────────────────────────────────────────────
    labels = df_clean[label_col].astype(str)
    label_counts = labels.value_counts()
    unique_labels = len(label_counts)

    if unique_labels < 2:
        return ValidationResult(
            ok=False,
            error=(
                f"Dataset has only 1 class ('{label_counts.index[0]}'). "
                "Need at least 2 distinct labels to train a classifier."
            ),
        )
    if unique_labels > 50:
        warnings.append(
            f"High number of classes ({unique_labels}). "
            "Consider grouping related labels; very fine-grained classification usually needs more data."
        )

    too_few = [f"'{lbl}' ({cnt})" for lbl, cnt in label_counts.items() if cnt < 2]
    if too_few:
        return ValidationResult(
            ok=False,
            error=(
                f"Classes with fewer than 2 samples: {', '.join(too_few)}. "
                "Add more examples or remove those classes."
            ),
        )

    per_class_warnings = [f"'{lbl}' ({cnt})" for lbl, cnt in label_counts.items() if cnt < 5]
    if per_class_warnings:
        warnings.append(
            f"Classes with fewer than 5 samples: {', '.join(per_class_warnings)}. "
            "These classes may have very poor recall."
        )

    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio > 10:
        warnings.append(
            f"Severe class imbalance ({imbalance_ratio:.0f}:1 ratio). "
            "The model may learn to always predict the majority class. "
            "Weighted F1 will be reported — accuracy may be misleading."
        )
    elif imbalance_ratio > 5:
        warnings.append(
            f"Moderate class imbalance ({imbalance_ratio:.1f}:1). "
            "F1 score is more informative than accuracy for your use case."
        )

    # ── 8. Text quality ───────────────────────────────────────────────────────
    avg_len = df_clean[text_col].astype(str).str.len().mean()
    if avg_len < 5:
        warnings.append(
            f"Average text length is very short ({avg_len:.0f} chars). "
            "Very short texts may not provide enough signal for the model."
        )
    elif avg_len > 1000:
        warnings.append(
            f"Average text length is long ({avg_len:.0f} chars). "
            "Texts will be truncated to max_length tokens — important signal near the end may be lost."
        )

    # ── 9. Disk space ─────────────────────────────────────────────────────────
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        free_mb = shutil.disk_usage(RUNS_DIR).free / (1024 ** 2)
        if free_mb < 300:
            return ValidationResult(
                ok=False,
                error=f"Insufficient disk space ({free_mb:.0f} MB free). Need at least 300 MB for model checkpoints.",
            )
        if free_mb < 1000:
            warnings.append(f"Low disk space ({free_mb:.0f} MB free). Large models may fail to save.")
    except Exception:
        pass

    return ValidationResult(ok=True, warnings=warnings)


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    model_path: str
    base_model: str
    training_approach: str
    num_epochs_completed: float
    final_train_loss: float | None
    training_time_seconds: float
    device: str
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    epoch_metrics: list[dict] = field(default_factory=list)


class TrainingDivergedError(RuntimeError):
    """Loss became NaN or Inf — training cannot continue."""


def _deduplicate_epoch_metrics(raw: list[dict]) -> list[dict]:
    """
    Collapse a raw per-log-step list into one entry per integer epoch.
    When multiple entries share an epoch, values are merged: the latest
    non-None value for each key wins. This ensures both train_loss (logged
    mid-epoch) and eval_loss (logged at epoch end) appear in the same entry.
    """
    epoch_map: dict[int, dict] = {}
    for entry in raw:
        epoch = entry["epoch"]
        if epoch not in epoch_map:
            epoch_map[epoch] = dict(entry)
        else:
            existing = epoch_map[epoch]
            epoch_map[epoch] = {
                "epoch": epoch,
                "step":  entry["step"],
                "loss":          entry["loss"]          if entry["loss"]          is not None else existing["loss"],
                "eval_loss":     entry["eval_loss"]     if entry["eval_loss"]     is not None else existing["eval_loss"],
                "learning_rate": entry["learning_rate"] if entry["learning_rate"] is not None else existing["learning_rate"],
            }
    return list(epoch_map.values())


# ---------------------------------------------------------------------------
# Blocking training (runs in a thread pool)
# ---------------------------------------------------------------------------

def _has_peft() -> bool:
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def _blocking_train(
    *,
    job_id: str,
    model_id: str,
    dataset_path: str,
    text_col: str,
    label_col: str,
    training_approach: str,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    weight_decay: float,
    warmup_ratio: float,
    lora_r: int = 8,
    use_cpu: bool,
    progress_log: list | None = None,
    progress_lock: "threading.Lock | None" = None,
) -> TrainingResult:
    """
    Full HuggingFace training + evaluation pipeline.
    Must be called via asyncio.to_thread() — it blocks for minutes.
    """
    # Late imports so the module is importable without ML libs.
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, classification_report,
    )
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        EarlyStoppingCallback,
    )
    from datasets import Dataset as HFDataset
    import pandas as pd

    t0 = time.time()
    accumulated_warnings: list[str] = []

    # ── Device ────────────────────────────────────────────────────────────────
    if use_cpu:
        device_str = "cpu"
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    logger.info("[%s] Device: %s", job_id, device_str)

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = RUNS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & clean ──────────────────────────────────────────────────────────
    suffix = Path(dataset_path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(dataset_path)
    elif suffix == ".jsonl":
        df = pd.read_json(dataset_path, lines=True)
    else:
        df = pd.read_json(dataset_path)

    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    df["_int_label"] = le.fit_transform(df[label_col].astype(str))
    num_labels = len(le.classes_)
    label_names: list[str] = le.classes_.tolist()

    counts = df["_int_label"].value_counts()
    if counts.max() / counts.min() > 5:
        accumulated_warnings.append(
            f"Class imbalance ({counts.max()/counts.min():.0f}:1) — reporting weighted F1."
        )

    # ── Stratified train/test split ───────────────────────────────────────────
    n = len(df)
    # Dynamic test size: 20% but at least 10 and at most 30% of dataset
    test_size = min(0.30, max(0.15, 10 / n))
    min_class_count = counts.min()

    try:
        if min_class_count >= 2:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=42, stratify=df["_int_label"]
            )
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            accumulated_warnings.append("Could not stratify split — some classes have < 2 samples.")
    except ValueError:
        # Fallback for edge cases (e.g., too few samples for stratify)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Clamp batch_size to training set size
    if batch_size > len(train_df):
        accumulated_warnings.append(
            f"batch_size reduced from {batch_size} to {len(train_df)} (equals training set size)."
        )
        batch_size = max(1, len(train_df))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("[%s] Loading tokenizer: %s", job_id, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,      # dynamic padding via DataCollatorWithPadding
            max_length=max_length,
        )

    def to_hf_dataset(split: "pd.DataFrame") -> "HFDataset":
        return (
            HFDataset.from_dict({
                "text": split[text_col].tolist(),
                "label": split["_int_label"].tolist(),
            })
            .map(tokenize, batched=True, remove_columns=["text"])
        )

    train_ds = to_hf_dataset(train_df)
    test_ds  = to_hf_dataset(test_df)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("[%s] Loading model %s (%d labels)", job_id, model_id, num_labels)

    is_lora  = training_approach in ("lora", "qlora")
    is_qlora = training_approach == "qlora"

    load_kwargs: dict[str, Any] = {
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }

    # QLoRA: 4-bit quantisation — only possible on CUDA with bitsandbytes
    if is_qlora and device_str == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        except Exception as exc:
            accumulated_warnings.append(
                f"QLoRA quantisation unavailable ({exc}) — falling back to regular LoRA."
            )
            is_qlora = False

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **load_kwargs)

    # LoRA / QLoRA: wrap model with PEFT adapter
    if is_lora:
        if _has_peft():
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
            if is_qlora and device_str == "cuda":
                model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            logger.info("[%s] LoRA: trainable=%.2f%% (%d/%d params)", job_id,
                        100 * trainable / max(total, 1), trainable, total)
        else:
            accumulated_warnings.append(
                "peft package not installed — falling back to full fine-tuning. "
                "Install: pip install peft>=0.7.0"
            )
            is_lora = False

    # ── Divergence callback ───────────────────────────────────────────────────
    class DivergenceCallback(TrainerCallback):
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kw: Any,
        ) -> None:
            if not logs:
                return
            loss = logs.get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                control.should_training_stop = True
                raise TrainingDivergedError(
                    f"Loss became {loss} at step {state.global_step}. "
                    "Try a smaller learning rate (e.g. 1e-5 or 2e-6)."
                )
            if loss is not None and loss > 100:
                accumulated_warnings.append(
                    f"Very high loss ({loss:.1f}) at step {state.global_step} — training may be unstable."
                )

    # ── Epoch progress callback (real-time loss streaming) ────────────────────
    class EpochProgressCallback(TrainerCallback):
        """
        Captures per-log-step metrics into a shared thread-safe list so the
        async keepalive loop can stream them to the SSE client in real-time.
        Only attached when progress_log/progress_lock are provided.
        """
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kw: Any,
        ) -> None:
            if not logs or progress_log is None or progress_lock is None:
                return
            raw_loss      = logs.get("loss")
            raw_eval_loss = logs.get("eval_loss")
            raw_lr        = logs.get("learning_rate")
            # Skip steps that carry neither loss value (e.g. pure eval_accuracy lines)
            if raw_loss is None and raw_eval_loss is None:
                return

            def _safe(v: float | None) -> float | None:
                if v is None:
                    return None
                return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)

            entry = {
                "epoch":         int(math.floor(state.epoch or 0)),
                "step":          state.global_step,
                "loss":          _safe(raw_loss),
                "eval_loss":     _safe(raw_eval_loss),
                "learning_rate": _safe(raw_lr),
            }
            with progress_lock:
                progress_log.append(entry)

    # ── Mixed precision & device flags ────────────────────────────────────────
    use_fp16 = False
    use_bf16 = False
    if device_str == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            use_bf16 = True    # Ampere+ (A100, RTX 3000+)
        elif cap[0] >= 7:
            use_fp16 = True    # Volta / Turing (V100, RTX 2000)

    no_cuda = (device_str != "cuda")
    log_steps = max(1, len(train_ds) // max(1, batch_size * 4))

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size * 2),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=log_steps,
        save_total_limit=1,
        report_to="none",
        no_cuda=no_cuda,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=(device_str == "cuda"),
        seed=42,
        data_seed=42,
        disable_tqdm=True,
    )

    epoch_cbs = [EpochProgressCallback()] if (progress_log is not None and progress_lock is not None) else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[
            DivergenceCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
            *epoch_cbs,
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        "[%s] Training: %d samples, %d epochs, lr=%.2e, batch=%d",
        job_id, len(train_ds), num_epochs, learning_rate, batch_size,
    )
    try:
        train_output = trainer.train()
        final_loss: float | None = round(float(train_output.training_loss), 4)
    except TrainingDivergedError:
        raise
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            # OOM: reduce batch size and retry once
            reduced = max(1, batch_size // 2)
            accumulated_warnings.append(
                f"GPU OOM at batch_size={batch_size} — retrying with batch_size={reduced}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Rebuild trainer with smaller batch
            training_args_retry = TrainingArguments(
                output_dir=str(output_dir / "checkpoints"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=reduced,
                per_device_eval_batch_size=max(1, reduced * 2),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type="cosine",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_steps=log_steps,
                save_total_limit=1,
                report_to="none",
                no_cuda=no_cuda,
                fp16=use_fp16,
                bf16=use_bf16,
                seed=42,
                data_seed=42,
                disable_tqdm=True,
            )
            trainer_retry = Trainer(
                model=model,
                args=training_args_retry,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                data_collator=collator,
                callbacks=[DivergenceCallback(), *epoch_cbs],
            )
            train_output = trainer_retry.train()
            trainer = trainer_retry
            final_loss = round(float(train_output.training_loss), 4)
        else:
            raise

    epochs_done = float(trainer.state.epoch or num_epochs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("[%s] Evaluating on %d test samples", job_id, len(test_ds))
    pred_out = trainer.predict(test_ds)
    y_pred: list[int] = pred_out.predictions.argmax(axis=-1).tolist()
    y_true: list[int] = test_ds["label"]

    acc       = float(accuracy_score(y_true, y_pred))
    f1        = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    per_class_f1 = {
        lbl: round(float(report[lbl]["f1-score"]), 4)
        for lbl in label_names
        if lbl in report
    }

    metrics: dict[str, Any] = {
        "accuracy":    round(acc, 4),
        "f1":          round(f1, 4),
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "per_class_f1": per_class_f1,
        "num_labels":  num_labels,
        "label_names": label_names,
        "train_samples": len(train_ds),
        "eval_samples":  len(test_ds),
    }

    # ── Save model + tokenizer ────────────────────────────────────────────────
    final_model_path = output_dir / "final_model"
    if is_lora:
        # Save PEFT adapter separately; inference_cache must merge on load
        model.save_pretrained(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
    else:
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

    # Remove checkpoints to reclaim disk space
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    elapsed = time.time() - t0
    logger.info(
        "[%s] Done in %.1fs — acc=%.3f f1=%.3f path=%s",
        job_id, elapsed, acc, f1, final_model_path,
    )

    final_epoch_metrics: list[dict] = []
    if progress_log is not None and progress_lock is not None:
        with progress_lock:
            final_epoch_metrics = _deduplicate_epoch_metrics(list(progress_log))

    return TrainingResult(
        model_path=str(final_model_path),
        base_model=model_id,
        training_approach=training_approach,
        num_epochs_completed=round(epochs_done, 1),
        final_train_loss=final_loss,
        training_time_seconds=round(elapsed, 1),
        device=device_str,
        metrics=metrics,
        warnings=accumulated_warnings,
        epoch_metrics=final_epoch_metrics,
    )


# ---------------------------------------------------------------------------
# Async wrapper (called from TrainAgent)
# ---------------------------------------------------------------------------

async def train_model_async(
    *,
    job_id: str,
    model_id: str,
    dataset_path: str,
    text_col: str,
    label_col: str,
    task_type: str = "text_classification",
    training_approach: str = "full_finetune",
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lora_r: int = 8,
    use_cpu: bool = False,
    progress_log: list | None = None,
    progress_lock: "threading.Lock | None" = None,
) -> TrainingResult:
    """Non-blocking wrapper — runs the blocking trainer in a thread pool."""
    return await asyncio.to_thread(
        _blocking_train,
        job_id=job_id,
        model_id=model_id,
        dataset_path=dataset_path,
        text_col=text_col,
        label_col=label_col,
        training_approach=training_approach,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lora_r=lora_r,
        use_cpu=use_cpu,
        progress_log=progress_log,
        progress_lock=progress_lock,
    )
