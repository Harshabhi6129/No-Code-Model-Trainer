"""
Train Agent — orchestrates the full training lifecycle:
  1. Pre-flight validation (always runs, no GPU libs needed)
  2. Graceful degradation if torch/transformers not installed
  3. Actual training via ml_core.train_model_async()
  4. Keepalive SSE events every 20 s so the stream doesn't time out
  5. Stores structured results in context.training_result
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncIterator, Any

from .base import BaseAgent, AgentContext, AgentResult
from .ml_core import (
    has_training_libs,
    validate_training_inputs,
    train_model_async,
    TrainingDivergedError,
)

logger = logging.getLogger(__name__)

_KEEPALIVE_INTERVAL = 20   # seconds between keepalive SSE events
_TRAINING_TIMEOUT   = 3600 # hard cap: 60 minutes


class TrainAgent(BaseAgent):
    name = "Train"

    # ------------------------------------------------------------------
    # run() — single-result interface (used by tests and direct callers)
    # ------------------------------------------------------------------

    async def run(self, context: AgentContext) -> AgentResult:
        result: AgentResult | None = None
        async for result in self.run_stream(context):
            pass
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # run_stream() — multi-result interface used by the pipeline.
    # Yields:
    #   • One "starting" result immediately (so the UI shows a card)
    #   • Keepalive results every 20 s while training runs
    #   • One final result with metrics (or error)
    # ------------------------------------------------------------------

    async def run_stream(self, context: AgentContext) -> AsyncIterator[AgentResult]:
        spec    = context.task_spec
        profile = context.data_profile
        recipe  = context.model_recipe

        ovr = context.hyperparameter_overrides  # user overrides win over agent recipe

        task_type         = str(spec.get("task_type", "text_classification"))
        text_col          = str(spec.get("input_column") or profile.get("input_col") or "text")
        label_col         = str(spec.get("label_column") or profile.get("label_col") or "label")
        model_id          = str(ovr.get("model_id") or recipe.get("base_model") or spec.get("base_model_hint", "distilbert-base-uncased"))
        training_approach = str(ovr.get("training_approach") or recipe.get("training_approach", "full_finetune"))
        learning_rate     = float(ovr.get("learning_rate") or recipe.get("learning_rate", 2e-5))
        num_epochs        = int(ovr.get("num_epochs") or recipe.get("num_epochs", 3))
        batch_size        = int(ovr.get("batch_size") or recipe.get("batch_size", 16))
        max_length        = int(ovr.get("max_length") or recipe.get("max_length", 128))
        weight_decay      = float(ovr.get("weight_decay") or recipe.get("weight_decay", 0.01))
        warmup_ratio      = float(ovr.get("warmup_ratio") or recipe.get("warmup_ratio", 0.1))
        lora_r            = int(ovr.get("lora_r") or recipe.get("lora_r", 8))

        # ── Pre-flight validation ─────────────────────────────────────────────
        val = validate_training_inputs(
            dataset_path=context.dataset_path,
            task_type=task_type,
            text_col=text_col,
            label_col=label_col,
            model_id=model_id,
        )

        if not val.ok:
            yield AgentResult(
                agent_name=self.name,
                success=False,
                output={"validation_error": val.error},
                message=f"Cannot start training: {val.error}",
                next_agent=None,
            )
            return

        # ── Check ML libraries ────────────────────────────────────────────────
        if not has_training_libs():
            # Graceful degradation: GPU libs not installed in this environment.
            # Store the recipe so nothing is lost — the user can re-run locally.
            context.training_result = {
                "status": "skipped",
                "reason": "Training libraries (PyTorch/Transformers) are not installed in this environment.",
                "recipe": recipe,
                "validation_warnings": val.warnings,
            }
            yield AgentResult(
                agent_name=self.name,
                success=True,
                output=context.training_result,
                message=(
                    "Training libraries are not available in this deployment environment. "
                    "Your model recipe has been saved — run ModelForge locally with GPU support "
                    "to execute training. Modal/GPU cloud support is on the roadmap!"
                ),
                next_agent="Eval",
            )
            return

        # ── Announce start ────────────────────────────────────────────────────
        warning_note = ""
        if val.warnings:
            warning_note = "\n**Heads up:** " + " | ".join(val.warnings)

        yield AgentResult(
            agent_name=self.name,
            success=True,
            output={"status": "starting", "model_id": model_id, "final": False},
            message=(
                f"Training `{model_id}` on your dataset.\n"
                f"Approach: {training_approach.replace('_', ' ').upper()} · "
                f"{num_epochs} epochs · lr={learning_rate} · batch={batch_size}"
                f"{warning_note}"
            ),
            next_agent="Eval",
        )

        # ── Launch training as a background task ──────────────────────────────
        start_time = time.monotonic()
        progress_log: list[dict] = []
        progress_lock = threading.Lock()
        _emitted_steps: set[int] = set()  # track which steps we've already streamed

        training_task = asyncio.create_task(
            train_model_async(
                job_id=context.run_id,
                model_id=model_id,
                dataset_path=context.dataset_path or "",
                text_col=text_col,
                label_col=label_col,
                task_type=task_type,
                training_approach=training_approach,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_length=max_length,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lora_r=lora_r,
                use_cpu=False,
                progress_log=progress_log,
                progress_lock=progress_lock,
            )
        )

        # ── Keepalive loop ────────────────────────────────────────────────────
        while not training_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(training_task), timeout=_KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                elapsed = int(time.monotonic() - start_time)
                if elapsed >= _TRAINING_TIMEOUT:
                    training_task.cancel()
                    yield AgentResult(
                        agent_name=self.name,
                        success=False,
                        output={"status": "timeout", "final": True},
                        message=f"Training timed out after {elapsed // 60} minutes. Try fewer epochs or a smaller model.",
                        next_agent=None,
                    )
                    return

                # Drain any new epoch entries from the progress log
                with progress_lock:
                    snapshot = list(progress_log)
                new_entries = [e for e in snapshot if e["step"] not in _emitted_steps]
                for entry in new_entries:
                    _emitted_steps.add(entry["step"])
                    yield AgentResult(
                        agent_name=self.name,
                        success=True,
                        output={
                            "status": "epoch",
                            "epoch": entry["epoch"],
                            "step": entry["step"],
                            "loss": entry["loss"],
                            "eval_loss": entry["eval_loss"],
                            "learning_rate": entry["learning_rate"],
                            "final": False,
                        },
                        message=f"Epoch {entry['epoch'] + 1} — loss={entry['loss']}",
                        next_agent="Eval",
                    )

                if not new_entries:
                    yield AgentResult(
                        agent_name=self.name,
                        success=True,
                        output={"status": "training", "elapsed_seconds": elapsed, "final": False},
                        message=f"Training in progress — {elapsed}s elapsed...",
                        next_agent="Eval",
                    )
            except asyncio.CancelledError:
                break

        # ── Resolve result ────────────────────────────────────────────────────
        try:
            result = training_task.result()
        except TrainingDivergedError as exc:
            yield AgentResult(
                agent_name=self.name,
                success=False,
                output={"status": "diverged", "final": True},
                message=(
                    f"Training diverged (loss became NaN/Inf). {exc}\n"
                    "**Suggestions:** reduce learning rate by 10×, or try `distilbert-base-uncased` "
                    "if using a custom model."
                ),
                next_agent=None,
            )
            return
        except asyncio.CancelledError:
            yield AgentResult(
                agent_name=self.name,
                success=False,
                output={"status": "cancelled", "final": True},
                message="Training was cancelled.",
                next_agent=None,
            )
            return
        except MemoryError:
            yield AgentResult(
                agent_name=self.name,
                success=False,
                output={"status": "oom", "final": True},
                message=(
                    "Out of memory. Try: smaller batch size, shorter max_length, or a lighter model "
                    "(e.g. `distilbert-base-uncased` instead of `bert-large`)."
                ),
                next_agent=None,
            )
            return
        except Exception as exc:
            logger.error("Training failed for run %s: %s", context.run_id, exc, exc_info=True)
            yield AgentResult(
                agent_name=self.name,
                success=False,
                output={"status": "error", "final": True},
                message=f"Training failed: {exc}",
                next_agent=None,
            )
            return

        # ── Store result in context for EvalAgent ─────────────────────────────
        context.training_result = {
            "model_path":           result.model_path,
            "base_model":           result.base_model,
            "training_approach":    result.training_approach,
            "num_epochs_completed": result.num_epochs_completed,
            "final_train_loss":     result.final_train_loss,
            "training_time_seconds": result.training_time_seconds,
            "device":               result.device,
            "warnings":             result.warnings + val.warnings,
            "epoch_metrics":        result.epoch_metrics,
            **result.metrics,
        }

        # ── Build final success message ────────────────────────────────────────
        m = result.metrics
        acc_pct = f"{m['accuracy'] * 100:.1f}%"
        f1_val  = f"{m['f1']:.3f}"
        mins    = int(result.training_time_seconds // 60)
        secs    = int(result.training_time_seconds % 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

        warnings_note = ""
        all_warnings = result.warnings + val.warnings
        if all_warnings:
            warnings_note = "\n**Warnings:** " + " | ".join(all_warnings)

        yield AgentResult(
            agent_name=self.name,
            success=True,
            output={**context.training_result, "final": True},
            message=(
                f"Training complete in {time_str} on {result.device.upper()}.\n"
                f"**Accuracy:** {acc_pct} · **Weighted F1:** {f1_val} "
                f"({m['train_samples']} train / {m['eval_samples']} test samples)"
                f"{warnings_note}"
            ),
            next_agent="Eval",
        )
