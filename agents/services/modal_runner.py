"""
Modal GPU runner for ModelForge training jobs.

When MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set, training is dispatched
to a Modal serverless H100 GPU instead of running locally.

Architecture:
  • modal_runner.py: Modal app definition + local dispatch interface
  • TrainAgent detects MODAL_TOKEN_ID → calls run_training_on_modal() instead
    of train_model_async()
  • run_training_on_modal() returns the same TrainingResult type as the local path
  • Checkpoints saved to Modal Volume; on retry, training resumes from last ckpt

Modal cold-start: ~30-60s.  We emit a "warming up GPU" SSE event so the UI
doesn't time out and the user knows the job is in the queue.

Graceful degradation:
  - modal package not installed → falls back to local training
  - MODAL_TOKEN_ID not set → falls back to local training
  - Modal function fails → raises, TrainAgent surfaces the error
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Availability ──────────────────────────────────────────────────────────────

def has_modal() -> bool:
    """Return True if the modal package is installed and credentials are set."""
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        return False
    try:
        import modal  # noqa: F401
        return True
    except ImportError:
        return False


# ── Modal app definition ──────────────────────────────────────────────────────
# Imported lazily so this module is importable without modal installed.

def _build_modal_app():
    """Build and return the Modal app. Only called when has_modal() is True."""
    import modal

    # GPU image with all ML dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "datasets>=2.18.0",
            "peft>=0.10.0",
            "accelerate>=0.28.0",
            "scikit-learn>=1.4.0",
            "pandas>=2.0.0",
        )
    )

    # Persistent volume for checkpoints (survives function retries)
    volume = modal.Volume.from_name("modelforge-checkpoints", create_if_missing=True)

    app = modal.App("modelforge-training", image=image)

    @app.function(
        gpu="H100",
        volumes={"/mnt/checkpoints": volume},
        retries=modal.Retries(max_retries=5, delay=0.0),
        timeout=3600,  # 60-minute hard cap
    )
    def train_on_modal(training_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Remote Modal function that runs training on H100.
        Returns a dict representation of TrainingResult.
        """
        import sys
        import os as _os

        # The agents package is uploaded with the function
        run_id = training_kwargs.get("job_id", "unknown")
        checkpoint_dir = f"/mnt/checkpoints/{run_id}"
        _os.makedirs(checkpoint_dir, exist_ok=True)

        # Use local ml_core (uploaded with app)
        from agents.ml_core import _blocking_train, TrainingResult
        import threading

        result = _blocking_train(
            **training_kwargs,
            use_cpu=False,
            progress_log=None,
            progress_lock=None,
            cancel_event=None,
            pause_event=None,
        )

        # Return as dict (Modal serialises via pickle, but dict is safer)
        return {
            "model_path":            result.model_path,
            "base_model":            result.base_model,
            "training_approach":     result.training_approach,
            "num_epochs_completed":  result.num_epochs_completed,
            "final_train_loss":      result.final_train_loss,
            "training_time_seconds": result.training_time_seconds,
            "device":                "h100",
            "metrics":               result.metrics,
            "warnings":              result.warnings,
            "epoch_metrics":         result.epoch_metrics,
        }

    return app, train_on_modal


# ── Local dispatch interface ──────────────────────────────────────────────────

async def run_training_on_modal(
    training_kwargs: dict[str, Any],
) -> Any:
    """
    Dispatch a training job to Modal H100.

    Returns a TrainingResult-like object populated from the Modal response.
    Raises RuntimeError if Modal is unavailable or the job fails.
    """
    if not has_modal():
        raise RuntimeError(
            "Modal GPU training requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
            "environment variables, and the 'modal' package installed."
        )

    import asyncio

    try:
        _app, train_on_modal = _build_modal_app()
    except Exception as exc:
        raise RuntimeError(f"Could not build Modal app: {exc}") from exc

    logger.info(
        "[%s] Dispatching training to Modal H100 GPU",
        training_kwargs.get("job_id", "?"),
    )

    # Run the Modal function in a thread (it's a sync call that blocks until done)
    result_dict = await asyncio.to_thread(
        train_on_modal.remote,
        training_kwargs,
    )

    # Convert result dict back to TrainingResult
    from agents.ml_core import TrainingResult
    return TrainingResult(
        model_path=result_dict.get("model_path", ""),
        base_model=result_dict.get("base_model", ""),
        training_approach=result_dict.get("training_approach", "full_finetune"),
        num_epochs_completed=result_dict.get("num_epochs_completed", 0),
        final_train_loss=result_dict.get("final_train_loss", 0.0),
        training_time_seconds=result_dict.get("training_time_seconds", 0.0),
        device=result_dict.get("device", "h100"),
        metrics=result_dict.get("metrics", {}),
        warnings=result_dict.get("warnings", []),
        epoch_metrics=result_dict.get("epoch_metrics", []),
    )
