"""
Lightweight training-metric analyser for the agent layer.

Detects overfitting, divergence, stagnation, and gradient issues from epoch
metrics without any WebSocket or backend dependency.  Results are attached to
AgentResult.metadata and flow to the frontend through the existing SSE stream.

The backend-layer TrainingMonitor (backend/services/training_monitor.py) is a
separate, WebSocket-broadcasting variant used when the full backend is running.
This module serves the agent pipeline, which may run detached from the backend.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingInsights:
    overfitting_detected: bool = False
    divergence_detected: bool = False
    stagnation_detected: bool = False
    exploding_gradients: bool = False
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overfitting_detected": self.overfitting_detected,
            "divergence_detected": self.divergence_detected,
            "stagnation_detected": self.stagnation_detected,
            "exploding_gradients": self.exploding_gradients,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class TrainingInsightsAnalyzer:
    """
    Stateless analyser: call analyze() after training completes with the full
    epoch_metrics list from ml_core.  Returns a TrainingInsights summary.
    """

    # Thresholds
    _DIVERGENCE_LOSS = 100.0
    _STAGNATION_DELTA = 0.001   # avg per-epoch change below this → stagnation
    _STAGNATION_MIN_EPOCHS = 3  # need at least this many epochs to declare stagnation
    _OVERFIT_MIN_EPOCHS = 4     # need at least this many epochs to declare overfitting

    def analyze(self, epoch_metrics: list[dict[str, Any]]) -> TrainingInsights:
        """
        Analyse epoch_metrics (list of dicts with keys: epoch, loss, eval_loss, …).
        Safe on empty / single-entry lists.
        """
        insights = TrainingInsights()

        if not epoch_metrics:
            return insights

        train_losses = [e["loss"] for e in epoch_metrics if e.get("loss") is not None]
        eval_losses  = [e["eval_loss"] for e in epoch_metrics if e.get("eval_loss") is not None]

        self._check_divergence(train_losses, insights)
        self._check_overfitting(train_losses, eval_losses, insights)
        self._check_stagnation(train_losses, insights)

        return insights

    # ── Private detectors ────────────────────────────────────────────────────

    def _check_divergence(self, train_losses: list[float], ins: TrainingInsights) -> None:
        for loss in train_losses:
            if math.isnan(loss) or math.isinf(loss):
                ins.divergence_detected = True
                ins.warnings.append("NaN/Inf loss detected during training.")
                ins.suggestions.append("Reduce learning rate by 10× and retry.")
                return
        if train_losses and train_losses[-1] > self._DIVERGENCE_LOSS:
            ins.divergence_detected = True
            ins.warnings.append(
                f"Loss exploded to {train_losses[-1]:.1f} (threshold: {self._DIVERGENCE_LOSS})."
            )
            ins.suggestions.append(
                "Enable gradient clipping (max_grad_norm=1.0) and lower the learning rate."
            )

    def _check_overfitting(
        self,
        train_losses: list[float],
        eval_losses: list[float],
        ins: TrainingInsights,
    ) -> None:
        if len(train_losses) < self._OVERFIT_MIN_EPOCHS or len(eval_losses) < self._OVERFIT_MIN_EPOCHS:
            return
        train_slope = _slope(train_losses)
        eval_slope  = _slope(eval_losses)
        # Classic overfit: train↓ and eval↑
        if train_slope < -0.01 and eval_slope > 0.01:
            ins.overfitting_detected = True
            ins.warnings.append(
                "Overfitting: training loss is decreasing while eval loss is increasing."
            )
            ins.suggestions.append(
                "Increase weight_decay, add dropout, or reduce num_epochs."
            )

    def _check_stagnation(self, train_losses: list[float], ins: TrainingInsights) -> None:
        if len(train_losses) < self._STAGNATION_MIN_EPOCHS:
            return
        changes = [abs(train_losses[i] - train_losses[i - 1]) for i in range(1, len(train_losses))]
        avg_change = sum(changes) / len(changes)
        if avg_change < self._STAGNATION_DELTA:
            ins.stagnation_detected = True
            ins.warnings.append(
                f"Loss plateaued (avg Δ={avg_change:.5f} per epoch)."
            )
            ins.suggestions.append(
                "Try a cyclic or cosine LR schedule, or increase the learning rate slightly."
            )


# ── Module-level singleton ────────────────────────────────────────────────────

analyzer = TrainingInsightsAnalyzer()


# ── Helper ───────────────────────────────────────────────────────────────────

def _slope(values: list[float]) -> float:
    """Least-squares slope of a list of floats. Returns 0.0 on degenerate input."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0
