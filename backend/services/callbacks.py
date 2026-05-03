"""
Custom HuggingFace Trainer Callbacks for WebSocket streaming.
All async sends use asyncio.run_coroutine_threadsafe with the captured main
event loop — never asyncio.new_event_loop(), which creates isolated loops
that have no knowledge of the existing WebSocket connections.
"""
import asyncio
import time
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WebSocketCallback(TrainerCallback):
    """
    Streams training metrics to WebSocket clients in real-time.
    Must be constructed with the running event loop so that callbacks
    fired from the trainer thread can schedule coroutines on the main loop.
    """

    def __init__(
        self,
        client_id: str,
        socket_manager,
        training_monitor=None,
        training_controller=None,
        ai_commentator=None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.client_id = client_id
        self.socket_manager = socket_manager
        self.training_monitor = training_monitor
        self.training_controller = training_controller
        # ai_commentator kept for API compat but no longer used
        self._loop = loop
        self.start_time: float | None = None

    def _send_update(self, data: dict) -> None:
        """Schedule a WebSocket broadcast on the main event loop (fire-and-forget)."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.socket_manager.broadcast_json(self.client_id, data),
                self._loop,
            )
        except Exception as exc:
            logger.warning("Failed to schedule WebSocket update: %s", exc)

    def _schedule(self, coro) -> None:
        """Schedule an arbitrary coroutine on the main event loop (fire-and-forget)."""
        if self._loop is None or self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as exc:
            logger.warning("Failed to schedule coroutine: %s", exc)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        self._send_update({
            "type": "train_begin",
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
        })

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._send_update({
            "type": "epoch_begin",
            "epoch": int(state.epoch) if state.epoch else 0,
        })

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.training_controller:
            self.training_controller.wait_if_paused()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            self._send_update({
                "type": "step",
                "step": state.global_step,
                "total_steps": state.max_steps,
                "epoch": int(state.epoch) if state.epoch else 0,
                "elapsed_time": round(elapsed, 1),
            })

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if logs is None:
            return

        metrics: dict[str, Any] = {
            "type": "metrics",
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
        }
        for key in ("loss", "learning_rate", "eval_loss", "eval_accuracy", "grad_norm"):
            if key in logs:
                metrics[key] = round(float(logs[key]), 6)
        if self.start_time:
            metrics["elapsed_time"] = round(time.time() - self.start_time, 1)

        self._send_update(metrics)

        if self.training_monitor:
            self._schedule(self.training_monitor.analyze_step(
                step=state.global_step,
                train_loss=logs.get("loss"),
                val_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
                grad_norm=logs.get("grad_norm"),
            ))

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._send_update({
            "type": "epoch_end",
            "epoch": int(state.epoch) if state.epoch else 0,
        })

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._send_update({
            "type": "train_end",
            "total_time": round(elapsed, 1),
            "final_step": state.global_step,
        })
