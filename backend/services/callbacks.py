"""
Custom HuggingFace Trainer Callbacks for WebSocket streaming.
"""
import asyncio
import time
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WebSocketCallback(TrainerCallback):
    """
    Custom callback that streams training metrics to WebSocket clients in real-time.
    Integrates AI monitoring, commentary, and training control.
    """
    
    def __init__(
        self,
        client_id: str,
        socket_manager,
        training_monitor=None,
        training_controller=None,
        ai_commentator=None
    ):
        """
        Initialize the callback.
        
        Args:
            client_id: Unique identifier for the training job
            socket_manager: Instance of ConnectionManager
            training_monitor: Optional TrainingMonitor for AI insights
            training_controller: Optional TrainingController for pause/resume
            ai_commentator: Optional AICommentator for natural language updates
        """
        self.client_id = client_id
        self.socket_manager = socket_manager
        self.training_monitor = training_monitor
        self.training_controller = training_controller
        self.ai_commentator = ai_commentator
        self.start_time = None
        self.step_start_time = None
    
    def _send_update(self, data: dict):
        """
        Send update via WebSocket (handles async from sync context).
        """
        try:
            # Create a new event loop if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.socket_manager.broadcast_json(self.client_id, data))
            loop.close()
        except Exception as e:
            logger.error(f"Failed to send WebSocket update: {e}")
    
    async def _async_analyze_metrics(self, step: int, logs: Dict[str, Any]):
        """
        Run AI analysis on metrics asynchronously.
        """
        if self.training_monitor:
            try:
                await self.training_monitor.analyze_step(
                    step=step,
                    train_loss=logs.get('loss'),
                    val_loss=logs.get('eval_loss'),
                    learning_rate=logs.get('learning_rate'),
                    grad_norm=logs.get('grad_norm')
                )
            except Exception as e:
                logger.error(f"Error in training monitor: {e}")
    
    async def _async_generate_commentary(self, step: int, logs: Dict[str, Any], epoch: Optional[float]):
        """
        Generate AI commentary asynchronously.
        """
        if self.ai_commentator:
            try:
                await self.ai_commentator.process_step(
                    step=step,
                    loss=logs.get('loss'),
                    accuracy=logs.get('eval_accuracy'),
                    learning_rate=logs.get('learning_rate'),
                    epoch=epoch
                )
            except Exception as e:
                logger.error(f"Error in AI commentator: {e}")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()
        
        self._send_update({
            "type": "train_begin",
            "total_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size
        })
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch."""
        self._send_update({
            "type": "epoch_begin",
            "epoch": int(state.epoch) if state.epoch else 0
        })
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each step - CHECK FOR PAUSE"""
        if self.training_controller:
            # This will block if training is paused
            self.training_controller.wait_if_paused()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        # Only send updates every N steps to avoid overwhelming the client
        if state.global_step % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            self._send_update({
                "type": "step",
                "step": state.global_step,
                "total_steps": state.max_steps,
                "epoch": int(state.epoch) if state.epoch else 0,
                "elapsed_time": elapsed
            })
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        """
        Called when logging occurs (this captures loss, learning_rate, etc.).
        This is the CRITICAL method for streaming metrics, AI analysis, and commentary.
        """
        if logs is None:
            return
        
        # Extract key metrics
        metrics = {
            "type": "metrics",
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
        }
        
        # Add all logged metrics
        if "loss" in logs:
            metrics["loss"] = float(logs["loss"])
        if "learning_rate" in logs:
            metrics["learning_rate"] = float(logs["learning_rate"])
        if "eval_loss" in logs:
            metrics["eval_loss"] = float(logs["eval_loss"])
        if "eval_accuracy" in logs:
            metrics["eval_accuracy"] = float(logs["eval_accuracy"])
        if "grad_norm" in logs:
            metrics["grad_norm"] = float(logs["grad_norm"])
        
        # Add timing info
        if self.start_time:
            metrics["elapsed_time"] = time.time() - self.start_time
        
        self._send_update(metrics)
        
        # Run AI analysis and commentary (async)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run monitor analysis
            if self.training_monitor:
                loop.run_until_complete(self._async_analyze_metrics(state.global_step, logs))
            
            # Generate AI commentary
            if self.ai_commentator:
                loop.run_until_complete(
                    self._async_generate_commentary(state.global_step, logs, state.epoch)
                )
            
            loop.close()
        except Exception as e:
            logger.error(f"Failed to run AI services: {e}")
    
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch."""
        self._send_update({
            "type": "epoch_end",
            "epoch": int(state.epoch) if state.epoch else 0
        })
        
        # Send milestone commentary
        if self.ai_commentator:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.ai_commentator.send_milestone_commentary(
                        f"Epoch {int(state.epoch)} complete"
                    )
                )
                loop.close()
            except Exception as e:
                logger.error(f"Failed to send milestone commentary: {e}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        self._send_update({
            "type": "train_end",
            "total_time": elapsed,
            "final_step": state.global_step
        })
        
        # Final milestone commentary
        if self.ai_commentator:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.ai_commentator.send_milestone_commentary(
                        "Training mission complete"
                    )
                )
                loop.close()
            except Exception as e:
                logger.error(f"Failed to send final commentary: {e}")
