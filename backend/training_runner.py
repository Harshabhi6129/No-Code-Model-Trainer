# training_runner.py
import os
import threading
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import wandb

from ws_broker import publish
from resource_monitor import start_resource_monitoring, stop_resource_monitoring

# Global training state management
training_jobs = {}

class TrainingState:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "running"  # running, paused, stopped, completed, failed
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.mutable_params = {}
        self.start_time = time.time()
        
    def pause(self):
        self.status = "paused"
        self.pause_event.set()
        
    def resume(self):
        self.status = "running"
        self.pause_event.clear()
        
    def stop(self):
        self.status = "stopped"
        self.stop_event.set()
        
    def update_params(self, **params):
        self.mutable_params.update(params)

def get_training_state(run_id: str) -> Optional[TrainingState]:
    return training_jobs.get(run_id)

def save_checkpoint(run_id: str, model, optimizer, epoch: int, batch: int, metrics: dict):
    """Save training checkpoint"""
    checkpoint_dir = Path("checkpoints") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch,
        "metrics": metrics,
        "random_state": torch.get_rng_state(),
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_batch_{batch}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(run_id: str, model, optimizer):
    """Load latest checkpoint"""
    checkpoint_dir = Path("checkpoints") / run_id
    if not checkpoint_dir.exists():
        return None
        
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
        
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    checkpoint = torch.load(latest_checkpoint)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["random_state"])
    
    return checkpoint

# Ensure required folders exist
Path("runs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

# ------------------------
# Helper: Prepare Dataset
# ------------------------
def prepare_dataset(dataset_path: str, text_col: str, label_col: str):
    """
    Load dataset from CSV and prepare for HuggingFace Trainer.
    """
    dataset = load_dataset("csv", data_files=str(dataset_path))

    # Auto-detect columns
    if text_col not in dataset["train"].column_names:
        text_col = dataset["train"].column_names[0]
    if label_col not in dataset["train"].column_names:
        label_col = dataset["train"].column_names[-1]

    return dataset, text_col, label_col

# ------------------------
# Background Training Task
# ------------------------
class PausableTrainer(Trainer):
    def __init__(self, training_state: TrainingState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_state = training_state
        self.step_count = 0
        
    def training_step(self, model, inputs):
        # Check for pause/stop before each batch
        if self.training_state.pause_event.is_set():
            publish(self.training_state.run_id, {"event": "training_paused", "epoch": self.training_state.current_epoch})
            while self.training_state.pause_event.is_set() and not self.training_state.stop_event.is_set():
                time.sleep(1)
                
        if self.training_state.stop_event.is_set():
            raise KeyboardInterrupt("Training stopped by user")
            
        # Apply parameter updates
        if self.training_state.mutable_params:
            self._apply_param_updates()
            
        # Perform training step
        loss = super().training_step(model, inputs)
        
        # Calculate gradient norm
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Update step count and emit metrics every 10 steps
        self.step_count += 1
        if self.step_count % 10 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Emit real-time metrics
            publish(self.training_state.run_id, {
                "type": "training_update",
                "epoch": self.training_state.current_epoch,
                "batch": self.step_count,
                "total_batches": self.training_state.total_batches,
                "metrics": {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "grad_norm": grad_norm
                },
                "time_elapsed": time.time() - self.training_state.start_time
            })
            
        return loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Get predictions for detailed metrics
        if eval_dataset is not None:
            predictions = self.predict(eval_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Calculate additional metrics
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            
            # Get sample predictions (first 5)
            sample_predictions = []
            for i in range(min(5, len(y_true))):
                confidence = np.max(torch.softmax(torch.tensor(predictions.predictions[i]), dim=0).numpy())
                sample_predictions.append({
                    "true_label": str(y_true[i]),
                    "predicted_label": str(y_pred[i]),
                    "confidence": float(confidence)
                })
            
            # Emit comprehensive evaluation metrics
            publish(self.training_state.run_id, {
                "type": "evaluation_update",
                "epoch": self.training_state.current_epoch,
                "metrics": {
                    "val_loss": eval_results.get(f"{metric_key_prefix}_loss", 0),
                    "val_accuracy": eval_results.get(f"{metric_key_prefix}_accuracy", 0),
                    "val_f1": f1,
                    "val_precision": precision,
                    "val_recall": recall
                },
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "sample_predictions": sample_predictions
            })
            
        return eval_results
        
    def _apply_param_updates(self):
        params = self.training_state.mutable_params
        changes_applied = {}
        
        if "learning_rate" in params:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = params["learning_rate"]
            changes_applied["learning_rate"] = params["learning_rate"]
            
        if "weight_decay" in params:
            for param_group in self.optimizer.param_groups:
                param_group["weight_decay"] = params["weight_decay"]
            changes_applied["weight_decay"] = params["weight_decay"]
            
        if "dropout" in params:
            # Update model dropout if supported
            if hasattr(self.model.config, 'hidden_dropout_prob'):
                self.model.config.hidden_dropout_prob = params["dropout"]
            changes_applied["dropout"] = params["dropout"]
            
        # Log parameter changes
        if changes_applied:
            publish(self.training_state.run_id, {
                "event": "params_updated",
                "params": changes_applied,
                "epoch": self.training_state.current_epoch
            })
            
        # Clear applied params
        self.training_state.mutable_params.clear()

def _train_task(run_id: str, config: Dict):
    """
    Run training in a background thread with pause/resume support.
    """
    training_state = TrainingState(run_id)
    training_jobs[run_id] = training_state
    
    # Start resource monitoring
    start_resource_monitoring(run_id)
    
    try:
        # Extract payload
        model_name = config["model"]
        dataset_path = config["dataset_path"]
        text_col = config.get("text_col", "text")
        label_col = config.get("label_col", "label")
        num_labels = config.get("num_labels", 2)
        output_dir = Path("runs") / run_id

        # ---------------------
        # 1️⃣ Init W&B
        # ---------------------
        wandb.init(project="no_code_finetune", name=run_id, config=config)
        publish(run_id, {"event": "wandb_url", "url": wandb.run.get_url()})

        # ---------------------
        # 2️⃣ Load Dataset
        # ---------------------
        publish(run_id, {"event": "log", "message": "Loading dataset..."})
        dataset, text_col, label_col = prepare_dataset(dataset_path, text_col, label_col)

        # ---------------------
        # 3️⃣ Tokenizer
        # ---------------------
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_fn(batch):
            return tokenizer(batch[text_col], truncation=True)

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized = tokenized.rename_column(label_col, "labels")
        tokenized.set_format("torch")

        # ---------------------
        # 4️⃣ Model
        # ---------------------
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # ---------------------
        # 5️⃣ Training Arguments
        # ---------------------
        args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.get("learning_rate", 5e-5),
            per_device_train_batch_size=config.get("batch_size", 16),
            per_device_eval_batch_size=config.get("batch_size", 16),
            num_train_epochs=config.get("epochs", 3),
            weight_decay=config.get("weight_decay", 0.01),
            logging_dir=str(output_dir / "logs"),
            report_to=["wandb"],
        )

        # ---------------------
        # 6️⃣ Trainer Setup
        # ---------------------
        trainer = PausableTrainer(
            training_state=training_state,
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation", None),
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        # ---------------------
        # 7️⃣ Start Training
        # ---------------------
        publish(run_id, {"event": "log", "message": "Starting training..."})
        
        # Set total batches for progress tracking
        training_state.total_batches = len(trainer.get_train_dataloader())
        
        # Check for existing checkpoint
        checkpoint = load_checkpoint(run_id, model, trainer.optimizer)
        resume_from_checkpoint = None
        if checkpoint:
            training_state.current_epoch = checkpoint["epoch"]
            training_state.current_batch = checkpoint["batch"]
            publish(run_id, {"event": "log", "message": f"Resuming from epoch {checkpoint['epoch']}, batch {checkpoint['batch']}"})
            
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # ---------------------
        # 8️⃣ Save Model
        # ---------------------
        if training_state.status != "stopped":
            trainer.save_model(output_dir)
            wandb.save(str(output_dir / "*"))
            training_state.status = "completed"
            publish(run_id, {"event": "log", "message": "Training completed!"})
            publish(run_id, {"event": "summary", "message": f"Model saved at {output_dir}"})
        else:
            publish(run_id, {"event": "log", "message": "Training stopped by user"})

    except KeyboardInterrupt:
        training_state.status = "stopped"
        # Save final checkpoint before stopping
        save_checkpoint(run_id, model, trainer.optimizer, training_state.current_epoch, training_state.current_batch, {})
        publish(run_id, {"event": "log", "message": "Training stopped by user"})
    except Exception as e:
        training_state.status = "failed"
        publish(run_id, {"event": "error", "message": str(e)})
    finally:
        # Clean up
        stop_resource_monitoring(run_id)
        if run_id in training_jobs:
            del training_jobs[run_id]
        wandb.finish()

# ------------------------
# Public API
# ------------------------
def start_training(run_id: str, config: Dict):
    """
    Launch training in background thread.
    """
    thread = threading.Thread(target=_train_task, args=(run_id, config), daemon=True)
    thread.start()
