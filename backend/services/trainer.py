"""
Training Orchestrator - NON-BLOCKING version with proper asyncio threading.
"""
import os
import uuid
import pandas as pd
import asyncio
from pathlib import Path
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import logging

from services.callbacks import WebSocketCallback
from services.socket_manager import manager as socket_manager

logger = logging.getLogger(__name__)

# Output directory for trained models
from core.config import OUTPUT_DIR as OUTPUTS_DIR


def _blocking_train_logic(job_id: str, config: Dict[str, Any], dataset_path: str, loop=None):
    """
    BLOCKING function that runs in a thread pool.
    Contains all the synchronous HuggingFace operations.
    """
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(exist_ok=True)
    
    def log_sync(msg):
        """Helper to send logs via WebSocket from sync thread"""
        if loop:
            asyncio.run_coroutine_threadsafe(
                socket_manager.send_status(job_id, "processing", msg),
                loop
            )
        logger.info(f"[{job_id}] {msg}")

    try:
        # Extract configuration
        model_id = config.get('model_id', 'distilbert-base-uncased')
        params = config.get('parameters', {})
        
        log_sync(f"🚀 Starting Job Setup...")
        
        # Load tokenizer
        log_sync("⬇️ Downloading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load and tokenize dataset
        log_sync(f"📂 Reading Dataset from {dataset_path}...")
        
        if not dataset_path or not Path(dataset_path).exists():
            log_sync("⚠️ Dataset path invalid, using debug dataset")
            dataset = load_dataset("glue", "sst2", split="train[:50]")
            dataset = dataset.rename_column("sentence", "text")
            num_labels = 2
        else:
            # Load CSV
            df = pd.read_csv(dataset_path)
            log_sync(f"✅ Loaded CSV. Columns: {df.columns.tolist()}")
            
            log_sync(f"🧠 Analyzing Columns...")
            
            # Heuristic column mapping
            text_col = None
            label_col = None
            
            # Find text column: longest average length
            text_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_len = df[col].astype(str).str.len().mean()
                    text_cols.append((col, avg_len))
            
            if text_cols:
                text_col = max(text_cols, key=lambda x: x[1])[0]
            
            # Find label column: few unique values
            for col in df.columns:
                if col != text_col:
                    unique_count = df[col].nunique()
                    if unique_count < 100:
                        label_col = col
                        break
            
            if not text_col or not label_col:
                raise ValueError(f"Could not auto-detect columns. Found: {df.columns.tolist()}")
            
            log_sync(f"✅ Detected - Text: {text_col}, Label: {label_col}")
            
            # Create Dataset
            df_subset = df[[text_col, label_col]].copy()
            df_subset.columns = ['text', 'label']
            
            # Convert labels to integers
            if df_subset['label'].dtype == 'object':
                df_subset['label'] = pd.Categorical(df_subset['label']).codes
            
            num_labels = int(df_subset['label'].nunique())
            dataset = Dataset.from_pandas(df_subset)
            
            # Dry Run Safety
            if len(dataset) > 0:
                log_sync(f"👀 First row sample: {dataset[0]}")
        
        # Tokenize
        log_sync("✂️ Tokenizing Data...")
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        log_sync(f"✅ Dataset ready: {len(tokenized_dataset)} samples, {num_labels} labels")
        
        # Load model with correct num_labels
        log_sync(f"⬇️ Downloading Model {model_id} (This may take a minute)...")
        log_sync(f"🔓 Loading custom model architecture for {model_id}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        # Extract hyperparameters
        learning_rate = float(params.get('learning_rate', {}).get('value', 2e-5))
        num_epochs = int(params.get('num_epochs', {}).get('value', 3))
        batch_size = int(params.get('batch_size', {}).get('value', 8))
        weight_decay = float(params.get('weight_decay', {}).get('value', 0.01))
        
        logger.info(f"[{job_id}] Hyperparams - LR: {learning_rate}, Epochs: {num_epochs}, BS: {batch_size}")
        
        use_cpu = config.get('use_cpu', False)
        if use_cpu:
            log_sync("Forcing CPU training (no_cuda=True)")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=10,
            save_steps=500,
            eval_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            report_to="none",
            no_cuda=use_cpu
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Initialize AI Training Monitor for real-time insights
        from services.training_monitor import TrainingMonitor
        
        log_sync("🔥 Starting Training Loop...")
        from services.training_controller import create_controller
        from services.commentator import AICommentator
        
        training_monitor = TrainingMonitor(socket_manager, job_id, window_size=10)
        
        # Initialize Training Controller for pause/resume and hot-swap
        training_controller = create_controller(job_id)
        
        # Initialize AI Commentator for natural language updates
        ai_commentator = AICommentator(socket_manager, job_id, trigger_interval=20)
        
        # Initialize custom callback with all AI services
        ws_callback = WebSocketCallback(
            job_id,
            socket_manager,
            training_monitor,
            training_controller,
            ai_commentator
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[ws_callback]
        )
        
        # Connect trainer to controller for hot-swap capability
        training_controller.trainer = trainer
        
        # THIS IS THE BLOCKING CALL
        trainer.train()
        
        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        logger.info(f"[{job_id}] Training completed successfully")
        
        return {
            'success': True,
            'model_path': str(final_model_path)
        }
        
    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {e}", exc_info=True)
        raise


async def start_training(config: Dict[str, Any], dataset_path: str, client_id: str = None):
    """
    Async wrapper that runs training in a thread pool (NON-BLOCKING).
    
    This function returns immediately and runs the training in the background.
    """
    if client_id is None:
        client_id = str(uuid.uuid4())
    
    job_id = client_id
    
    try:
        # Send initial status BEFORE starting heavy work
        await socket_manager.send_status(job_id, "initializing", "Preparing training environment...")
        
        logger.info(f"[{job_id}] Starting training job (async wrapper)")
        
        # Send status update
        await socket_manager.send_status(job_id, "loading", "Loading model and dataset...")
        
        # Run the BLOCKING function in a thread pool
        # This prevents it from blocking the async event loop
        loop = asyncio.get_running_loop()
        result = await asyncio.to_thread(
            _blocking_train_logic,
            job_id,
            config,
            dataset_path,
            loop
        )
        
        # Send completion
        await socket_manager.send_completion(
            job_id,
            success=True,
            model_path=result.get('model_path', '')
        )
        
        logger.info(f"[{job_id}] Training job completed")
        return job_id
        
    except Exception as e:
        logger.error(f"[{job_id}] Training job failed: {e}", exc_info=True)
        
        # Send error to frontend
        await socket_manager.broadcast_json(
            job_id,
            {
                "type": "error",
                "message": f"Training failed: {str(e)}",
                "error_type": type(e).__name__
            }
        )
        
        raise
