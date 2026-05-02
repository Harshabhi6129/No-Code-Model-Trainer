"""
Legacy REST training path (/train endpoint).
Uses HuggingFace Trainer directly with WebSocket streaming.
The agent pipeline (/chat endpoint) is the production path for new runs.
"""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Resolved so it's stable regardless of cwd
_RUNS_DIR = Path(__file__).parent.parent / "runs"


async def start_training(config: dict[str, Any], dataset_path: str, client_id: str | None = None) -> str:
    """
    Async wrapper — runs the blocking HF Trainer in a thread pool.
    Streams progress updates to the client via WebSocket.
    """
    from services.socket_manager import manager as socket_manager
    from services.training_monitor import TrainingMonitor
    from services.training_controller import create_controller

    job_id = client_id or str(uuid.uuid4())

    try:
        await socket_manager.send_status(job_id, "initializing", "Preparing training environment…")
        loop = asyncio.get_running_loop()

        result = await asyncio.to_thread(
            _blocking_train,
            job_id=job_id,
            config=config,
            dataset_path=dataset_path,
            socket_manager=socket_manager,
            loop=loop,
        )

        await socket_manager.send_completion(job_id, success=True, model_path=result.get("model_path", ""))
        logger.info("[%s] Training completed", job_id)
        return job_id

    except Exception as exc:
        logger.error("[%s] Training failed: %s", job_id, exc, exc_info=True)
        from services.socket_manager import manager as _sm
        await _sm.broadcast_json(job_id, {
            "type": "error",
            "message": "Training failed — check server logs for details.",
            "error_type": type(exc).__name__,
        })
        raise


def _blocking_train(
    *,
    job_id: str,
    config: dict[str, Any],
    dataset_path: str,
    socket_manager: Any,
    loop: Any,
) -> dict[str, Any]:
    """Blocking HF Trainer — must be called via asyncio.to_thread()."""
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Training libraries not installed. Run: pip install torch transformers datasets"
        ) from exc

    def _ws(msg: str) -> None:
        if loop:
            asyncio.run_coroutine_threadsafe(
                socket_manager.send_status(job_id, "processing", msg), loop
            )
        logger.info("[%s] %s", job_id, msg)

    output_dir = _RUNS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = config.get("model_id", "distilbert-base-uncased")
    params = config.get("parameters", {})
    use_cpu = config.get("use_cpu", False)

    _ws(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    _ws(f"Reading dataset: {dataset_path}")
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".json", ".jsonl"):
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Auto-detect columns
    text_col = max(
        (c for c in df.columns if df[c].dtype == "object"),
        key=lambda c: df[c].astype(str).str.len().mean(),
        default=None,
    )
    label_col = next(
        (c for c in df.columns if c != text_col and df[c].nunique() < 100),
        None,
    )
    if not text_col or not label_col:
        raise ValueError(f"Cannot auto-detect text/label columns. Found: {list(df.columns)}")

    _ws(f"Detected text='{text_col}', label='{label_col}'")
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})
    if df["label"].dtype == "object":
        df["label"] = pd.Categorical(df["label"]).codes
    num_labels = int(df["label"].nunique())
    dataset = Dataset.from_pandas(df)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

    tokenized = dataset.map(_tokenize, batched=True)

    _ws(f"Loading model {model_id} ({num_labels} labels)…")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    lr = float(_get(params, "learning_rate", 2e-5))
    epochs = int(_get(params, "num_epochs", 3))
    bs = int(_get(params, "batch_size", 8))
    wd = float(_get(params, "weight_decay", 0.01))

    device_str = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    _ws(f"Training on {device_str} — {epochs} epochs, lr={lr}, batch={bs}")

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        learning_rate=lr,
        weight_decay=wd,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        no_cuda=(device_str == "cpu"),
        disable_tqdm=True,
    )

    from services.callbacks import WebSocketCallback
    from services.training_monitor import TrainingMonitor
    from services.training_controller import create_controller

    monitor = TrainingMonitor(socket_manager, job_id, window_size=10)
    controller = create_controller(job_id)
    callback = WebSocketCallback(job_id, socket_manager, monitor, controller, None)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[callback],
    )
    controller.trainer = trainer

    _ws("Training started…")
    trainer.train()

    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    _ws(f"Model saved to {final_path}")

    return {"success": True, "model_path": str(final_path)}


def _get(params: dict, key: str, default: Any) -> Any:
    """Extract value from nested {'value': x} or flat dict."""
    v = params.get(key, {})
    if isinstance(v, dict):
        return v.get("value", default)
    return v if v is not None else default
