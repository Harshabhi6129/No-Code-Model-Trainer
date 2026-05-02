"""
LRU in-memory model cache for the /infer endpoint.
Holds up to MAX_CACHED loaded models; evicts the least-recently-used on overflow.
All model loads and forward passes run in a thread pool via asyncio.to_thread().
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_CACHED = 3
_INFER_TIMEOUT = 30   # seconds hard cap on inference


@dataclass
class _CachedModel:
    run_id:      str
    model:       Any
    tokenizer:   Any
    label_names: list[str]
    last_used:   float = field(default_factory=time.monotonic)


class ModelCache:
    def __init__(self) -> None:
        self._cache: dict[str, _CachedModel] = {}

    async def predict(
        self,
        *,
        run_id: str,
        text: str,
        artifact_path: str,
        label_names: list[str],
    ) -> dict[str, Any]:
        """
        Run classification on `text`. Loads model on first call, then reuses cache.
        Raises asyncio.TimeoutError if inference exceeds _INFER_TIMEOUT seconds.
        """
        entry = self._cache.get(run_id)
        if entry is None:
            entry = await self._load(run_id=run_id, artifact_path=artifact_path, label_names=label_names)

        entry.last_used = time.monotonic()

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._infer, entry, text),
                timeout=_INFER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Inference timed out after {_INFER_TIMEOUT}s. "
                "Try shorter text or a lighter model."
            )

        return result

    async def _load(self, *, run_id: str, artifact_path: str, label_names: list[str]) -> _CachedModel:
        """Load model + tokenizer from disk. Evicts LRU if cache is full."""
        # Evict least recently used
        if len(self._cache) >= MAX_CACHED:
            lru_id = min(self._cache, key=lambda k: self._cache[k].last_used)
            logger.info("ModelCache: evicting run %s from cache", lru_id)
            del self._cache[lru_id]

        model_dir = Path(artifact_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model files not found at '{artifact_path}'. "
                "The server may have restarted since training completed."
            )

        logger.info("ModelCache: loading model from %s", model_dir)
        model, tokenizer = await asyncio.to_thread(
            self._blocking_load, str(model_dir), label_names
        )

        entry = _CachedModel(
            run_id=run_id,
            model=model,
            tokenizer=tokenizer,
            label_names=label_names,
        )
        self._cache[run_id] = entry
        logger.info("ModelCache: loaded %d model(s) in cache", len(self._cache))
        return entry

    @staticmethod
    def _blocking_load(model_dir: str, label_names: list[str]) -> tuple:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError(
                "Inference libraries not installed. pip install torch transformers"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        # Attach id2label from label_names if not already set
        if label_names and (
            not model.config.id2label
            or model.config.id2label == {0: "LABEL_0"}
        ):
            model.config.id2label = {i: lbl for i, lbl in enumerate(label_names)}
            model.config.label2id = {lbl: i for i, lbl in enumerate(label_names)}

        return model, tokenizer

    @staticmethod
    def _infer(entry: _CachedModel, text: str) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        inputs = entry.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            logits = entry.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs]

        label_names = entry.label_names or [
            entry.model.config.id2label.get(i, f"class_{i}") for i in range(len(probs))
        ]

        scored = sorted(
            [{"label": lbl, "score": round(float(p), 6), "pct": round(float(p) * 100, 1)}
             for lbl, p in zip(label_names, probs)],
            key=lambda x: x["score"],
            reverse=True,
        )

        return {
            "predicted_label": scored[0]["label"],
            "confidence": scored[0]["score"],
            "all_scores": scored,
        }

    def evict(self, run_id: str) -> None:
        self._cache.pop(run_id, None)


# Module-level singleton used by main.py
cache = ModelCache()
