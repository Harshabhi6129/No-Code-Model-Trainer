"""
Episodic Memory for ModelForge.

Stores successful recipe+grade pairs indexed by a feature vector derived from
the dataset profile. On subsequent runs with similar datasets, the recalled
recipe is injected as a few-shot exemplar in the ModelAgent's system prompt,
giving Claude a warm start grounded in real performance evidence.

Architecture:
  • Feature vector: 5 floats derived from DataProfile fields
  • Similarity: cosine similarity on the feature vectors
  • Storage: Supabase episodic_memories table (optional) + in-memory list (L1)
  • Only memories with eval_grade in {A, B} are stored
  • Recall threshold: cosine_similarity ≥ 0.85 (high similarity required)

The memory system is intentionally read-only in the ModelAgent — it never
replaces the LLM call, only pre-fills the context. The LLM still makes the
final decision.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_RECALL_THRESHOLD   = 0.85   # minimum cosine similarity to recall a memory
_CACHEABLE_GRADES   = frozenset({"A", "B"})
_MAX_L1_ENTRIES     = 500    # in-process memory (across all users, per worker)
_SUPABASE_TABLE     = "episodic_memories"


# ── Feature vector ────────────────────────────────────────────────────────────

def _feature_vector(data_profile: dict[str, Any]) -> list[float]:
    """
    Extract a 5-dimensional feature vector from a DataProfile dict.
    All features are scaled to [0, 1] or log-normalized for comparability.

      0: log(num_rows + 1) / 15         — dataset scale
      1: num_classes / 50               — classification complexity
      2: log(avg_word_count + 1) / 10   — text length scale
      3: label_noise_estimate            — data quality
      4: text_quality_score              — text quality
    """
    num_rows    = float(data_profile.get("num_rows", 0) or 0)
    num_classes = float(
        data_profile.get("num_classes")
        or len(data_profile.get("label_distribution", {}))
        or 1
    )
    avg_words   = float(data_profile.get("avg_word_count", 50) or 50)
    noise       = float(data_profile.get("label_noise_estimate", 0) or 0)
    quality     = float(data_profile.get("text_quality_score", 1.0) or 1.0)

    return [
        min(math.log(num_rows + 1) / 15.0, 1.0),
        min(num_classes / 50.0, 1.0),
        min(math.log(avg_words + 1) / 10.0, 1.0),
        min(noise, 1.0),
        min(quality, 1.0),
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


# ── Memory entry ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    feature_vector:   list[float]
    task_type:        str
    model_recipe:     dict[str, Any]
    eval_grade:       str
    eval_f1:          float
    created_at:       float  = field(default_factory=time.time)


# ── Supabase L2 (optional) ────────────────────────────────────────────────────

def _supabase_client():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _l2_recall(
    feature_vec: list[float],
    task_type: str,
    threshold: float,
) -> list[MemoryEntry]:
    """Fetch candidate memories from Supabase. Returns [] on error."""
    client = _supabase_client()
    if client is None:
        return []
    try:
        import json as _json
        resp = (
            client.table(_SUPABASE_TABLE)
            .select("feature_vector, task_type, model_recipe_json, eval_grade, eval_f1, created_at")
            .eq("task_type", task_type)
            .execute()
        )
        results = []
        for row in (resp.data or []):
            fv = row.get("feature_vector")
            if not fv or not isinstance(fv, list):
                continue
            sim = _cosine_similarity(feature_vec, fv)
            if sim < threshold:
                continue
            recipe = _json.loads(row.get("model_recipe_json") or "{}")
            results.append(MemoryEntry(
                feature_vector=fv,
                task_type=row.get("task_type", task_type),
                model_recipe=recipe,
                eval_grade=row.get("eval_grade", "B"),
                eval_f1=float(row.get("eval_f1", 0)),
                created_at=_parse_iso(row.get("created_at", "")),
            ))
        return sorted(results, key=lambda e: e.eval_f1, reverse=True)
    except Exception as exc:
        logger.debug("EpisodicMemory L2 recall error: %s", exc)
        return []


def _l2_memorize(entry: MemoryEntry) -> None:
    client = _supabase_client()
    if client is None:
        return
    try:
        import json as _json
        client.table(_SUPABASE_TABLE).insert({
            "feature_vector":   entry.feature_vector,
            "task_type":        entry.task_type,
            "model_recipe_json": _json.dumps(entry.model_recipe),
            "eval_grade":       entry.eval_grade,
            "eval_f1":          entry.eval_f1,
        }).execute()
    except Exception as exc:
        logger.debug("EpisodicMemory L2 memorize error: %s", exc)


def _parse_iso(ts_str: str) -> float:
    if not ts_str:
        return time.time()
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


# ── EpisodicMemory (public interface) ─────────────────────────────────────────

class EpisodicMemory:
    """
    Read-before-LLM, write-after-eval memory for model recipes.

    Thread-safety: the L1 list is not thread-safe. For the asyncio agent
    pipeline (single event loop), this is not an issue.
    """

    def __init__(self) -> None:
        self._l1: list[MemoryEntry] = []

    def recall(
        self,
        data_profile: dict[str, Any],
        task_type: str,
        top_k: int = 3,
    ) -> list[MemoryEntry]:
        """
        Return the top-k most similar memories above the recall threshold.

        Queries L1 first, then L2 (Supabase). Results are sorted by eval_f1 desc.
        Returns [] on no match or if memory is empty.
        """
        fv = _feature_vector(data_profile)

        # L1 scan
        l1_hits = [
            e for e in self._l1
            if e.task_type == task_type
            and _cosine_similarity(fv, e.feature_vector) >= _RECALL_THRESHOLD
        ]
        l1_hits.sort(key=lambda e: e.eval_f1, reverse=True)

        if l1_hits:
            logger.info("EpisodicMemory L1 hit (%d candidate(s))", len(l1_hits))
            return l1_hits[:top_k]

        # L2 fallback
        try:
            l2_hits = _l2_recall(fv, task_type, _RECALL_THRESHOLD)
        except Exception as exc:
            logger.debug("EpisodicMemory L2 error: %s", exc)
            l2_hits = []

        if l2_hits:
            logger.info("EpisodicMemory L2 hit (%d candidate(s))", len(l2_hits))
            # Promote best to L1
            for e in l2_hits[:top_k]:
                self._add_to_l1(e)

        return l2_hits[:top_k]

    def memorize(
        self,
        data_profile: dict[str, Any],
        task_type: str,
        model_recipe: dict[str, Any],
        eval_grade: str,
        eval_f1: float,
    ) -> None:
        """
        Store a successful recipe in memory.
        Only stores if eval_grade ∈ {A, B}. Silent on any error.
        """
        if eval_grade not in _CACHEABLE_GRADES:
            return

        fv = _feature_vector(data_profile)
        entry = MemoryEntry(
            feature_vector=fv,
            task_type=task_type,
            model_recipe=model_recipe,
            eval_grade=eval_grade,
            eval_f1=eval_f1,
        )
        self._add_to_l1(entry)
        try:
            _l2_memorize(entry)
        except Exception as exc:
            logger.debug("EpisodicMemory memorize error: %s", exc)

        logger.info(
            "EpisodicMemory: stored recipe for task=%s grade=%s f1=%.3f",
            task_type, eval_grade, eval_f1,
        )

    def size(self) -> int:
        return len(self._l1)

    def clear(self) -> None:
        self._l1.clear()

    def _add_to_l1(self, entry: MemoryEntry) -> None:
        self._l1.append(entry)
        if len(self._l1) > _MAX_L1_ENTRIES:
            # Evict the oldest entry
            self._l1.pop(0)


# ── Module-level singleton ────────────────────────────────────────────────────

episodic_memory = EpisodicMemory()


# ── Formatting helper (for ModelAgent prompt injection) ───────────────────────

def format_memory_exemplar(entry: MemoryEntry) -> str:
    """
    Format a MemoryEntry as a concise exemplar string for Claude's system prompt.
    """
    recipe = entry.model_recipe
    base   = recipe.get("base_model", "unknown")
    approach = recipe.get("training_approach", "unknown")
    lr     = recipe.get("learning_rate", "?")
    lora_r = recipe.get("lora_r")
    lora_info = f", LoRA-r={lora_r}" if lora_r else ""
    return (
        f"Similar dataset achieved grade {entry.eval_grade} "
        f"(F1={entry.eval_f1:.3f}) with: "
        f"{base} / {approach}{lora_info} / lr={lr:.2e}"
    )
