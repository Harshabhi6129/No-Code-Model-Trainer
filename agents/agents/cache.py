"""
Recipe Memoization Cache for ModelForge.

Two-tier cache: L1 (in-memory LRU, always available) + L2 (Supabase, optional).

Cache key: coarse SHA-256 of the dataset profile + task type.
Bucket strategy keeps keys stable across similar (not identical) datasets —
a 950-row dataset and a 1,200-row dataset both hit the "sm" bucket and
benefit from the same LLM warm-start recipe.

Only recipes that achieved eval grade ≥ B are cached; poor recipes are
never stored so they can't poison future runs.

Supabase L2 is optional — if SUPABASE_URL / SUPABASE_KEY are not set, or if
the supabase package isn't installed, the cache falls back gracefully to L1-only.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_L1_MAX_ENTRIES = 100
_CACHE_TTL_DAYS = 30
_CACHE_TTL_SECS = _CACHE_TTL_DAYS * 86_400

# Minimum eval grade to cache (A or B only)
_CACHEABLE_GRADES: frozenset[str] = frozenset({"A", "B"})

# Supabase table (created via migration if L2 is available)
_TABLE = "recipe_cache"


# ── Row type ──────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    cache_key: str
    model_recipe: dict[str, Any]
    eval_grade: str
    stored_at: float   # time.time()


# ── Bucketing helpers ─────────────────────────────────────────────────────────

def _rows_bucket(num_rows: int) -> str:
    if num_rows < 200:
        return "xs"
    if num_rows < 1_000:
        return "sm"
    if num_rows < 10_000:
        return "md"
    return "lg"


def _word_count_bucket(avg_word_count: float) -> str:
    if avg_word_count < 20:
        return "short"
    if avg_word_count < 100:
        return "medium"
    return "long"


def make_cache_key(data_profile: dict[str, Any], task_type: str) -> str:
    """
    Derive a stable, coarse cache key from a dataset profile + task type.

    Coarseness is intentional: similar datasets should hit the same bucket
    and benefit from a previously discovered recipe without requiring an LLM call.
    """
    rows_bucket  = _rows_bucket(int(data_profile.get("num_rows", 0)))
    num_classes  = int(data_profile.get("num_classes") or len(data_profile.get("label_distribution", {})) or 0)
    wc_bucket    = _word_count_bucket(float(data_profile.get("avg_word_count", 0)))
    noise_tier   = "noisy" if float(data_profile.get("label_noise_estimate", 0)) > 0.10 else "clean"

    key_str = f"{rows_bucket}|{num_classes}|{wc_bucket}|{noise_tier}|{task_type.lower()}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:20]


# ── Supabase L2 (optional) ────────────────────────────────────────────────────

def _supabase_client():
    """Return a Supabase client, or None if not configured."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _l2_get(cache_key: str) -> CacheEntry | None:
    """Fetch from Supabase L2. Returns None on miss or any error."""
    client = _supabase_client()
    if client is None:
        return None
    try:
        import json as _json
        resp = (
            client.table(_TABLE)
            .select("cache_key, model_recipe_json, eval_grade, created_at")
            .eq("cache_key", cache_key)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return None
        row = rows[0]
        # TTL check — reject stale entries
        created_ts = _parse_iso(row.get("created_at", ""))
        if created_ts and (time.time() - created_ts) > _CACHE_TTL_SECS:
            logger.debug("RecipeCache L2: stale entry for key %s (TTL expired)", cache_key[:8])
            return None
        recipe = _json.loads(row["model_recipe_json"])
        return CacheEntry(
            cache_key=cache_key,
            model_recipe=recipe,
            eval_grade=row["eval_grade"],
            stored_at=created_ts or time.time(),
        )
    except Exception as exc:
        logger.debug("RecipeCache L2 get error: %s", exc)
        return None


def _l2_set(entry: CacheEntry) -> None:
    """Write to Supabase L2. Silently ignores any error."""
    client = _supabase_client()
    if client is None:
        return
    try:
        import json as _json
        client.table(_TABLE).upsert({
            "cache_key":         entry.cache_key,
            "model_recipe_json": _json.dumps(entry.model_recipe),
            "eval_grade":        entry.eval_grade,
        }, on_conflict="cache_key").execute()
    except Exception as exc:
        logger.debug("RecipeCache L2 set error: %s", exc)


def _parse_iso(ts_str: str) -> float | None:
    """Parse ISO-8601 timestamp → Unix epoch float. Returns None on failure."""
    if not ts_str:
        return None
    try:
        from datetime import datetime, timezone
        # Supabase returns e.g. "2026-05-30T12:34:56.789Z"
        dt = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


# ── RecipeCache (public interface) ────────────────────────────────────────────

class RecipeCache:
    """
    Two-tier recipe cache.

    L1: OrderedDict LRU (up to _L1_MAX_ENTRIES entries, process-scoped).
    L2: Supabase table (optional, persistent across restarts, 30-day TTL).

    Thread-safety: L1 is not thread-safe. For the async agent pipeline,
    all access happens in the asyncio event loop so a lock isn't needed.
    """

    def __init__(self) -> None:
        # OrderedDict for LRU eviction: most recently used moves to the end
        self._l1: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, data_profile: dict[str, Any], task_type: str) -> CacheEntry | None:
        """
        Look up a cached recipe.

        Returns a CacheEntry on hit (valid, non-expired, grade ≥ B).
        Returns None on miss, expired TTL, or Supabase error.
        """
        key = make_cache_key(data_profile, task_type)

        # ── L1 hit ────────────────────────────────────────────────────────────
        if key in self._l1:
            entry = self._l1[key]
            if (time.time() - entry.stored_at) > _CACHE_TTL_SECS:
                del self._l1[key]
                logger.debug("RecipeCache L1: evicted stale key %s", key[:8])
            else:
                self._l1.move_to_end(key)  # mark as most recently used
                logger.info("RecipeCache L1 HIT for key %s (grade %s)", key[:8], entry.eval_grade)
                return entry

        # ── L2 hit (promote to L1) ────────────────────────────────────────────
        try:
            entry = _l2_get(key)
        except Exception as exc:
            logger.debug("RecipeCache L2 get error (L1-only fallback): %s", exc)
            entry = None
        if entry is not None:
            self._set_l1(key, entry)
            logger.info("RecipeCache L2 HIT for key %s (grade %s)", key[:8], entry.eval_grade)
            return entry

        return None

    def set(
        self,
        data_profile: dict[str, Any],
        task_type: str,
        model_recipe: dict[str, Any],
        eval_grade: str,
    ) -> None:
        """
        Store a recipe+grade pair.

        Only caches if eval_grade is in _CACHEABLE_GRADES (A or B).
        Silently ignores uncacheable grades so callers don't need to check.
        """
        if eval_grade not in _CACHEABLE_GRADES:
            logger.debug(
                "RecipeCache: grade %s below threshold — not caching", eval_grade
            )
            return

        key = make_cache_key(data_profile, task_type)
        entry = CacheEntry(
            cache_key=key,
            model_recipe=model_recipe,
            eval_grade=eval_grade,
            stored_at=time.time(),
        )
        self._set_l1(key, entry)
        try:
            _l2_set(entry)
        except Exception as exc:
            logger.debug("RecipeCache L2 set error (L1-only write): %s", exc)
        logger.info("RecipeCache: stored recipe for key %s (grade %s)", key[:8], eval_grade)

    def invalidate(self, data_profile: dict[str, Any], task_type: str) -> None:
        """Remove a cache entry (e.g., if the cached model is no longer available)."""
        key = make_cache_key(data_profile, task_type)
        self._l1.pop(key, None)

    def clear(self) -> None:
        """Clear the L1 cache (does not affect L2/Supabase)."""
        self._l1.clear()

    def size(self) -> int:
        """Return the current L1 cache size."""
        return len(self._l1)

    # ── Private ───────────────────────────────────────────────────────────────

    def _set_l1(self, key: str, entry: CacheEntry) -> None:
        if key in self._l1:
            self._l1.move_to_end(key)
        self._l1[key] = entry
        # Evict the oldest entry when at capacity
        while len(self._l1) > _L1_MAX_ENTRIES:
            evicted_key, _ = self._l1.popitem(last=False)
            logger.debug("RecipeCache L1: evicted LRU key %s", evicted_key[:8])


# Module-level singleton shared across the agent process
recipe_cache = RecipeCache()
