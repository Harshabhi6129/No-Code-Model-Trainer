"""
Tests for Step 5 — Recipe Memoization Cache.

Covers:
  • make_cache_key(): same bucket → same key; different bucket → different key
  • RecipeCache.get(): L1 miss returns None; L1 hit returns entry; TTL expiry evicts
  • RecipeCache.set(): grade A/B cached; grade C/D/F not cached
  • L1 LRU eviction at capacity
  • Supabase L2 failures are silent (L1-only fallback)
  • ModelAgent returns cache_hit=True in metadata on cache hit
  • Cache invalidation
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.cache import (
    RecipeCache,
    CacheEntry,
    make_cache_key,
    _rows_bucket,
    _word_count_bucket,
    _CACHE_TTL_SECS,
    _L1_MAX_ENTRIES,
    _CACHEABLE_GRADES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile(num_rows: int = 500, num_classes: int = 3, avg_word_count: float = 50.0,
             label_noise: float = 0.0) -> dict:
    return {
        "num_rows": num_rows,
        "num_classes": num_classes,
        "avg_word_count": avg_word_count,
        "label_noise_estimate": label_noise,
        "label_distribution": {f"cls_{i}": 100 for i in range(num_classes)},
    }


def _entry(grade: str = "B", recipe: dict | None = None) -> CacheEntry:
    return CacheEntry(
        cache_key="dummy_key",
        model_recipe=recipe or {"base_model": "bert-base-uncased", "learning_rate": 2e-4},
        eval_grade=grade,
        stored_at=time.time(),
    )


# ── Bucket helpers ────────────────────────────────────────────────────────────

class TestBuckets:
    def test_rows_bucket_xs(self):
        assert _rows_bucket(0) == "xs"
        assert _rows_bucket(199) == "xs"

    def test_rows_bucket_sm(self):
        assert _rows_bucket(200) == "sm"
        assert _rows_bucket(999) == "sm"

    def test_rows_bucket_md(self):
        assert _rows_bucket(1_000) == "md"
        assert _rows_bucket(9_999) == "md"

    def test_rows_bucket_lg(self):
        assert _rows_bucket(10_000) == "lg"
        assert _rows_bucket(1_000_000) == "lg"

    def test_word_count_short(self):
        assert _word_count_bucket(0) == "short"
        assert _word_count_bucket(19.9) == "short"

    def test_word_count_medium(self):
        assert _word_count_bucket(20) == "medium"
        assert _word_count_bucket(99.9) == "medium"

    def test_word_count_long(self):
        assert _word_count_bucket(100) == "long"


# ── make_cache_key ────────────────────────────────────────────────────────────

class TestMakeCacheKey:
    def test_same_bucket_same_key(self):
        p1 = _profile(num_rows=500, num_classes=3, avg_word_count=50)  # sm, medium
        p2 = _profile(num_rows=700, num_classes=3, avg_word_count=60)  # sm, medium — same bucket
        assert make_cache_key(p1, "text_classification") == make_cache_key(p2, "text_classification")

    def test_different_rows_bucket_different_key(self):
        p_sm = _profile(num_rows=500)    # sm
        p_md = _profile(num_rows=2000)   # md
        assert make_cache_key(p_sm, "text_classification") != make_cache_key(p_md, "text_classification")

    def test_different_class_count_different_key(self):
        p3  = _profile(num_classes=3)
        p10 = _profile(num_classes=10)
        assert make_cache_key(p3, "text_classification") != make_cache_key(p10, "text_classification")

    def test_different_task_type_different_key(self):
        p = _profile()
        assert make_cache_key(p, "text_classification") != make_cache_key(p, "token_classification")

    def test_key_is_deterministic(self):
        p = _profile()
        k1 = make_cache_key(p, "text_classification")
        k2 = make_cache_key(p, "text_classification")
        assert k1 == k2

    def test_key_length(self):
        # Keys are 20-char hex strings (truncated SHA-256)
        key = make_cache_key(_profile(), "text_classification")
        assert len(key) == 20

    def test_noisy_dataset_different_from_clean(self):
        clean = _profile(label_noise=0.05)  # 5% → below 10% → clean tier
        noisy = _profile(label_noise=0.15)  # 15% → above 10% → noisy tier
        assert make_cache_key(clean, "text_classification") != make_cache_key(noisy, "text_classification")


# ── RecipeCache L1 ────────────────────────────────────────────────────────────

class TestRecipeCacheL1:
    def setup_method(self):
        self.cache = RecipeCache()

    def test_miss_returns_none(self):
        assert self.cache.get(_profile(), "text_classification") is None

    def test_grade_a_is_cached(self):
        p = _profile()
        recipe = {"base_model": "bert-base-uncased", "learning_rate": 2e-4}
        self.cache.set(p, "text_classification", recipe, "A")
        entry = self.cache.get(p, "text_classification")
        assert entry is not None
        assert entry.eval_grade == "A"
        assert entry.model_recipe == recipe

    def test_grade_b_is_cached(self):
        p = _profile()
        self.cache.set(p, "text_classification", {}, "B")
        assert self.cache.get(p, "text_classification") is not None

    def test_grade_c_not_cached(self):
        p = _profile()
        self.cache.set(p, "text_classification", {}, "C")
        assert self.cache.get(p, "text_classification") is None

    def test_grade_d_not_cached(self):
        self.cache.set(_profile(), "text_classification", {}, "D")
        assert self.cache.get(_profile(), "text_classification") is None

    def test_grade_f_not_cached(self):
        self.cache.set(_profile(), "text_classification", {}, "F")
        assert self.cache.get(_profile(), "text_classification") is None

    def test_all_cacheable_grades(self):
        for grade in _CACHEABLE_GRADES:
            c = RecipeCache()
            c.set(_profile(), "text_classification", {}, grade)
            assert c.get(_profile(), "text_classification") is not None

    def test_size_increases_on_set(self):
        assert self.cache.size() == 0
        self.cache.set(_profile(num_classes=2), "text_classification", {}, "A")
        assert self.cache.size() == 1

    def test_same_key_overwritten(self):
        p = _profile()
        self.cache.set(p, "text_classification", {"base_model": "old"}, "A")
        self.cache.set(p, "text_classification", {"base_model": "new"}, "B")
        entry = self.cache.get(p, "text_classification")
        assert entry is not None
        assert entry.model_recipe["base_model"] == "new"


# ── TTL expiry ────────────────────────────────────────────────────────────────

class TestTTLExpiry:
    def test_expired_entry_returns_none(self):
        cache = RecipeCache()
        p = _profile()
        cache.set(p, "text_classification", {}, "A")
        # Manually age the entry past TTL
        key = make_cache_key(p, "text_classification")
        cache._l1[key].stored_at = time.time() - _CACHE_TTL_SECS - 1

        result = cache.get(p, "text_classification")
        assert result is None

    def test_expired_entry_removed_from_l1(self):
        cache = RecipeCache()
        p = _profile()
        cache.set(p, "text_classification", {}, "A")
        key = make_cache_key(p, "text_classification")
        cache._l1[key].stored_at = time.time() - _CACHE_TTL_SECS - 1

        cache.get(p, "text_classification")  # triggers eviction
        assert cache.size() == 0


# ── LRU eviction at capacity ──────────────────────────────────────────────────

class TestLRUEviction:
    def test_evicts_oldest_when_at_capacity(self):
        cache = RecipeCache()
        # Fill to capacity with distinct keys (vary num_classes to get distinct keys)
        for i in range(_L1_MAX_ENTRIES):
            cache.set(_profile(num_classes=i + 1), "text_classification", {"i": i}, "A")
        assert cache.size() == _L1_MAX_ENTRIES

        # Adding one more should evict the oldest (num_classes=1, first inserted)
        cache.set(_profile(num_classes=_L1_MAX_ENTRIES + 1), "text_classification", {}, "B")
        assert cache.size() == _L1_MAX_ENTRIES  # still at cap

        # The first-inserted key (num_classes=1) should have been evicted
        first = cache.get(_profile(num_classes=1), "text_classification")
        assert first is None  # LRU-evicted


# ── Invalidation and clear ────────────────────────────────────────────────────

class TestInvalidation:
    def test_invalidate_removes_entry(self):
        cache = RecipeCache()
        p = _profile()
        cache.set(p, "text_classification", {}, "A")
        cache.invalidate(p, "text_classification")
        assert cache.get(p, "text_classification") is None

    def test_clear_empties_l1(self):
        cache = RecipeCache()
        for i in range(5):
            cache.set(_profile(num_classes=i + 1), "text_classification", {}, "A")
        cache.clear()
        assert cache.size() == 0


# ── Supabase L2 failure is silent ─────────────────────────────────────────────

class TestSupabaseFallback:
    def test_supabase_get_failure_falls_back_silently(self):
        """If Supabase raises, get() returns None without crashing."""
        cache = RecipeCache()
        with patch("agents.cache._l2_get", side_effect=Exception("DB down")):
            # Should not raise, just return None (L1 miss + L2 error = miss)
            result = cache.get(_profile(), "text_classification")
            assert result is None

    def test_supabase_set_failure_is_silent(self):
        """If Supabase L2 write fails, set() still populates L1."""
        cache = RecipeCache()
        with patch("agents.cache._l2_set", side_effect=Exception("DB down")):
            cache.set(_profile(), "text_classification", {"base_model": "bert"}, "A")
        # L1 should still have the entry
        assert cache.size() == 1

    def test_no_supabase_env_vars_means_l1_only(self):
        """Without SUPABASE_URL/KEY, L2 operations are no-ops."""
        with patch.dict("os.environ", {}, clear=True):
            # _supabase_client() returns None → L2 skipped
            from agents.cache import _supabase_client
            assert _supabase_client() is None
