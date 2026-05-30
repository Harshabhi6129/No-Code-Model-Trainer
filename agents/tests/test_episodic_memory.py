"""
Tests for Step 10 — Episodic Memory.

Covers:
  • _feature_vector(): bounded [0,1], correct dimensionality
  • _cosine_similarity(): identical vectors → 1.0; orthogonal → 0.0; zero → 0.0
  • EpisodicMemory.recall(): L1 miss returns []; L1 hit returns entry
  • EpisodicMemory.memorize(): grade A/B stored; grade C/D/F not stored
  • Recall threshold: similarity ≥ 0.85 → recalled; < 0.85 → not recalled
  • L2 Supabase failure is silent (L1-only fallback)
  • format_memory_exemplar(): correct format
  • Empty memory → recall returns []
  • memorize() on first run ever → no crash
"""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from agents.memory import (
    EpisodicMemory,
    MemoryEntry,
    _feature_vector,
    _cosine_similarity,
    _RECALL_THRESHOLD,
    _CACHEABLE_GRADES,
    format_memory_exemplar,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile(num_rows: int = 500, num_classes: int = 3,
             avg_word_count: float = 50.0, noise: float = 0.0,
             quality: float = 1.0) -> dict:
    return {
        "num_rows": num_rows,
        "num_classes": num_classes,
        "avg_word_count": avg_word_count,
        "label_noise_estimate": noise,
        "text_quality_score": quality,
        "label_distribution": {f"cls_{i}": 100 for i in range(num_classes)},
    }


def _entry(grade: str = "B", f1: float = 0.85, task_type: str = "text_classification") -> MemoryEntry:
    fv = _feature_vector(_profile())
    return MemoryEntry(
        feature_vector=fv,
        task_type=task_type,
        model_recipe={"base_model": "bert-base-uncased", "learning_rate": 2e-4},
        eval_grade=grade,
        eval_f1=f1,
    )


# ── Feature vector ────────────────────────────────────────────────────────────

class TestFeatureVector:
    def test_length_is_5(self):
        fv = _feature_vector(_profile())
        assert len(fv) == 5

    def test_all_values_bounded_0_1(self):
        fv = _feature_vector(_profile(num_rows=1_000_000, num_classes=100, avg_word_count=9999))
        for v in fv:
            assert 0.0 <= v <= 1.0, f"Feature out of range: {v}"

    def test_zero_rows_no_crash(self):
        fv = _feature_vector(_profile(num_rows=0))
        assert isinstance(fv, list)
        assert len(fv) == 5

    def test_large_dataset_has_larger_first_feature(self):
        fv_small = _feature_vector(_profile(num_rows=100))
        fv_large = _feature_vector(_profile(num_rows=10_000))
        assert fv_large[0] > fv_small[0]

    def test_more_classes_has_larger_second_feature(self):
        fv_2  = _feature_vector(_profile(num_classes=2))
        fv_20 = _feature_vector(_profile(num_classes=20))
        assert fv_20[1] > fv_2[1]


# ── Cosine similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_is_one(self):
        v = [0.5, 0.3, 0.8, 0.1, 0.9]
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_is_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_is_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similar_vectors_high_score(self):
        a = [0.5, 0.4, 0.6, 0.1, 0.9]
        b = [0.5, 0.4, 0.6, 0.1, 0.9]
        assert _cosine_similarity(a, b) > 0.99

    def test_different_vectors_lower_score(self):
        a = [1.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 0.0, 1.0]
        sim = _cosine_similarity(a, b)
        assert sim < 0.5


# ── EpisodicMemory ────────────────────────────────────────────────────────────

class TestEpisodicMemory:
    def setup_method(self):
        self.mem = EpisodicMemory()

    def test_empty_memory_returns_empty_list(self):
        result = self.mem.recall(_profile(), "text_classification")
        assert result == []

    def test_grade_a_is_stored(self):
        p = _profile()
        self.mem.memorize(p, "text_classification", {}, "A", 0.92)
        assert self.mem.size() == 1

    def test_grade_b_is_stored(self):
        p = _profile()
        self.mem.memorize(p, "text_classification", {}, "B", 0.85)
        assert self.mem.size() == 1

    def test_grade_c_not_stored(self):
        p = _profile()
        self.mem.memorize(p, "text_classification", {}, "C", 0.72)
        assert self.mem.size() == 0

    def test_grade_d_not_stored(self):
        self.mem.memorize(_profile(), "text_classification", {}, "D", 0.50)
        assert self.mem.size() == 0

    def test_grade_f_not_stored(self):
        self.mem.memorize(_profile(), "text_classification", {}, "F", 0.30)
        assert self.mem.size() == 0

    def test_identical_profile_recalled(self):
        p = _profile(num_rows=500, num_classes=3, avg_word_count=50)
        recipe = {"base_model": "bert-base-uncased", "learning_rate": 2e-4}
        self.mem.memorize(p, "text_classification", recipe, "A", 0.91)

        results = self.mem.recall(p, "text_classification")
        assert len(results) == 1
        assert results[0].eval_grade == "A"
        assert results[0].model_recipe == recipe

    def test_very_different_profile_not_recalled(self):
        """Large 100-class dataset vs tiny 2-class dataset → below threshold."""
        p_stored = _profile(num_rows=10_000, num_classes=50, avg_word_count=200)
        p_query  = _profile(num_rows=50, num_classes=2, avg_word_count=10)
        self.mem.memorize(p_stored, "text_classification", {}, "A", 0.90)
        results = self.mem.recall(p_query, "text_classification")
        assert results == []

    def test_different_task_type_not_recalled(self):
        p = _profile()
        self.mem.memorize(p, "text_classification", {}, "A", 0.90)
        results = self.mem.recall(p, "token_classification")
        assert results == []

    def test_top_k_limits_results(self):
        p = _profile()
        for i in range(5):
            self.mem.memorize(p, "text_classification", {}, "A", 0.80 + i * 0.02)
        results = self.mem.recall(p, "text_classification", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_f1_desc(self):
        p = _profile()
        self.mem.memorize(p, "text_classification", {"lr": 1}, "B", 0.80)
        self.mem.memorize(p, "text_classification", {"lr": 2}, "A", 0.92)
        self.mem.memorize(p, "text_classification", {"lr": 3}, "A", 0.88)
        results = self.mem.recall(p, "text_classification")
        f1_scores = [e.eval_f1 for e in results]
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_clear_empties_l1(self):
        self.mem.memorize(_profile(), "text_classification", {}, "A", 0.9)
        self.mem.clear()
        assert self.mem.size() == 0

    def test_supabase_failure_silent_on_recall(self):
        """L2 errors must not propagate to the caller."""
        with patch("agents.memory._l2_recall", side_effect=Exception("DB error")):
            result = self.mem.recall(_profile(), "text_classification")
            assert result == []

    def test_supabase_failure_silent_on_memorize(self):
        """L2 write errors must not propagate."""
        with patch("agents.memory._l2_memorize", side_effect=Exception("DB error")):
            self.mem.memorize(_profile(), "text_classification", {}, "A", 0.90)
            assert self.mem.size() == 1  # L1 write succeeded


# ── format_memory_exemplar ────────────────────────────────────────────────────

class TestFormatMemoryExemplar:
    def test_contains_grade(self):
        entry = _entry(grade="A", f1=0.91)
        out = format_memory_exemplar(entry)
        assert "A" in out

    def test_contains_f1(self):
        entry = _entry(f1=0.876)
        out = format_memory_exemplar(entry)
        assert "0.876" in out

    def test_contains_base_model(self):
        entry = _entry()
        out = format_memory_exemplar(entry)
        assert "bert-base-uncased" in out

    def test_is_string(self):
        out = format_memory_exemplar(_entry())
        assert isinstance(out, str)
        assert len(out) > 20
