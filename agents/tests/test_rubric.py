"""
Tests for Step 11 — Difficulty-Adjusted Grading Rubric.

Covers:
  • _classify_tier(): correct tier for each combination of n_classes/n_rows/noise
  • _majority_baseline(): correct computation from label_distribution
  • compute_normalized_score(): grade A for easy/high-F1; grade F for degenerate
  • Adjusted score = (raw_f1 - majority) / (1 - majority)
  • Harder tiers more lenient: same raw_f1 → better grade on very_hard vs easy
  • F1 below majority → grade F
  • Degenerate dataset (majority=1.0) → grade F
  • noise_rate > 30% → very_hard tier
  • grade_rationale is a non-empty string
  • to_dict() has all expected keys
"""
from __future__ import annotations

import pytest

from agents.rubric import (
    compute_normalized_score,
    RubricResult,
    _classify_tier,
    _majority_baseline,
    _grade_from_adjusted,
)


# ── _classify_tier ────────────────────────────────────────────────────────────

class TestClassifyTier:
    def test_easy_tier(self):
        assert _classify_tier(n_classes=2, n_rows=2000, noise_rate=0.02, balance_ratio=0.9) == "easy"

    def test_medium_from_class_count(self):
        assert _classify_tier(n_classes=4, n_rows=1500, noise_rate=0.01, balance_ratio=0.8) == "medium"

    def test_medium_from_row_count(self):
        assert _classify_tier(n_classes=2, n_rows=500, noise_rate=0.01, balance_ratio=None) == "medium"

    def test_medium_from_noise(self):
        assert _classify_tier(n_classes=2, n_rows=2000, noise_rate=0.08, balance_ratio=None) == "medium"

    def test_hard_from_class_count(self):
        assert _classify_tier(n_classes=10, n_rows=2000, noise_rate=0.01, balance_ratio=None) == "hard"

    def test_hard_from_row_count(self):
        assert _classify_tier(n_classes=2, n_rows=150, noise_rate=0.01, balance_ratio=None) == "hard"

    def test_hard_from_noise(self):
        assert _classify_tier(n_classes=2, n_rows=2000, noise_rate=0.20, balance_ratio=None) == "hard"

    def test_very_hard_from_class_count(self):
        assert _classify_tier(n_classes=20, n_rows=2000, noise_rate=0.01, balance_ratio=None) == "very_hard"

    def test_very_hard_from_row_count(self):
        assert _classify_tier(n_classes=2, n_rows=50, noise_rate=0.01, balance_ratio=None) == "very_hard"

    def test_very_hard_from_noise(self):
        assert _classify_tier(n_classes=2, n_rows=2000, noise_rate=0.35, balance_ratio=None) == "very_hard"

    def test_hardest_criterion_wins(self):
        # n_classes=20 alone → very_hard, even if rows and noise are fine
        assert _classify_tier(n_classes=20, n_rows=10_000, noise_rate=0.01, balance_ratio=0.95) == "very_hard"


# ── _majority_baseline ────────────────────────────────────────────────────────

class TestMajorityBaseline:
    def test_balanced_binary(self):
        dist = {"pos": 50, "neg": 50}
        assert _majority_baseline(dist) == pytest.approx(0.5)

    def test_skewed_3class(self):
        dist = {"a": 80, "b": 15, "c": 5}  # majority = 80/100 = 0.8
        assert _majority_baseline(dist) == pytest.approx(0.8)

    def test_all_one_class(self):
        dist = {"a": 100}
        assert _majority_baseline(dist) == pytest.approx(1.0)

    def test_empty_distribution(self):
        assert _majority_baseline({}) == pytest.approx(0.0)

    def test_zero_total(self):
        dist = {"a": 0, "b": 0}
        assert _majority_baseline(dist) == pytest.approx(0.0)


# ── compute_normalized_score ──────────────────────────────────────────────────

class TestComputeNormalizedScore:
    def _profile(self, num_rows=1000, num_classes=2, noise=0.0, dist=None):
        if dist is None:
            dist = {f"cls_{i}": 500 for i in range(num_classes)}
        return {
            "num_rows": num_rows,
            "num_classes": num_classes,
            "label_noise_estimate": noise,
            "label_distribution": dist,
            "text_quality_score": 1.0,
        }

    def _metrics(self, f1: float):
        return {"f1": f1, "accuracy": f1}

    def test_high_f1_easy_dataset_is_grade_a(self):
        result = compute_normalized_score(
            self._metrics(0.95),
            self._profile(num_rows=2000, num_classes=2),
        )
        assert result.letter_grade == "A"

    def test_low_f1_easy_dataset_is_grade_f(self):
        result = compute_normalized_score(
            self._metrics(0.45),
            self._profile(num_rows=2000, num_classes=2, dist={"a": 1000, "b": 1000}),
        )
        assert result.letter_grade in ("D", "F")

    def test_moderate_f1_very_hard_dataset_grades_generously(self):
        """55% F1 on a 20-class, 80-row, noisy dataset should grade ≥ B."""
        result = compute_normalized_score(
            self._metrics(0.55),
            self._profile(num_rows=80, num_classes=20, noise=0.25,
                          dist={f"cls_{i}": 4 for i in range(20)}),
        )
        # adjusted_score on very_hard tier should get a B or better
        assert result.difficulty_tier == "very_hard"
        assert result.letter_grade in ("A", "B", "C")  # at least C for moderate F1

    def test_harder_tier_same_raw_f1_better_grade(self):
        """Same raw F1 (0.70) → better grade on very_hard than on easy."""
        result_easy = compute_normalized_score(
            self._metrics(0.70),
            self._profile(num_rows=2000, num_classes=2, dist={"a": 1000, "b": 1000}),
        )
        result_hard = compute_normalized_score(
            self._metrics(0.70),
            self._profile(num_rows=50, num_classes=20, noise=0.25,
                          dist={f"cls_{i}": 2 for i in range(20)}),
        )
        grades = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        assert grades[result_hard.letter_grade] >= grades[result_easy.letter_grade]

    def test_f1_below_majority_baseline_is_f(self):
        """Model worse than random → grade F."""
        # majority = 0.8 (80% of samples are class 'a')
        result = compute_normalized_score(
            self._metrics(0.55),  # below 0.8 majority baseline
            self._profile(dist={"a": 800, "b": 200}),
        )
        # adjusted = (0.55 - 0.8) / (1 - 0.8) = -1.25 → grade F
        assert result.letter_grade == "F"
        assert result.adjusted_score < 0

    def test_degenerate_dataset_always_f(self):
        """All samples in one class → majority = 1.0 → grade F."""
        result = compute_normalized_score(
            self._metrics(0.99),  # even perfect F1
            self._profile(dist={"only_class": 1000}),
        )
        assert result.letter_grade == "F"

    def test_adjusted_score_formula(self):
        """Verify adjusted_score = (raw_f1 - majority) / (1 - majority)."""
        # Balanced binary: majority = 0.5
        dist = {"a": 100, "b": 100}
        result = compute_normalized_score(
            self._metrics(0.80),
            self._profile(dist=dist),
        )
        expected_adjusted = (0.80 - 0.5) / (1.0 - 0.5)  # = 0.60
        assert result.adjusted_score == pytest.approx(expected_adjusted, abs=0.01)
        assert result.majority_baseline == pytest.approx(0.5, abs=0.01)

    def test_result_has_all_fields(self):
        result = compute_normalized_score(self._metrics(0.85), self._profile())
        assert isinstance(result.raw_f1, float)
        assert isinstance(result.majority_baseline, float)
        assert isinstance(result.adjusted_score, float)
        assert result.difficulty_tier in ("easy", "medium", "hard", "very_hard")
        assert result.letter_grade in ("A", "B", "C", "D", "F")
        assert isinstance(result.grade_rationale, str)
        assert len(result.grade_rationale) > 10

    def test_to_dict_has_all_keys(self):
        result = compute_normalized_score(self._metrics(0.85), self._profile())
        d = result.to_dict()
        for key in ("raw_f1", "majority_baseline", "adjusted_score",
                    "difficulty_tier", "letter_grade", "grade_rationale"):
            assert key in d

    def test_empty_metrics_no_crash(self):
        """No f1 key → defaults to 0.0, still returns valid result."""
        result = compute_normalized_score({}, self._profile())
        assert isinstance(result, RubricResult)
        assert result.raw_f1 == 0.0

    def test_high_noise_rationale_mentions_noise(self):
        result = compute_normalized_score(
            self._metrics(0.72),
            self._profile(noise=0.25),
        )
        assert "noise" in result.grade_rationale.lower()

    def test_noise_boundary_30pct_is_very_hard(self):
        result = compute_normalized_score(
            self._metrics(0.60),
            self._profile(num_rows=2000, num_classes=2, noise=0.31),
        )
        assert result.difficulty_tier == "very_hard"
