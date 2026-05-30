"""
Deterministic difficulty-adjusted grading rubric for ModelForge.

Research basis: AutoML-Agent uses Normalized Performance Score + Comprehensive
Score instead of raw accuracy. This module implements a similar idea:
  - adjusted_score = (raw_f1 - majority_baseline) / (1 - majority_baseline)
  - This normalizes against random chance, making grades fair across
    imbalanced/few-class/noisy datasets.

Design rules:
  - Zero LLM calls — 100% deterministic
  - EvalAgent receives RubricResult and narrates it (LLM explains WHY,
    not what — the grade itself comes from this module)
  - Grade thresholds scale with difficulty tier so harder tasks are
    rewarded more generously for the same relative improvement
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


# ── Types ──────────────────────────────────────────────────────────────────────

DifficultyTier = Literal["easy", "medium", "hard", "very_hard"]
LetterGrade    = Literal["A", "B", "C", "D", "F"]


@dataclass
class RubricResult:
    raw_f1:              float
    majority_baseline:   float
    adjusted_score:      float  # (raw_f1 - majority) / (1 - majority); -∞..1
    difficulty_tier:     DifficultyTier
    letter_grade:        LetterGrade
    grade_rationale:     str     # short deterministic explanation for Claude to narrate

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_f1":            self.raw_f1,
            "majority_baseline": self.majority_baseline,
            "adjusted_score":    self.adjusted_score,
            "difficulty_tier":   self.difficulty_tier,
            "letter_grade":      self.letter_grade,
            "grade_rationale":   self.grade_rationale,
        }


# ── Difficulty tiers ───────────────────────────────────────────────────────────
# Classification is ordered from hardest criterion to easiest.
# A dataset is placed in the HARDEST tier it qualifies for.

def _classify_tier(
    n_classes: int,
    n_rows: int,
    noise_rate: float,
    balance_ratio: float | None,  # min_count / max_count; None = unknown
) -> DifficultyTier:
    """
    Classify dataset into a difficulty tier.

    very_hard: n_classes > 15  OR  n_rows < 100  OR  noise_rate > 30%
    hard:      n_classes 6-15  OR  n_rows 100-199  OR  noise_rate 15-30%
    medium:    n_classes 3-5   OR  n_rows 200-999  OR  noise_rate 5-15%
    easy:      n_classes ≤ 2   AND n_rows ≥ 1000  AND  noise_rate < 5%
    """
    if n_classes > 15 or n_rows < 100 or noise_rate > 0.30:
        return "very_hard"
    if n_classes >= 6 or n_rows < 200 or noise_rate > 0.15:
        return "hard"
    if n_classes >= 3 or n_rows < 1000 or noise_rate > 0.05:
        return "medium"
    return "easy"


# ── Grade thresholds (on adjusted_score) ──────────────────────────────────────
# Harder tiers reward more generously for the same relative improvement.
# Thresholds are on the adjusted score (normalized against random-chance baseline).

_THRESHOLDS: dict[DifficultyTier, list[tuple[float, LetterGrade]]] = {
    "easy":      [(0.90, "A"), (0.75, "B"), (0.55, "C"), (0.35, "D")],
    "medium":    [(0.80, "A"), (0.62, "B"), (0.44, "C"), (0.26, "D")],
    "hard":      [(0.65, "A"), (0.47, "B"), (0.30, "C"), (0.15, "D")],
    "very_hard": [(0.50, "A"), (0.35, "B"), (0.20, "C"), (0.08, "D")],
}


def _grade_from_adjusted(adjusted: float, tier: DifficultyTier) -> LetterGrade:
    # Round to 8 decimal places to avoid FP edge cases (e.g. 0.8999... vs 0.90)
    adj = round(adjusted, 8)
    for threshold, grade in _THRESHOLDS[tier]:
        if adj >= threshold:
            return grade
    return "F"


# ── Majority class baseline ────────────────────────────────────────────────────

def _majority_baseline(label_distribution: dict[str, int]) -> float:
    """
    Majority class baseline F1 (macro-weighted for balanced, majority-label
    proportion for imbalanced datasets).

    The majority-class classifier always predicts the largest class.
    Its weighted F1 = majority_class_proportion (all other classes get F1=0).

    Special case: empty distribution → returns 0.0 (cannot compute baseline).
    Degenerate case: all samples same class → returns 1.0 (grade = F regardless).
    """
    if not label_distribution:
        return 0.0
    total = sum(label_distribution.values())
    if total == 0:
        return 0.0
    max_count = max(label_distribution.values())
    return max_count / total


# ── Public entry point ─────────────────────────────────────────────────────────

def compute_normalized_score(
    metrics:      dict[str, Any],
    data_profile: dict[str, Any],
) -> RubricResult:
    """
    Compute a difficulty-adjusted grade from training metrics + data profile.

    Args:
        metrics:      training_result dict (must contain "f1")
        data_profile: data profile dict

    Returns:
        RubricResult with all grading fields populated.
    """
    raw_f1 = float(metrics.get("f1") or 0.0)

    # Dataset characteristics
    n_rows    = int(data_profile.get("num_rows", 0) or 0)
    n_classes = int(
        data_profile.get("num_classes")
        or len(data_profile.get("label_distribution", {}))
        or 1
    )
    noise_rate = float(data_profile.get("label_noise_estimate", 0.0) or 0.0)
    label_dist = data_profile.get("label_distribution") or {}

    # Balance ratio: min_count / max_count (0=totally imbalanced, 1=perfect balance)
    if label_dist and len(label_dist) > 1:
        counts = list(label_dist.values())
        mn, mx = min(counts), max(counts)
        balance_ratio = mn / mx if mx > 0 else None
    else:
        balance_ratio = None

    tier = _classify_tier(n_classes, n_rows, noise_rate, balance_ratio)
    majority = _majority_baseline(label_dist)

    # Adjusted score — normalised against random-chance baseline
    # Clamp to [-2.0, 1.0] to keep it bounded
    if abs(1.0 - majority) < 1e-9:
        # Degenerate: all samples same class → majority = 1.0 → baseline = trivial
        adjusted = 0.0
    else:
        adjusted = (raw_f1 - majority) / (1.0 - majority)
        adjusted = max(-2.0, min(1.0, adjusted))

    letter = _grade_from_adjusted(adjusted, tier)

    # Degenerate dataset override
    if majority >= 0.999:
        letter = "F"

    # Build deterministic rationale for EvalAgent to narrate
    rationale = _build_rationale(raw_f1, majority, adjusted, tier, n_rows, n_classes, noise_rate)

    return RubricResult(
        raw_f1=round(raw_f1, 4),
        majority_baseline=round(majority, 4),
        adjusted_score=round(adjusted, 4),
        difficulty_tier=tier,
        letter_grade=letter,
        grade_rationale=rationale,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_rationale(
    raw_f1: float,
    majority: float,
    adjusted: float,
    tier: DifficultyTier,
    n_rows: int,
    n_classes: int,
    noise_rate: float,
) -> str:
    """Build a concise, factual rationale string for Claude to narrate."""
    parts: list[str] = []

    tier_label = tier.replace("_", " ").capitalize()
    parts.append(f"Difficulty: {tier_label} ({n_rows} rows, {n_classes} classes).")

    if majority >= 0.999:
        parts.append("Dataset is degenerate (all samples in one class) — grade F.")
    else:
        parts.append(
            f"Majority baseline F1: {majority:.3f}. "
            f"Adjusted score: {adjusted:.3f} "
            f"(normalized against random chance)."
        )

    if noise_rate >= 0.20:
        parts.append(
            f"High label noise ({noise_rate*100:.0f}%) severely limits achievable F1 — "
            "reported metrics are a lower bound."
        )
    elif noise_rate >= 0.05:
        parts.append(
            f"Moderate label noise ({noise_rate*100:.0f}%) may limit F1 ceiling."
        )

    if raw_f1 < majority:
        parts.append(
            "Model performs worse than the majority-class baseline — "
            "training may not have converged."
        )

    return " ".join(parts)
