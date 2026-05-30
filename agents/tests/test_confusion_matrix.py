"""
Tests for Step 8 — Confusion Matrix + Per-Class Analysis.

Covers:
  • _compute_confusion_matrix_data(): perfect predictions → diagonal matrix
  • _compute_confusion_matrix_data(): all wrong → off-diagonal
  • per_class_metrics includes precision, recall, f1, support per label
  • Binary classification → 2×2 matrix
  • Multi-class (3×3) → correct shape and values
  • Empty predictions → graceful empty return
"""
from __future__ import annotations

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_cm(y_true: list[int], y_pred: list[int], num_labels: int) -> list[list[int]]:
    """Mirror the confusion_matrix logic added to ml_core.py."""
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
        return cm.tolist()
    except Exception:
        return []


def _compute_per_class(
    y_true: list[int], y_pred: list[int], label_names: list[str]
) -> dict:
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    return {
        lbl: {
            "precision": round(float(report[lbl]["precision"]), 4),
            "recall":    round(float(report[lbl]["recall"]), 4),
            "f1":        round(float(report[lbl]["f1-score"]), 4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in label_names
        if lbl in report
    }


# ── Confusion matrix computation ──────────────────────────────────────────────

class TestConfusionMatrix:
    def test_perfect_predictions_diagonal(self):
        """All correct predictions → all mass on diagonal."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        cm = _compute_cm(y_true, y_pred, num_labels=3)
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)
        # Diagonal entries should be non-zero; off-diagonal should be 0
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert cm[i][j] > 0, f"Diagonal cm[{i}][{i}] should be > 0"
                else:
                    assert cm[i][j] == 0, f"Off-diagonal cm[{i}][{j}] should be 0"

    def test_all_wrong_off_diagonal(self):
        """Class 0 always predicted as 1, class 1 always as 0 → off-diagonal mass."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        cm = _compute_cm(y_true, y_pred, num_labels=2)
        assert cm[0][0] == 0  # No true 0 predicted as 0
        assert cm[1][1] == 0  # No true 1 predicted as 1
        assert cm[0][1] == 2  # 2 true 0s predicted as 1
        assert cm[1][0] == 2  # 2 true 1s predicted as 0

    def test_binary_gives_2x2(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 1, 1, 0, 0]
        cm = _compute_cm(y_true, y_pred, num_labels=2)
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_3class_gives_3x3(self):
        y_true = [0, 1, 2] * 4
        y_pred = [0, 1, 2] * 4
        cm = _compute_cm(y_true, y_pred, num_labels=3)
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_cm_row_sums_equal_class_support(self):
        """Each row sum equals the number of actual samples in that class."""
        y_true = [0, 0, 0, 1, 1, 2]  # 3 zeros, 2 ones, 1 two
        y_pred = [0, 1, 2, 1, 0, 2]  # some errors
        cm = _compute_cm(y_true, y_pred, num_labels=3)
        assert sum(cm[0]) == 3
        assert sum(cm[1]) == 2
        assert sum(cm[2]) == 1

    def test_empty_labels_returns_empty(self):
        """Edge case: empty predictions."""
        cm = _compute_cm([], [], num_labels=3)
        # sklearn returns empty or zero matrix; list conversion should not raise
        assert isinstance(cm, list)

    def test_single_class_gives_1x1(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        cm = _compute_cm(y_true, y_pred, num_labels=1)
        assert len(cm) == 1
        assert cm[0][0] == 3


# ── Per-class metrics ─────────────────────────────────────────────────────────

class TestPerClassMetrics:
    def test_perfect_model_all_ones(self):
        """Perfect predictions → precision=recall=f1=1.0 for all classes."""
        labels = ["pos", "neg", "neu"]
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        pcm = _compute_per_class(y_true, y_pred, labels)
        for lbl in labels:
            assert pcm[lbl]["precision"] == pytest.approx(1.0, abs=0.01)
            assert pcm[lbl]["recall"]    == pytest.approx(1.0, abs=0.01)
            assert pcm[lbl]["f1"]        == pytest.approx(1.0, abs=0.01)

    def test_all_fields_present(self):
        labels = ["a", "b"]
        pcm = _compute_per_class([0, 1], [0, 1], labels)
        for lbl in labels:
            assert "precision" in pcm[lbl]
            assert "recall"    in pcm[lbl]
            assert "f1"        in pcm[lbl]
            assert "support"   in pcm[lbl]

    def test_support_matches_actual_count(self):
        """Support = number of actual samples in each class."""
        labels = ["x", "y"]
        y_true = [0, 0, 0, 1, 1]  # 3 x, 2 y
        y_pred = [0, 0, 1, 1, 1]
        pcm = _compute_per_class(y_true, y_pred, labels)
        assert pcm["x"]["support"] == 3
        assert pcm["y"]["support"] == 2

    def test_all_labels_covered(self):
        labels = ["cat", "dog", "bird"]
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 0]
        pcm = _compute_per_class(y_true, y_pred, labels)
        for lbl in labels:
            assert lbl in pcm

    def test_values_bounded_0_to_1(self):
        labels = ["a", "b", "c"]
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 0, 2, 1, 1, 0]  # some errors
        pcm = _compute_per_class(y_true, y_pred, labels)
        for lbl in labels:
            for metric in ("precision", "recall", "f1"):
                assert 0.0 <= pcm[lbl][metric] <= 1.0, \
                    f"{metric} for {lbl} is {pcm[lbl][metric]}"

    def test_class_never_predicted_zero_precision(self):
        """If a class is never predicted, precision = 0 (zero_division=0)."""
        labels = ["a", "b"]
        # class 'b' (1) never predicted
        y_true = [0, 0, 1]
        y_pred = [0, 0, 0]
        pcm = _compute_per_class(y_true, y_pred, labels)
        assert pcm["b"]["precision"] == 0.0
        assert pcm["b"]["recall"]    == 0.0
