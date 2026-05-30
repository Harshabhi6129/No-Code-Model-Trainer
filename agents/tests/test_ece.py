"""
Unit tests for _compute_ece (Phase C1) and B3 DataAgent text profiling.
No GPU or ML deps required.
"""
import csv
from pathlib import Path

import pytest

from agents.ml_core import _compute_ece


# ── ECE tests ──────────────────────────────────────────────────────────────

def test_perfect_calibration():
    """A perfectly calibrated model has ECE ≈ 0."""
    import numpy as np
    # Binary: predict with confidence exactly matching actual accuracy
    y_true   = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # All correct, near-certain confidence (softmax ≈ 0.993) → ECE should be low
    logits = np.array(
        [[5.0, 0.0]] * 5 + [[0.0, 5.0]] * 5,
        dtype=float,
    )
    ece = _compute_ece(y_true, logits)
    assert ece < 0.10, f"Expected low ECE, got {ece:.4f}"


def test_overconfident_wrong_predictions():
    """A model always wrong with high confidence should have high ECE."""
    import numpy as np
    y_true = [0, 0, 0, 0, 0]         # all class 0
    logits = np.array(
        [[0.0, 5.0]] * 5,             # always predicts class 1 with high confidence
        dtype=float,
    )
    ece = _compute_ece(y_true, logits)
    assert ece > 0.50, f"Expected high ECE for overconfident wrong model, got {ece:.4f}"


def test_ece_never_raises():
    """_compute_ece returns 0.0 gracefully on bad input."""
    ece = _compute_ece([], None)         # type: ignore
    assert ece == 0.0
    ece2 = _compute_ece([0], "invalid")  # type: ignore
    assert ece2 == 0.0


def test_ece_multiclass():
    """Multiclass ECE works correctly."""
    import numpy as np
    n = 30
    y_true = [i % 3 for i in range(n)]
    # Correct predictions with moderate confidence
    logits = []
    for label in y_true:
        row = [-1.0, -1.0, -1.0]
        row[label] = 2.0
        logits.append(row)
    ece = _compute_ece(y_true, np.array(logits))
    assert 0.0 <= ece <= 1.0


# ── B3 DataAgent text profiling tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_data_agent_word_token_stats(tmp_path: Path):
    """DataAgent should produce word count and token estimates."""
    from agents.base import AgentContext
    from agents.data import DataAgent

    path = tmp_path / "data.csv"
    rows = [
        {"text": "This is a short sentence with about eight words here", "label": "pos"},
        {"text": "Another somewhat longer piece of text that has twelve or so total words in it", "label": "neg"},
    ] * 10   # 20 rows total

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    agent = DataAgent.__new__(DataAgent)
    ctx   = AgentContext(
        run_id="test_b3", user_intent="classify",
        dataset_path=str(path),
        task_spec={"input_column": "text", "label_column": "label"},
    )
    result = await agent.run(ctx)

    assert result.success is True
    profile = ctx.data_profile
    assert profile["avg_word_count"] > 0
    assert profile["estimated_tokens_avg"] > 0
    assert profile["estimated_tokens_p95"] >= profile["estimated_tokens_avg"]
    assert 0.0 <= profile["vocabulary_richness"] <= 1.0
    assert 0.0 <= profile["text_quality_score"] <= 1.0


@pytest.mark.asyncio
async def test_data_agent_flags_html_noise(tmp_path: Path):
    """DataAgent should flag low text quality when text contains HTML tags."""
    from agents.base import AgentContext
    from agents.data import DataAgent

    path = tmp_path / "noisy.csv"
    rows = [
        {"text": f"<div class='x'>Noisy HTML content {i} <span>tag</span></div>", "label": "a"}
        for i in range(30)
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    agent = DataAgent.__new__(DataAgent)
    ctx   = AgentContext(
        run_id="test_html", user_intent="classify",
        dataset_path=str(path),
        task_spec={"input_column": "text", "label_column": "label"},
    )
    await agent.run(ctx)
    assert ctx.data_profile["text_quality_score"] < 0.80
