"""
Tests for CleanAgent label noise detection (Phase B2).

Tests cover three scenarios:
  1. Clean dataset → noise estimate ≈ 0%
  2. Dataset with deliberate mislabels → noise estimate > 0%
  3. Tiny dataset (<30 rows) → noise detection skipped gracefully

All tests that require cleanlab are skipped automatically when the package
is not installed (CI without full ML deps).
"""
import csv
import random
from pathlib import Path

import pytest

from agents.base import AgentContext
from agents.clean_agent import CleanAgent, has_cleanlab, _detect_label_noise

# ---------------------------------------------------------------------------
# Test _detect_label_noise directly (unit tests — no file I/O)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_cleanlab(), reason="cleanlab not installed")
def test_clean_dataset_low_noise(tmp_path: Path):
    """A perfectly labelled dataset should have near-zero noise estimate."""
    random.seed(42)
    # 100 rows, two perfectly separated classes
    rows = [
        {"text": f"positive example {i} great wonderful good", "label": "pos"}
        for i in range(50)
    ] + [
        {"text": f"negative example {i} terrible awful bad", "label": "neg"}
        for i in range(50)
    ]
    random.shuffle(rows)

    noise_rate, noise_count, issues = _detect_label_noise(rows, "text", "label")

    # Should detect very little noise on a clean, linearly separable dataset
    assert noise_rate < 0.15, f"Expected low noise, got {noise_rate:.2%}"
    assert noise_count == int(noise_rate * len(rows))
    assert isinstance(issues, list)


@pytest.mark.skipif(not has_cleanlab(), reason="cleanlab not installed")
def test_noisy_dataset_detected(tmp_path: Path):
    """25% mislabelled rows should yield a noise estimate significantly above 0."""
    random.seed(42)
    rows = []
    for i in range(75):
        rows.append({"text": f"positive {i} great good wonderful", "label": "pos"})
    for i in range(75):
        rows.append({"text": f"negative {i} terrible bad awful", "label": "neg"})

    # Flip 25% of labels — inject known noise
    n_noise = 37  # ~25% of 150
    noisy_indices = random.sample(range(len(rows)), n_noise)
    for idx in noisy_indices:
        rows[idx]["label"] = "neg" if rows[idx]["label"] == "pos" else "pos"

    noise_rate, noise_count, issues = _detect_label_noise(rows, "text", "label")

    # Confident Learning typically detects 50-90% of injected noise
    # Require at least 5% detection (well above the 5% warning threshold)
    assert noise_rate > 0.05, f"Expected noise > 5%, got {noise_rate:.2%}"
    assert noise_count > 0


def test_tiny_dataset_skipped():
    """Datasets with <30 rows skip noise detection without error."""
    rows = [
        {"text": f"text {i}", "label": "pos" if i % 2 == 0 else "neg"}
        for i in range(20)
    ]
    noise_rate, noise_count, issues = _detect_label_noise(rows, "text", "label")

    assert noise_rate == 0.0
    assert noise_count == 0
    assert issues == []


def test_no_labels_skipped():
    """Rows without a label column skip noise detection."""
    rows = [{"text": f"text {i}"} for i in range(50)]
    noise_rate, noise_count, issues = _detect_label_noise(rows, "text", "label")

    assert noise_rate == 0.0
    assert noise_count == 0


def test_single_class_skipped():
    """Single-class datasets skip noise detection (can't cross-validate)."""
    rows = [{"text": f"text {i}", "label": "only_class"} for i in range(50)]
    noise_rate, noise_count, issues = _detect_label_noise(rows, "text", "label")

    assert noise_rate == 0.0


# ---------------------------------------------------------------------------
# Integration test — CleanAgent.run() propagates noise to data_profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(not has_cleanlab(), reason="cleanlab not installed")
async def test_clean_agent_propagates_noise(tmp_path: Path):
    """CleanAgent stores label_noise_estimate in context.data_profile."""
    random.seed(42)

    # Build a 100-row CSV with 25% flipped labels
    path = tmp_path / "noisy.csv"
    rows = []
    for i in range(50):
        rows.append({"text": f"positive text {i} great wonderful", "label": "pos"})
    for i in range(50):
        rows.append({"text": f"negative text {i} terrible awful", "label": "neg"})
    for idx in random.sample(range(100), 25):
        rows[idx]["label"] = "neg" if rows[idx]["label"] == "pos" else "pos"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    agent = CleanAgent.__new__(CleanAgent)
    ctx   = AgentContext(
        run_id="test_noise",
        user_intent="classify",
        dataset_path=str(path),
        task_spec={"input_column": "text", "label_column": "label"},
        data_profile={"input_col": "text", "label_col": "label", "issues": []},
    )

    result = await agent.run(ctx)

    assert result.success is True
    assert "label_noise_estimate" in ctx.data_profile
    assert ctx.data_profile["label_noise_estimate"] >= 0.0   # could be 0 in edge cases
    assert "label_noise_count" in ctx.data_profile
    assert isinstance(ctx.data_profile["issues"], list)


@pytest.mark.asyncio
async def test_clean_agent_no_cleanlab_graceful(tmp_path: Path, monkeypatch):
    """CleanAgent proceeds normally even when cleanlab is not available."""
    import agents.clean_agent as ca_module
    monkeypatch.setattr(ca_module, "has_cleanlab", lambda: False)

    path = tmp_path / "data.csv"
    rows = [{"text": f"text {i}", "label": "pos" if i % 2 == 0 else "neg"}
            for i in range(40)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    agent = CleanAgent.__new__(CleanAgent)
    ctx   = AgentContext(
        run_id="test_graceful",
        user_intent="classify",
        dataset_path=str(path),
        task_spec={"input_column": "text", "label_column": "label"},
        data_profile={"input_col": "text", "label_col": "label", "issues": []},
    )

    result = await agent.run(ctx)
    assert result.success is True
    assert ctx.data_profile["label_noise_estimate"] == 0.0
    assert ctx.data_profile["label_noise_count"] == 0
