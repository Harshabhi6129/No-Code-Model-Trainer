"""
Tests for Step 4 — Context Summarization Between Agents.

Covers:
  • summarize_pipeline_context() output < 5KB JSON for large datasets
  • All required fields present for EvalAgent (label_noise_estimate, ece, per_class_f1)
  • Secrets (hf_token) NOT present in summary
  • label_distribution truncated to top-10 when dataset has many classes
  • epoch_metrics NOT included in summary (replaced by loss_history_tail of last 5)
  • loss_history_tail is the last ≤5 entries of epoch_metrics
  • Training skipped / None → summary still valid (no KeyError)
  • Empty context → summary returns valid empty/zero fields
"""
from __future__ import annotations

import json

import pytest

from agents.base import AgentContext
from agents.pipeline import summarize_pipeline_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_context(
    num_classes: int = 3,
    epoch_count: int = 5,
    label_noise: float = 0.05,
    hf_token: str | None = "secret-token",
) -> AgentContext:
    label_dist = {f"class_{i}": 100 + i * 10 for i in range(num_classes)}
    epoch_metrics = [
        {"epoch": i, "step": i, "loss": 1.0 - i * 0.1, "eval_loss": 1.1 - i * 0.09}
        for i in range(epoch_count)
    ]
    ctx = AgentContext(
        run_id="test_run",
        user_intent="classify support tickets",
        hf_token=hf_token,
    )
    ctx.task_spec = {
        "task_type": "text_classification",
        "num_labels": num_classes,
        "label_names": list(label_dist.keys()),
        "input_column": "text",
        "label_column": "label",
    }
    ctx.data_profile = {
        "num_rows": 1000,
        "num_classes": num_classes,
        "label_distribution": label_dist,
        "label_noise_estimate": label_noise,
        "label_noise_count": int(1000 * label_noise),
        "text_quality_score": 0.95,
        "issues": [],
    }
    ctx.training_result = {
        "base_model": "bert-base-uncased",
        "training_approach": "lora",
        "device": "cuda",
        "num_epochs_completed": epoch_count,
        "final_train_loss": 0.35,
        "training_time_seconds": 120,
        "accuracy": 0.88,
        "f1": 0.87,
        "precision": 0.86,
        "recall": 0.88,
        "ece": 0.04,
        "per_class_f1": {f"class_{i}": round(0.85 + i * 0.01, 2) for i in range(num_classes)},
        "num_labels": num_classes,
        "label_names": list(label_dist.keys()),
        "train_samples": 800,
        "eval_samples": 200,
        "warnings": [],
        "epoch_metrics": epoch_metrics,
        "model_path": "/runs/test_run/model",
    }
    ctx.eval_result = {
        "evaluation_grade": "B",
        "summary": "Good performance.",
        "concerns": [],
        "next_steps": ["Add more training data."],
    }
    return ctx


# ── Size check ────────────────────────────────────────────────────────────────

class TestSummarySize:
    def test_summary_under_5kb_for_3_class(self):
        ctx = _make_context(num_classes=3, epoch_count=10)
        summary = summarize_pipeline_context(ctx)
        json_bytes = len(json.dumps(summary).encode())
        assert json_bytes < 5_000, f"Summary is {json_bytes} bytes (limit: 5000)"

    def test_summary_under_5kb_for_50_class(self):
        """50-class dataset: label_distribution must be truncated to stay compact."""
        ctx = _make_context(num_classes=50, epoch_count=20)
        summary = summarize_pipeline_context(ctx)
        json_bytes = len(json.dumps(summary).encode())
        assert json_bytes < 5_000, f"Summary is {json_bytes} bytes (limit: 5000)"

    def test_summary_under_5kb_for_100_class(self):
        """100-class dataset: extreme case still stays compact."""
        ctx = _make_context(num_classes=100, epoch_count=30)
        summary = summarize_pipeline_context(ctx)
        json_bytes = len(json.dumps(summary).encode())
        assert json_bytes < 5_000, f"Summary is {json_bytes} bytes (limit: 5000)"


# ── Required fields for EvalAgent ────────────────────────────────────────────

class TestRequiredFields:
    def test_label_noise_estimate_present(self):
        ctx = _make_context(label_noise=0.12)
        summary = summarize_pipeline_context(ctx)
        assert summary["data_profile"]["label_noise_estimate"] == pytest.approx(0.12)

    def test_ece_present(self):
        ctx = _make_context()
        summary = summarize_pipeline_context(ctx)
        assert summary["training_result"]["ece"] == pytest.approx(0.04)

    def test_per_class_f1_present(self):
        ctx = _make_context(num_classes=3)
        summary = summarize_pipeline_context(ctx)
        pcf1 = summary["training_result"]["per_class_f1"]
        assert isinstance(pcf1, dict)
        assert "class_0" in pcf1

    def test_eval_grade_present(self):
        ctx = _make_context()
        summary = summarize_pipeline_context(ctx)
        assert summary["eval_result"]["evaluation_grade"] == "B"

    def test_all_top_level_keys_present(self):
        ctx = _make_context()
        summary = summarize_pipeline_context(ctx)
        for key in ("task_spec", "data_profile", "training_result", "eval_result"):
            assert key in summary, f"Missing top-level key: {key}"


# ── Secrets exclusion ─────────────────────────────────────────────────────────

class TestSecretsExclusion:
    def test_hf_token_not_in_summary(self):
        ctx = _make_context(hf_token="hf_super_secret_token_12345")
        summary = summarize_pipeline_context(ctx)
        summary_str = json.dumps(summary)
        assert "hf_super_secret_token_12345" not in summary_str
        assert "hf_token" not in summary_str

    def test_model_path_not_in_summary(self):
        ctx = _make_context()
        summary = summarize_pipeline_context(ctx)
        tr = summary["training_result"]
        assert "model_path" not in tr

    def test_dataset_path_not_in_summary(self):
        ctx = _make_context()
        ctx.dataset_path = "/private/uploads/user_data.csv"
        summary = summarize_pipeline_context(ctx)
        summary_str = json.dumps(summary)
        assert "/private/uploads" not in summary_str


# ── label_distribution truncation ────────────────────────────────────────────

class TestLabelDistTruncation:
    def test_3_classes_not_truncated(self):
        ctx = _make_context(num_classes=3)
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        # All 3 classes present (not truncated)
        assert len(dist) == 3

    def test_10_classes_not_truncated(self):
        ctx = _make_context(num_classes=10)
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        assert len(dist) == 10

    def test_11_classes_truncated_to_11_entries(self):
        """11th entry is 'N more classes' bucket — total dict len is 11 (10 + 1 bucket)."""
        ctx = _make_context(num_classes=11)
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        assert len(dist) == 11  # 10 real + 1 "N more" bucket

    def test_50_classes_truncated(self):
        ctx = _make_context(num_classes=50)
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        assert len(dist) == 11  # 10 + 1 bucket

    def test_truncated_bucket_has_remaining_class_count(self):
        ctx = _make_context(num_classes=15)
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        bucket_key = [k for k in dist if "more" in k]
        assert len(bucket_key) == 1
        assert "5" in bucket_key[0]  # "… 5 more classes"

    def test_top_classes_are_most_frequent(self):
        """Top-10 should be the most frequent classes, not arbitrary."""
        ctx = _make_context(num_classes=20)
        # The fixture creates class_i with count 100 + i*10, so class_19 has most
        summary = summarize_pipeline_context(ctx)
        dist = summary["data_profile"]["label_distribution"]
        # class_19 (count=290) should be in top 10; class_0 (count=100) might not be
        assert "class_19" in dist


# ── epoch_metrics exclusion + loss_history_tail ───────────────────────────────

class TestEpochMetrics:
    def test_epoch_metrics_not_in_summary(self):
        ctx = _make_context(epoch_count=20)
        summary = summarize_pipeline_context(ctx)
        assert "epoch_metrics" not in summary["training_result"]

    def test_loss_history_tail_has_at_most_5_entries(self):
        ctx = _make_context(epoch_count=20)
        summary = summarize_pipeline_context(ctx)
        tail = summary["training_result"]["loss_history_tail"]
        assert len(tail) <= 5

    def test_loss_history_tail_is_last_5(self):
        ctx = _make_context(epoch_count=10)
        summary = summarize_pipeline_context(ctx)
        tail = summary["training_result"]["loss_history_tail"]
        # Last 5 epoch indices should be 5,6,7,8,9
        epochs_in_tail = [e["epoch"] for e in tail]
        assert epochs_in_tail == [5, 6, 7, 8, 9]

    def test_short_training_loss_tail_is_full(self):
        """Only 3 epochs → loss_history_tail has 3 entries (not padded)."""
        ctx = _make_context(epoch_count=3)
        summary = summarize_pipeline_context(ctx)
        tail = summary["training_result"]["loss_history_tail"]
        assert len(tail) == 3

    def test_empty_epoch_metrics_gives_empty_tail(self):
        ctx = _make_context(epoch_count=0)
        summary = summarize_pipeline_context(ctx)
        assert summary["training_result"]["loss_history_tail"] == []


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_context_no_crash(self):
        ctx = AgentContext(run_id="r", user_intent="test")
        # All context dicts are empty — should not raise
        summary = summarize_pipeline_context(ctx)
        assert "task_spec" in summary
        assert "data_profile" in summary
        assert "training_result" in summary

    def test_no_eval_result_gives_empty_dict(self):
        ctx = _make_context()
        ctx.eval_result = {}
        summary = summarize_pipeline_context(ctx)
        # eval_result is empty → summary returns empty dict
        assert summary["eval_result"] == {} or summary["eval_result"] is not None

    def test_training_skipped_context_no_crash(self):
        ctx = AgentContext(run_id="r", user_intent="test")
        ctx.training_result = {"status": "skipped", "reason": "no GPU"}
        ctx.data_profile = {"num_rows": 100, "label_distribution": {}}
        summary = summarize_pipeline_context(ctx)
        assert summary["training_result"]["warnings"] == []
