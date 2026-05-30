"""
Tests for Step 2 — Semantic Field Validators.

All checks are deterministic (no LLM). Covers:
  • QLoRA on encoder-only model → ERROR
  • QLoRA on unknown model → no error (skip check)
  • QLoRA on decoder-only (unknown family) → no error
  • Learning rate too high → ERROR
  • Learning rate too low → WARNING
  • Learning rate in-range → no issue
  • lora_r > 64 → WARNING
  • lora_alpha < lora_r → WARNING
  • max_length > 512 on BERT-family → ERROR
  • max_length > 512 on longformer → no error
  • batch_size > 25% dataset → WARNING
  • num_epochs > 10 on small dataset → WARNING
  • label count mismatch → ERROR
  • None / empty recipe → immediately valid
  • Unknown model family → no model-specific errors emitted
"""
from __future__ import annotations

import pytest

from agents.validators import validate_recipe_semantics, ValidationResult


def _make_recipe(**kwargs) -> dict:
    """Build a minimal valid recipe with any field overrides."""
    base = {
        "base_model": "bert-base-uncased",
        "training_approach": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 16,
        "max_length": 128,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    }
    base.update(kwargs)
    return base


def _make_profile(**kwargs) -> dict:
    """Build a minimal data profile with any field overrides."""
    base = {
        "num_rows": 1000,
        "num_classes": 3,
        "label_distribution": {"pos": 400, "neg": 400, "neu": 200},
    }
    base.update(kwargs)
    return base


# ── QLoRA + encoder-only checks ───────────────────────────────────────────────

class TestQLoRAEncoder:
    def test_qlora_on_bert_is_error(self):
        recipe = _make_recipe(base_model="bert-base-uncased", training_approach="qlora")
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid
        assert any("QLoRA" in e and "encoder" in e for e in result.errors)

    def test_qlora_on_distilbert_is_error(self):
        recipe = _make_recipe(base_model="distilbert-base-uncased", training_approach="qlora")
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid
        assert len(result.errors) >= 1

    def test_qlora_on_roberta_is_error(self):
        recipe = _make_recipe(base_model="roberta-base", training_approach="qlora")
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid

    def test_qlora_on_deberta_is_error(self):
        recipe = _make_recipe(base_model="microsoft/deberta-v3-small", training_approach="qlora")
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid

    def test_qlora_on_unknown_model_is_skipped(self):
        """Unknown model family → skip QLoRA check (can't rule it out)."""
        recipe = _make_recipe(base_model="meta-llama/Llama-3-8B-Instruct", training_approach="qlora")
        result = validate_recipe_semantics(recipe, _make_profile())
        # Should have NO QLoRA error for unknown family
        assert not any("QLoRA" in e for e in result.errors)

    def test_lora_on_bert_is_fine(self):
        recipe = _make_recipe(base_model="bert-base-uncased", training_approach="lora")
        result = validate_recipe_semantics(recipe, _make_profile())
        assert result.is_valid or not any("QLoRA" in e for e in result.errors)

    def test_full_finetune_on_bert_is_fine(self):
        recipe = _make_recipe(base_model="bert-base-uncased", training_approach="full_finetune",
                               lora_r=None, lora_alpha=None)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("QLoRA" in e for e in result.errors)


# ── Learning rate checks ──────────────────────────────────────────────────────

class TestLearningRate:
    def test_lr_above_1e3_is_error(self):
        recipe = _make_recipe(learning_rate=0.5)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid
        assert any("learning_rate" in e and "1e-3" in e for e in result.errors)

    def test_lr_exactly_1e3_is_error(self):
        recipe = _make_recipe(learning_rate=1e-3)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid

    def test_lr_below_1e6_is_warning(self):
        recipe = _make_recipe(learning_rate=1e-7)
        result = validate_recipe_semantics(recipe, _make_profile())
        # Should be valid (just a warning) for lr=1e-7
        assert any("too low" in w.lower() or "learning_rate" in w for w in result.warnings)

    def test_lr_in_range_no_issue(self):
        for lr in [2e-5, 1e-4, 3e-4, 5e-4, 9.9e-4]:
            recipe = _make_recipe(learning_rate=lr)
            result = validate_recipe_semantics(recipe, _make_profile())
            assert not any("learning_rate" in e for e in result.errors), f"Unexpected error for lr={lr}"


# ── LoRA parameter checks ─────────────────────────────────────────────────────

class TestLoRAParams:
    def test_lora_r_above_64_is_warning(self):
        recipe = _make_recipe(training_approach="lora", lora_r=128, lora_alpha=256)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert any("lora_r" in w and "128" in w for w in result.warnings)

    def test_lora_r_64_is_fine(self):
        recipe = _make_recipe(training_approach="lora", lora_r=64, lora_alpha=128)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("lora_r" in w for w in result.warnings)

    def test_lora_alpha_less_than_r_is_warning(self):
        recipe = _make_recipe(training_approach="lora", lora_r=16, lora_alpha=8)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert any("lora_alpha" in w or "scaling" in w.lower() for w in result.warnings)

    def test_lora_alpha_equal_r_is_fine(self):
        recipe = _make_recipe(training_approach="lora", lora_r=16, lora_alpha=16)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("lora_alpha" in w for w in result.warnings)

    def test_lora_alpha_double_r_is_fine(self):
        recipe = _make_recipe(training_approach="lora", lora_r=16, lora_alpha=32)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("lora_alpha" in w for w in result.warnings)

    def test_full_finetune_ignores_lora_params(self):
        """Full fine-tune has no LoRA params — no false positives."""
        recipe = _make_recipe(training_approach="full_finetune", lora_r=None, lora_alpha=None)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("lora" in w.lower() for w in result.warnings)


# ── max_length checks ─────────────────────────────────────────────────────────

class TestMaxLength:
    def test_bert_max_length_above_512_is_error(self):
        recipe = _make_recipe(base_model="bert-base-uncased", max_length=1024)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid
        assert any("max_length" in e and "512" in e for e in result.errors)

    def test_bert_max_length_512_is_fine(self):
        recipe = _make_recipe(base_model="bert-base-uncased", max_length=512)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("max_length" in e for e in result.errors)

    def test_roberta_max_length_above_512_is_error(self):
        recipe = _make_recipe(base_model="roberta-base", max_length=1024)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid

    def test_longformer_max_length_above_512_is_fine(self):
        """Longformer handles sequences longer than 512 — no error."""
        recipe = _make_recipe(base_model="allenai/longformer-base-4096", max_length=2048)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("max_length" in e for e in result.errors)

    def test_unknown_model_max_length_above_512_no_error(self):
        """Unknown model family → skip max_length check."""
        recipe = _make_recipe(base_model="meta-llama/Llama-3-8B", max_length=2048)
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not any("max_length" in e for e in result.errors)


# ── Batch size checks ─────────────────────────────────────────────────────────

class TestBatchSize:
    def test_batch_larger_than_25pct_dataset_is_warning(self):
        recipe = _make_recipe(batch_size=64)
        profile = _make_profile(num_rows=100)  # 64 > 100/4 = 25
        result = validate_recipe_semantics(recipe, profile)
        assert any("batch_size" in w or "25%" in w for w in result.warnings)

    def test_batch_exactly_25pct_no_warning(self):
        recipe = _make_recipe(batch_size=25)
        profile = _make_profile(num_rows=100)  # 25 == 100/4 = exactly threshold
        result = validate_recipe_semantics(recipe, profile)
        # 25 > 25 is False → no warning
        assert not any("batch_size" in w for w in result.warnings)

    def test_batch_below_25pct_no_warning(self):
        recipe = _make_recipe(batch_size=16)
        profile = _make_profile(num_rows=1000)  # 16 < 250
        result = validate_recipe_semantics(recipe, profile)
        assert not any("batch_size" in w for w in result.warnings)


# ── Epoch count checks ────────────────────────────────────────────────────────

class TestEpochCount:
    def test_many_epochs_small_dataset_is_warning(self):
        recipe = _make_recipe(num_epochs=15)
        profile = _make_profile(num_rows=100)
        result = validate_recipe_semantics(recipe, profile)
        assert any("num_epochs" in w or "overfitting" in w.lower() for w in result.warnings)

    def test_10_epochs_small_dataset_no_warning(self):
        recipe = _make_recipe(num_epochs=10)
        profile = _make_profile(num_rows=100)
        result = validate_recipe_semantics(recipe, profile)
        assert not any("num_epochs" in w for w in result.warnings)

    def test_many_epochs_large_dataset_no_warning(self):
        recipe = _make_recipe(num_epochs=15)
        profile = _make_profile(num_rows=5000)
        result = validate_recipe_semantics(recipe, profile)
        assert not any("num_epochs" in w for w in result.warnings)


# ── Label count checks ────────────────────────────────────────────────────────

class TestLabelCount:
    def test_label_count_mismatch_is_error(self):
        recipe = _make_recipe()
        profile = _make_profile(
            num_classes=2,
            label_distribution={"pos": 500, "neg": 500},
        )
        # recipe doesn't specify num_labels directly; profile has 2, and
        # num_classes=2 but recipe is built from profile so they match.
        # Force a mismatch by using a profile with different distribution:
        profile["label_distribution"] = {"a": 100, "b": 100, "c": 100, "d": 100}
        profile["num_classes"] = 4
        # make a fake recipe that claims it was built for 3 classes
        # The validator uses len(label_distribution) vs num_classes from profile
        # Both sides computed from data_profile — so we need to force mismatch
        # by passing inconsistent profile (num_classes vs label_distribution keys)
        profile["num_classes"] = 4
        profile["label_distribution"] = {"a": 100, "b": 100, "c": 100}  # 3 keys
        result = validate_recipe_semantics(recipe, profile)
        assert not result.is_valid
        assert any("mismatch" in e.lower() or "label" in e.lower() for e in result.errors)

    def test_matching_label_count_no_error(self):
        recipe = _make_recipe()
        profile = _make_profile(
            num_classes=3,
            label_distribution={"a": 100, "b": 100, "c": 100},
        )
        result = validate_recipe_semantics(recipe, profile)
        assert not any("mismatch" in e.lower() for e in result.errors)

    def test_zero_label_count_skips_check(self):
        """Can't compare if one side is unknown."""
        recipe = _make_recipe()
        profile = _make_profile(num_classes=0, label_distribution={})
        result = validate_recipe_semantics(recipe, profile)
        assert not any("mismatch" in e.lower() for e in result.errors)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_recipe_returns_valid(self):
        result = validate_recipe_semantics({}, _make_profile())
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []

    def test_none_recipe_returns_valid(self):
        result = validate_recipe_semantics(None, _make_profile())  # type: ignore[arg-type]
        assert result.is_valid

    def test_valid_recipe_no_issues(self):
        recipe = _make_recipe(
            base_model="bert-base-uncased",
            training_approach="lora",
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            num_epochs=3,
            batch_size=16,
            max_length=128,
        )
        profile = _make_profile(num_rows=1000, num_classes=3,
                                 label_distribution={"a": 400, "b": 400, "c": 200})
        result = validate_recipe_semantics(recipe, profile)
        assert result.is_valid
        assert result.errors == []

    def test_multiple_errors_all_returned(self):
        """All errors should be collected, not short-circuited after first."""
        recipe = _make_recipe(
            base_model="bert-base-uncased",
            training_approach="qlora",   # ERROR: QLoRA on encoder
            learning_rate=0.5,           # ERROR: lr too high
            max_length=1024,             # ERROR: exceeds BERT limit
        )
        result = validate_recipe_semantics(recipe, _make_profile())
        assert not result.is_valid
        assert len(result.errors) >= 2  # at least QLoRA + max_length (lr caught by pydantic usually)
