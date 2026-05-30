"""
Tests for Step 9 — Standalone Training Script Export.

Covers:
  • Generated script is valid Python (ast.parse succeeds)
  • CONFIG dict contains correct hyperparameters from recipe
  • base_model name is in the script
  • QLoRA: BitsAndBytesConfig import present
  • LoRA: peft imports present
  • Full fine-tune: no LoRA/PEFT imports
  • Label names embedded correctly
  • CodeGenerationError never raised on valid inputs
  • Empty training_result → handled gracefully (no KeyError)
"""
from __future__ import annotations

import ast

import pytest

# Locate the code generator (it's in agents/services/, not agents/agents/)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services"))

from code_generator import generate_training_script, CodeGenerationError


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _task_spec(**kwargs):
    base = {
        "task_type": "text_classification",
        "num_labels": 3,
        "label_names": ["positive", "neutral", "negative"],
        "input_column": "text",
        "label_column": "label",
    }
    base.update(kwargs)
    return base


def _data_profile(**kwargs):
    base = {
        "num_rows": 500,
        "num_classes": 3,
        "label_distribution": {"positive": 200, "neutral": 150, "negative": 150},
    }
    base.update(kwargs)
    return base


def _recipe(approach: str = "full_finetune", **kwargs):
    base = {
        "base_model":         "bert-base-uncased",
        "training_approach":  approach,
        "learning_rate":      2e-5,
        "num_epochs":         3,
        "batch_size":         16,
        "max_length":         128,
        "warmup_ratio":       0.1,
        "weight_decay":       0.01,
        "lora_r":             16,
        "lora_alpha":         32,
    }
    base.update(kwargs)
    return base


def _training_result(**kwargs):
    base = {
        "accuracy":    0.88,
        "f1":          0.87,
        "label_names": ["positive", "neutral", "negative"],
        "num_labels":  3,
    }
    base.update(kwargs)
    return base


# ── Syntax correctness ────────────────────────────────────────────────────────

class TestSyntaxCorrectness:
    def test_full_finetune_is_valid_python(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("full_finetune"), _training_result()
        )
        ast.parse(script)  # no exception = valid

    def test_lora_is_valid_python(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("lora"), _training_result()
        )
        ast.parse(script)

    def test_qlora_is_valid_python(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("qlora"), _training_result()
        )
        ast.parse(script)

    def test_many_labels_is_valid_python(self):
        labels = [f"class_{i}" for i in range(20)]
        script = generate_training_script(
            _task_spec(label_names=labels, num_labels=20),
            _data_profile(),
            _recipe(),
            _training_result(label_names=labels, num_labels=20),
        )
        ast.parse(script)

    def test_empty_label_names_is_valid_python(self):
        """Falls back to empty list — still valid Python."""
        script = generate_training_script(
            _task_spec(label_names=None),
            _data_profile(label_distribution={}),
            _recipe(),
            _training_result(label_names=None),
        )
        ast.parse(script)


# ── CONFIG dict content ───────────────────────────────────────────────────────

class TestConfigContent:
    def test_base_model_in_config(self):
        script = generate_training_script(
            _task_spec(), _data_profile(),
            _recipe(base_model="roberta-base"), _training_result()
        )
        assert '"roberta-base"' in script or "'roberta-base'" in script

    def test_learning_rate_in_config(self):
        script = generate_training_script(
            _task_spec(), _data_profile(),
            _recipe(learning_rate=3e-5), _training_result()
        )
        assert "3e-05" in script or "3e-5" in script or "0.00003" in script

    def test_num_epochs_in_config(self):
        script = generate_training_script(
            _task_spec(), _data_profile(),
            _recipe(num_epochs=7), _training_result()
        )
        assert "7" in script

    def test_batch_size_in_config(self):
        script = generate_training_script(
            _task_spec(), _data_profile(),
            _recipe(batch_size=32), _training_result()
        )
        assert "32" in script

    def test_label_names_in_script(self):
        script = generate_training_script(
            _task_spec(label_names=["spam", "ham"]), _data_profile(),
            _recipe(), _training_result(label_names=["spam", "ham"])
        )
        assert "spam" in script
        assert "ham" in script

    def test_input_column_in_script(self):
        script = generate_training_script(
            _task_spec(input_column="review_text"), _data_profile(),
            _recipe(), _training_result()
        )
        assert "review_text" in script

    def test_label_column_in_script(self):
        script = generate_training_script(
            _task_spec(label_column="sentiment"), _data_profile(),
            _recipe(), _training_result()
        )
        assert "sentiment" in script


# ── Approach-specific content ─────────────────────────────────────────────────

class TestApproachContent:
    def test_qlora_has_bnb_import(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("qlora"), _training_result()
        )
        assert "BitsAndBytesConfig" in script

    def test_lora_has_peft_import(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("lora"), _training_result()
        )
        assert "peft" in script
        assert "LoraConfig" in script

    def test_full_finetune_no_qlora(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("full_finetune"), _training_result()
        )
        assert "BitsAndBytesConfig" not in script

    def test_full_finetune_no_lora_adapter(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("full_finetune"), _training_result()
        )
        assert "LoraConfig" not in script
        assert "get_peft_model" not in script

    def test_qlora_has_load_in_4bit(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe("qlora"), _training_result()
        )
        assert "load_in_4bit" in script


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_training_result_no_crash(self):
        """Missing metrics → header shows 'unavailable' but script still valid."""
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe(), {}
        )
        ast.parse(script)
        assert "metrics unavailable" in script or "unavailable" in script

    def test_special_chars_in_label_names(self):
        """Label names with special characters are repr()-escaped safely."""
        labels = ["class/A", "class B", "class'C"]
        script = generate_training_script(
            _task_spec(label_names=labels),
            _data_profile(),
            _recipe(),
            _training_result(label_names=labels),
        )
        ast.parse(script)  # must not have syntax errors

    def test_returns_string(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe(), _training_result()
        )
        assert isinstance(script, str)
        assert len(script) > 500  # non-trivial script

    def test_has_main_guard(self):
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe(), _training_result()
        )
        assert '__name__ == "__main__"' in script or "__main__" in script

    def test_has_argparse(self):
        """Script must accept --data_path argument."""
        script = generate_training_script(
            _task_spec(), _data_profile(), _recipe(), _training_result()
        )
        assert "argparse" in script
        assert "--data_path" in script
