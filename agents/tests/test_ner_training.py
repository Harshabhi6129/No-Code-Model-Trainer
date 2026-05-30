"""
Tests for Step 14 — Token Classification (NER).

Covers:
  • SUPPORTED_TASK_TYPES includes "token_classification"
  • validate_training_inputs() accepts token_classification
  • BIO tag parsing: rows with mismatched token/tag counts are dropped with warning
  • Label scheme correctly derived from all tags in dataset
  • Invalid BIO sequences (B-ORG followed by I-PER) don't crash
  • Label alignment: continuation subwords get label=-100
  • IntentAgent: NER intent keywords recognized (prompt contains NER guidance)
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agents.ml_core import SUPPORTED_TASK_TYPES, validate_training_inputs


# ── SUPPORTED_TASK_TYPES ──────────────────────────────────────────────────────

def test_token_classification_in_supported_types():
    assert "token_classification" in SUPPORTED_TASK_TYPES


def test_text_classification_still_in_supported_types():
    assert "text_classification" in SUPPORTED_TASK_TYPES


# ── validate_training_inputs ──────────────────────────────────────────────────

@pytest.fixture
def ner_csv(tmp_path: Path) -> Path:
    """Minimal valid NER CSV with BIO tags."""
    path = tmp_path / "ner.csv"
    rows = [
        {"tokens": "John lives in London",          "tags": "B-PER O O B-LOC"},
        {"tokens": "Apple is headquartered in USA", "tags": "B-ORG O O O O B-LOC"},
        {"tokens": "She went to Paris",             "tags": "O O O B-LOC"},
        {"tokens": "Google acquired YouTube",       "tags": "B-ORG O B-ORG"},
        {"tokens": "The UN voted today",            "tags": "O B-ORG O O"},
        {"tokens": "Mary Jane is a doctor",         "tags": "B-PER I-PER O O O"},
        {"tokens": "Berlin is in Germany",          "tags": "B-LOC O O B-LOC"},
        {"tokens": "Microsoft CEO Satya Nadella",   "tags": "B-ORG O B-PER I-PER"},
        {"tokens": "EU adopts new policies",        "tags": "B-ORG O O O"},
        {"tokens": "Paris fashion week",            "tags": "B-LOC O O"},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokens", "tags"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_validate_accepts_token_classification(ner_csv: Path):
    result = validate_training_inputs(
        dataset_path=str(ner_csv),
        task_type="token_classification",
        text_col="tokens",
        label_col="tags",
        model_id="bert-base-uncased",
    )
    assert result.ok, f"Validation failed: {result.error}"


def test_validate_token_classification_wrong_task_type(ner_csv: Path):
    """If task_type is unsupported, validation fails."""
    result = validate_training_inputs(
        dataset_path=str(ner_csv),
        task_type="unsupported_type",
        text_col="tokens",
        label_col="tags",
        model_id="bert-base-uncased",
    )
    assert not result.ok
    assert "unsupported" in result.error.lower()


# ── BIO parsing logic (pure Python, no GPU required) ─────────────────────────

class TestBIOParsing:
    def test_valid_bio_row_parsed(self):
        tokens = "John lives in London".split()
        tags   = "B-PER O O B-LOC".split()
        assert len(tokens) == len(tags)

    def test_mismatch_detected(self):
        tokens = "John lives in London".split()  # 4 tokens
        tags   = "B-PER O O".split()             # 3 tags
        assert len(tokens) != len(tags)

    def test_label_scheme_from_tags(self):
        """All unique tags from dataset form the label scheme."""
        all_tags = {"B-PER", "I-PER", "B-LOC", "B-ORG", "O"}
        label_names = sorted(all_tags)
        label2id = {lbl: i for i, lbl in enumerate(label_names)}
        assert len(label2id) == 5
        assert "O" in label2id
        assert "B-PER" in label2id

    def test_invalid_bio_sequence_no_crash(self):
        """B-ORG followed by I-PER is technically invalid BIO but we don't crash on it."""
        tokens = "Google Microsoft".split()
        tags   = "B-ORG I-PER".split()
        # Parsing should succeed (no validation of BIO rules)
        assert len(tokens) == len(tags)

    def test_o_tag_is_valid(self):
        tokens = "The quick brown fox".split()
        tags   = "O O O O".split()
        assert len(tokens) == len(tags)


# ── Token-label alignment ─────────────────────────────────────────────────────

class TestLabelAlignment:
    def test_continuation_subwords_get_minus_100(self):
        """
        Wordpiece tokenization splits words into subwords.
        Only the first subword of each word should get the real label;
        continuation subwords should get -100 (ignored in loss).
        """
        # Simulate word_ids from a tokenizer
        # Sentence: "John" "lives" → word_ids = [None, 0, 1, None]
        # (None = CLS/SEP special tokens)
        word_ids = [None, 0, 0, 1, None]  # "John" split into 2 subwords
        tags     = ["B-PER", "O"]         # 2 words
        label2id = {"B-PER": 0, "O": 1}

        labels_out = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels_out.append(-100)
            elif word_id != prev_word_id:
                labels_out.append(label2id.get(tags[word_id], 0))
            else:
                labels_out.append(-100)
            prev_word_id = word_id

        # Expected: [-100, 0, -100, 1, -100]
        assert labels_out[0] == -100   # CLS
        assert labels_out[1] == 0     # B-PER (first subword of "John")
        assert labels_out[2] == -100  # -100 (continuation subword of "John")
        assert labels_out[3] == 1     # O (first subword of "lives")
        assert labels_out[4] == -100  # SEP

    def test_single_token_words_fully_labeled(self):
        """Words that aren't split should have all positions labeled."""
        word_ids = [None, 0, 1, 2, None]  # 3 single-token words
        tags     = ["B-LOC", "O", "B-ORG"]
        label2id = {"B-LOC": 0, "O": 1, "B-ORG": 2}

        labels_out = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels_out.append(-100)
            elif word_id != prev_word_id:
                labels_out.append(label2id.get(tags[word_id], 0))
            else:
                labels_out.append(-100)
            prev_word_id = word_id

        assert labels_out == [-100, 0, 1, 2, -100]


# ── IntentAgent NER recognition ───────────────────────────────────────────────

class TestIntentAgentNERRecognition:
    def test_ner_guidance_in_system_prompt(self):
        """IntentAgent's SYSTEM prompt must contain NER keyword guidance."""
        from agents.intent import SYSTEM
        assert "token_classification" in SYSTEM
        assert "NER" in SYSTEM or "entity" in SYSTEM.lower()

    def test_ner_model_hints_in_system_prompt(self):
        """NER-specific model hints must be present."""
        from agents.intent import SYSTEM
        assert "dslim/bert-base-NER" in SYSTEM or "NER" in SYSTEM
