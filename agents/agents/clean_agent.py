"""
Clean Agent — runs between Data and Model.

Phase B2 additions:
  • Label noise detection via cleanlab Confident Learning
    Uses TF-IDF + LogisticRegression cross-val to get pred_probs,
    then cleanlab.filter.find_label_issues() to identify suspect rows.
    Rows are FLAGGED (not deleted) — surfaces the estimate to the user
    and to EvalAgent for calibrated grading.
  • All cleanlab/sklearn imports are lazy — agent stays importable in
    environments without those packages.

Original deterministic steps (unchanged):
  • Drop empty-text rows
  • Drop exact-text duplicates
  • Normalise label whitespace
  • Write cleaned file if anything changed
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path

from .base import BaseAgent, AgentContext, AgentResult
from .schemas import CleanResult

logger = logging.getLogger(__name__)

# Minimum rows needed for a meaningful cross-val noise estimate
_MIN_ROWS_FOR_NOISE  = 30
# Cap rows fed to cleanlab — prevents blocking on huge datasets (subsample)
_MAX_ROWS_FOR_NOISE  = 5_000
# Report noise in issues if estimated rate exceeds this threshold
_NOISE_WARN_THRESH   = 0.05   # 5%
_NOISE_HIGH_THRESH   = 0.20   # 20%


def has_cleanlab() -> bool:
    """True when cleanlab + sklearn are importable (lazy probe)."""
    try:
        import cleanlab  # noqa: F401
        import sklearn   # noqa: F401
        return True
    except ImportError:
        return False


class CleanAgent(BaseAgent):
    name = "Clean"
    # Fully deterministic — zero LLM calls

    async def run(self, context: AgentContext) -> AgentResult:
        if not context.dataset_path:
            return AgentResult(
                agent_name=self.name, success=True,
                output={"status": "skipped", "reason": "no dataset"},
                message="No dataset to clean — skipping.",
                next_agent="Model",
            )

        path = Path(context.dataset_path)
        if not path.exists():
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message=f"Dataset file not found: {path}",
            )

        profile   = context.data_profile
        text_col  = str(profile.get("input_col")  or context.task_spec.get("input_column")  or "text")
        label_col = str(profile.get("label_col")  or context.task_spec.get("label_column")  or "label")

        rows = _load(path)
        if not rows:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message="Dataset appears empty after loading.",
            )

        original_count       = len(rows)
        rows, removed_nulls  = _drop_empty_text(rows, text_col)
        rows, removed_dups   = _drop_duplicates(rows, text_col)
        rows                 = _normalise_labels(rows, label_col)
        cleaned_count        = len(rows)
        removed_total        = original_count - cleaned_count

        # ── B2: Label noise detection ─────────────────────────────────────────
        noise_rate, noise_count, noise_issues = _detect_label_noise(
            rows=rows,
            text_col=text_col,
            label_col=label_col,
        )

        # Propagate noise metrics into data_profile so downstream agents
        # (ModelAgent compact profile, EvalAgent grading) can see them
        context.data_profile["label_noise_estimate"] = round(noise_rate, 4)
        context.data_profile["label_noise_count"]    = noise_count
        existing_issues: list[str] = list(context.data_profile.get("issues", []))
        context.data_profile["issues"] = existing_issues + noise_issues

        # ── Write cleaned file if anything changed ────────────────────────────
        if removed_total > 0:
            cleaned_path = path.with_stem(path.stem + "_cleaned")
            _save(rows, cleaned_path, path.suffix)
            context.dataset_path = str(cleaned_path)
            logger.info(
                "CleanAgent: wrote cleaned dataset to %s (%d → %d rows)",
                cleaned_path, original_count, cleaned_count,
            )
        else:
            cleaned_path = path

        result = CleanResult(
            original_rows=original_count,
            cleaned_rows=cleaned_count,
            removed_nulls=removed_nulls,
            removed_dups=removed_dups,
            cleaned_path=str(cleaned_path),
        )
        output = result.model_dump()
        output["label_noise_estimate"] = noise_rate
        output["label_noise_count"]    = noise_count

        # ── Build message ─────────────────────────────────────────────────────
        parts: list[str] = []
        if removed_total > 0:
            parts.append(f"Dataset cleaned: {original_count:,} → **{cleaned_count:,} rows**.")
            if removed_nulls:
                parts.append(f"Removed {removed_nulls} empty-text row(s).")
            if removed_dups:
                parts.append(f"Removed {removed_dups} duplicate row(s).")
        else:
            parts.append("Dataset is clean — no rows removed.")

        if noise_count > 0:
            pct = noise_rate * 100
            parts.append(
                f"{'⚠' if pct >= _NOISE_WARN_THRESH * 100 else 'ℹ'} "
                f"Label noise: **{pct:.1f}%** of rows ({noise_count}) may be mislabelled "
                f"(detected via Confident Learning)."
            )
            if pct >= _NOISE_HIGH_THRESH * 100:
                parts.append(
                    "Consider reviewing your labels — high noise can severely limit model accuracy."
                )

        return AgentResult(
            agent_name=self.name, success=True, output=output,
            message=" ".join(parts),
            next_agent="Model",
        )


# ── Label noise detection (Confident Learning) ────────────────────────────────

def _detect_label_noise(
    rows: list[dict],
    text_col: str,
    label_col: str,
) -> tuple[float, int, list[str]]:
    """
    Estimate label noise using cleanlab's Confident Learning algorithm.

    Returns (noise_rate, noise_count, issue_strings).
    - noise_rate: fraction of suspected mislabelled rows (0.0–1.0)
    - noise_count: absolute count
    - issue_strings: human-readable issue messages to append to data_profile["issues"]

    Returns (0.0, 0, []) on any failure — never raises.

    Algorithm:
      1. Vectorise text with TF-IDF (max 5 000 features)
      2. Cross-val predicted probabilities via LogisticRegression
      3. cleanlab.filter.find_label_issues() using Confident Learning
         (estimates joint distribution of noisy vs true labels)
    """
    if not has_cleanlab():
        logger.debug("CleanAgent: cleanlab/sklearn not installed — skipping noise detection")
        return 0.0, 0, []

    # Extract valid (text, label) pairs
    pairs = [
        (str(r.get(text_col, "")).strip(), str(r.get(label_col, "")).strip())
        for r in rows
        if str(r.get(text_col, "")).strip() and str(r.get(label_col, "")).strip()
    ]
    n = len(pairs)

    if n < _MIN_ROWS_FOR_NOISE:
        logger.debug("CleanAgent: %d rows < %d — skipping noise detection", n, _MIN_ROWS_FOR_NOISE)
        return 0.0, 0, []

    # Subsample large datasets to cap wall-clock time
    import random
    if n > _MAX_ROWS_FOR_NOISE:
        pairs = random.sample(pairs, _MAX_ROWS_FOR_NOISE)
        n = _MAX_ROWS_FOR_NOISE

    texts_raw, labels_raw = zip(*pairs)

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import LabelEncoder
        from cleanlab.filter import find_label_issues

        le = LabelEncoder()
        y  = le.fit_transform(labels_raw)
        n_classes = len(le.classes_)

        if n_classes < 2:
            return 0.0, 0, []

        # Minimum CV folds = min(5, fewest examples in any class)
        class_counts = Counter(y.tolist())
        min_class    = min(class_counts.values())
        cv_folds     = min(5, min_class)

        if cv_folds < 2:
            logger.debug(
                "CleanAgent: smallest class has %d samples — need ≥2 for CV", min_class
            )
            return 0.0, 0, []

        # TF-IDF + bigrams for richer features without GPU
        vectorizer = TfidfVectorizer(
            max_features=5_000,
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = vectorizer.fit_transform(texts_raw)

        clf = LogisticRegression(
            max_iter=500,
            C=1.0,
            solver="saga",
            multi_class="auto",
            n_jobs=-1,
        )
        pred_probs: np.ndarray = cross_val_predict(
            clf, X, y,
            cv=cv_folds,
            method="predict_proba",
        )

        issue_idx = find_label_issues(
            labels=y,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
            filter_by="both",          # identifies both label errors and near-duplicate issues
        )

        noise_count = int(len(issue_idx))
        noise_rate  = noise_count / n

        issue_msgs: list[str] = []
        if noise_rate >= _NOISE_WARN_THRESH:
            issue_msgs.append(
                f"{noise_rate * 100:.1f}% label noise detected — "
                f"{noise_count} rows may be mislabelled (cleanlab Confident Learning)"
            )
        if noise_rate >= _NOISE_HIGH_THRESH:
            issue_msgs.append(
                "High label noise (>20%) — model accuracy may be severely limited; "
                "review and correct labels before training"
            )

        logger.info(
            "CleanAgent: noise detection — %.1f%% suspected mislabelled (%d/%d rows)",
            noise_rate * 100, noise_count, n,
        )
        return noise_rate, noise_count, issue_msgs

    except Exception as exc:
        logger.warning("CleanAgent: noise detection failed (continuing): %s", exc)
        return 0.0, 0, []


# ── Basic cleaning helpers ────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(ln) for ln in content.splitlines() if ln.strip()]
    return []


def _save(rows: list[dict], dest: Path, suffix: str) -> None:
    if suffix == ".csv":
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)


def _drop_empty_text(rows: list[dict], text_col: str) -> tuple[list[dict], int]:
    kept = [r for r in rows if str(r.get(text_col, "")).strip()]
    return kept, len(rows) - len(kept)


def _drop_duplicates(rows: list[dict], text_col: str) -> tuple[list[dict], int]:
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        key = str(row.get(text_col, "")).strip()
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, len(rows) - len(kept)


def _normalise_labels(rows: list[dict], label_col: str) -> list[dict]:
    for row in rows:
        if label_col in row and row[label_col] is not None:
            row[label_col] = str(row[label_col]).strip()
    return rows
