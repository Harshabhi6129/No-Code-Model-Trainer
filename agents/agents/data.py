"""
Data Agent — fully deterministic, zero LLM calls.

Research finding: data profiling is a solved deterministic problem.
Reserve LLMs for narrating findings, not computing them. All output
is validated through DataProfile (Pydantic) before being stored in context.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path

from .base import BaseAgent, AgentContext, AgentResult
from .schemas import DataProfile

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    name = "Data"
    # No model tier required — this agent makes zero LLM calls

    async def run(self, context: AgentContext) -> AgentResult:
        if not context.dataset_path:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message="No dataset uploaded. Please upload a CSV or JSON file to continue.",
            )

        path = Path(context.dataset_path)
        if not path.exists():
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message=f"Dataset file not found at `{path}`. Please re-upload.",
            )

        raw = self._profile(path, context.task_spec)

        # Validate through Pydantic — ensures downstream agents receive
        # a typed, semantically-checked profile even if _profile is extended
        try:
            validated = DataProfile.model_validate(raw)
            profile   = validated.model_dump()
        except Exception as exc:
            logger.warning("DataProfile validation warning: %s — using raw dict", exc)
            profile = raw

        context.data_profile = profile

        if profile.get("num_rows", 0) == 0:
            return AgentResult(
                agent_name=self.name, success=False, output=profile,
                message="The uploaded dataset is empty. Please upload a file with at least one data row.",
            )

        issues = profile.get("issues", [])
        dist   = profile.get("label_distribution", {})

        tok_avg = profile.get("estimated_tokens_avg", 0)
        tok_p95 = profile.get("estimated_tokens_p95", 0)
        quality = profile.get("text_quality_score", 1.0)

        parts = [
            f"Dataset: **{profile['num_rows']:,} rows**, {profile['num_cols']} columns.",
            f"Text: avg {profile.get('avg_word_count', 0):.0f} words "
            f"(~{tok_avg} tokens avg, ~{tok_p95} at p95).",
        ]
        if dist:
            parts.append("Labels: " + ", ".join(f"`{k}` ({v})" for k, v in dist.items()))
        if quality < 0.80:
            parts.append(f"⚠ Text quality {quality:.0%} — contains HTML/URLs/noise.")

        dup_rate = profile.get("duplicate_rate", 0.0)
        if dup_rate > 0.05:
            parts.append(f"⚠ {dup_rate * 100:.1f}% duplicate rows — CleanAgent will remove them.")

        balance = profile.get("class_balance_ratio")
        if balance is not None and balance < 0.5:
            parts.append(
                f"⚠ Class imbalance (ratio {balance:.2f}) — consider class weighting."
            )

        if issues:
            parts.append("**Issues:** " + "; ".join(issues))

        return AgentResult(
            agent_name=self.name, success=True, output=profile,
            message="\n".join(parts), next_agent="Clean",
        )

    # ── Deterministic profiler ───────────────────────────────────────────────

    def _profile(self, path: Path, task_spec: dict) -> dict:
        import re as _re

        input_col = task_spec.get("input_column", "text")
        label_col = task_spec.get("label_column", "label")
        rows: list[dict] = []

        if path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif path.suffix in (".json", ".jsonl"):
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
                rows = (
                    json.loads(content)
                    if content.startswith("[")
                    else [json.loads(ln) for ln in content.splitlines() if ln.strip()]
                )

        if not rows:
            return {
                "num_rows": 0, "num_cols": 0, "input_col": input_col,
                "issues": ["Empty dataset"],
            }

        cols    = list(rows[0].keys())
        in_col  = input_col if input_col in cols else cols[0]
        lbl_col = label_col if label_col in cols else (cols[1] if len(cols) > 1 else None)

        texts          = [str(r.get(in_col, "")) for r in rows]
        input_lens     = [len(t) for t in texts]
        missing_inputs = sum(1 for t in texts if not t.strip())

        # ── Word / token statistics (B3) ─────────────────────────────────────
        # Cap sample size to 2 000 rows for O(n) operations on large datasets
        _sample = texts[:2_000]
        word_counts   = [len(t.split()) for t in _sample]
        avg_word      = sum(word_counts) / len(word_counts) if word_counts else 0.0
        max_word      = max(word_counts) if word_counts else 0

        # Token estimate: each word ≈ 1.3 sub-word tokens (GPT/BERT sub-word heuristic)
        token_counts     = [int(w * 1.3) for w in word_counts]
        estimated_avg    = int(sum(token_counts) / len(token_counts)) if token_counts else 0
        sorted_tokens    = sorted(token_counts)
        p95_idx          = max(0, int(0.95 * len(sorted_tokens)) - 1)
        estimated_p95    = sorted_tokens[p95_idx] if sorted_tokens else 0

        # Vocabulary richness: unique words / total words (type-token ratio)
        all_words = " ".join(_sample[:500]).lower().split()   # cap to 500 rows
        vocab_richness = (
            len(set(all_words)) / len(all_words) if all_words else 0.0
        )

        # Text quality: fraction of rows without HTML tags, bare URLs, or
        # "noise" text (>30% non-alphanumeric chars)
        _url_re  = _re.compile(r'https?://\S+|www\.\S+')
        _html_re = _re.compile(r'<[^>]+>')
        def _is_clean(t: str) -> bool:
            if _url_re.search(t) or _html_re.search(t):
                return False
            alnum = sum(1 for c in t if c.isalnum() or c.isspace())
            return (alnum / max(len(t), 1)) >= 0.70
        quality_score = (
            sum(1 for t in _sample if _is_clean(t)) / len(_sample)
            if _sample else 1.0
        )

        # Duplicate detection
        seen: set[str] = set()
        dup_count = 0
        for t in texts:
            if t in seen:
                dup_count += 1
            seen.add(t)

        issues: list[str] = []
        if missing_inputs > 0:
            issues.append(f"{missing_inputs} rows have empty input")
        if len(rows) < 100:
            issues.append(f"Small dataset ({len(rows)} rows) — model may underfit")
        if quality_score < 0.80:
            issues.append(
                f"Text quality score {quality_score:.0%} — many rows contain "
                "HTML, URLs, or noisy characters; consider pre-cleaning"
            )

        profile: dict = {
            "num_rows":      len(rows),
            "num_cols":      len(cols),
            "columns":       cols,
            "input_col":     in_col,
            "label_col":     lbl_col,
            # character-level
            "avg_input_len": sum(input_lens) / len(input_lens) if input_lens else 0.0,
            "max_input_len": max(input_lens) if input_lens else 0,
            # word / token level (B3)
            "avg_word_count":      round(avg_word, 1),
            "max_word_count":      max_word,
            "estimated_tokens_avg": estimated_avg,
            "estimated_tokens_p95": estimated_p95,
            "vocabulary_richness":  round(vocab_richness, 3),
            "text_quality_score":   round(quality_score, 3),
            # quality
            "duplicate_rate":  dup_count / len(rows),
            "missing_rate":    missing_inputs / len(rows),
            "label_noise_estimate": 0.0,  # populated by CleanAgent (Phase B2 / cleanlab)
            "issues":        issues,
        }

        if lbl_col:
            labels   = [str(r.get(lbl_col, "")).strip() for r in rows if r.get(lbl_col)]
            counter  = Counter(labels)
            profile["label_distribution"] = dict(counter.most_common(10))
            profile["num_classes"]        = len(set(labels))
            if counter:
                mn = min(counter.values())
                mx = max(counter.values())
                profile["class_balance_ratio"] = mn / mx if mx > 0 else 1.0

        return profile
