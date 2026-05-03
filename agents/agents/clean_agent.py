"""
Clean Agent — runs between Data and Model.
Removes duplicates, drops empty-text rows, normalises label whitespace,
and writes a cleaned dataset file.  Updates context.dataset_path so every
downstream agent (Model, Train, Eval) works on the cleaned data.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .base import BaseAgent, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class CleanAgent(BaseAgent):
    name = "Clean"

    async def run(self, context: AgentContext) -> AgentResult:
        if not context.dataset_path:
            # Nothing to clean — pass through silently
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
        text_col  = str(profile.get("input_col") or context.task_spec.get("input_column") or "text")
        label_col = str(profile.get("label_col") or context.task_spec.get("label_column") or "label")

        rows = _load(path)
        if not rows:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message="Dataset appears empty after loading.",
            )

        original_count = len(rows)
        rows, removed_nulls  = _drop_empty_text(rows, text_col)
        rows, removed_dups   = _drop_duplicates(rows, text_col)
        rows                 = _normalise_labels(rows, label_col)

        cleaned_count = len(rows)
        removed_total = original_count - cleaned_count

        # Write cleaned file only if something changed
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

        output: dict[str, Any] = {
            "original_rows": original_count,
            "cleaned_rows":  cleaned_count,
            "removed_nulls": removed_nulls,
            "removed_dups":  removed_dups,
            "cleaned_path":  str(cleaned_path),
        }

        parts = [f"Dataset cleaned: {original_count:,} → **{cleaned_count:,} rows**."]
        if removed_nulls:
            parts.append(f"Removed {removed_nulls} row(s) with empty text.")
        if removed_dups:
            parts.append(f"Removed {removed_dups} duplicate row(s).")
        if removed_total == 0:
            parts = ["Dataset is clean — no rows removed."]

        return AgentResult(
            agent_name=self.name, success=True, output=output,
            message=" ".join(parts),
            next_agent="Model",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
