from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import Counter
from .base import BaseAgent, AgentContext, AgentResult


class DataAgent(BaseAgent):
    name = "Data"

    async def run(self, context: AgentContext) -> AgentResult:
        if not context.dataset_path:
            return AgentResult(agent_name=self.name, success=False, output={},
                               message="No dataset uploaded. Please upload a CSV or JSON file to continue.")

        path = Path(context.dataset_path)
        if not path.exists():
            return AgentResult(agent_name=self.name, success=False, output={},
                               message=f"Dataset file not found at {path}.")

        profile = self._profile(path, context.task_spec)
        context.data_profile = profile
        issues = profile.get("issues", [])

        msg_parts = [
            f"Dataset: **{profile['num_rows']:,} rows**, {profile['num_cols']} columns.",
            f"Input column: `{profile['input_col']}` — avg {profile['avg_input_len']:.0f} chars.",
        ]
        if profile.get("label_distribution"):
            dist = profile["label_distribution"]
            msg_parts.append(f"Labels: {', '.join(f'`{k}` ({v})' for k, v in dist.items())}")
        if issues:
            msg_parts.append("**Issues found:** " + "; ".join(issues))

        return AgentResult(agent_name=self.name, success=True, output=profile,
                           message="\n".join(msg_parts), next_agent="Model")

    def _profile(self, path: Path, task_spec: dict) -> dict:
        input_col = task_spec.get("input_column", "text")
        label_col = task_spec.get("label_column", "label")
        rows: list[dict] = []

        if path.suffix == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif path.suffix in (".json", ".jsonl"):
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
                rows = json.loads(content) if content.startswith("[") else \
                       [json.loads(l) for l in content.splitlines() if l.strip()]

        if not rows:
            return {"num_rows": 0, "num_cols": 0, "issues": ["Empty dataset"]}

        cols = list(rows[0].keys())
        actual_input_col = input_col if input_col in cols else cols[0]
        actual_label_col = label_col if label_col in cols else (cols[1] if len(cols) > 1 else None)
        input_lens = [len(str(r.get(actual_input_col, ""))) for r in rows]
        missing_inputs = sum(1 for r in rows if not str(r.get(actual_input_col, "")).strip())

        issues = []
        if missing_inputs > 0:
            issues.append(f"{missing_inputs} rows have empty input")
        if len(rows) < 100:
            issues.append(f"Dataset is small ({len(rows)} rows) — model may underfit")

        profile: dict = {
            "num_rows": len(rows), "num_cols": len(cols), "columns": cols,
            "input_col": actual_input_col, "label_col": actual_label_col,
            "avg_input_len": sum(input_lens) / len(input_lens) if input_lens else 0,
            "max_input_len": max(input_lens) if input_lens else 0,
            "issues": issues,
        }
        if actual_label_col:
            labels = [str(r.get(actual_label_col, "")) for r in rows if r.get(actual_label_col)]
            profile["label_distribution"] = dict(Counter(labels).most_common(10))
            profile["num_classes"] = len(set(labels))
        return profile
