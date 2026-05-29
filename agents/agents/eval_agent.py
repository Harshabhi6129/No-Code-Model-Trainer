"""
Eval Agent — calibrated post-training evaluation report.
Uses Claude to interpret raw metrics in context of task difficulty,
dataset size, and class distribution. Flags overfitting, class-specific
issues, and gives concrete next steps.
"""
from __future__ import annotations

import json
import logging

from .base import BaseAgent, AgentContext, AgentResult, SONNET
from .schemas import EvalReport

logger = logging.getLogger(__name__)

SYSTEM = """You are the Eval Agent for ModelForge, an AI model training platform.
You have received training results and must produce a calibrated evaluation report.

You will be given:
- Task type, dataset size, number of classes, class distribution
- Training metrics: accuracy, weighted F1, precision, recall, per-class F1
- Training metadata: model, approach, device, epochs, warnings

Output ONLY valid JSON (no prose, no markdown):
{
  "evaluation_grade": "A" | "B" | "C" | "D" | "F",
  "summary": "<2-3 sentence plain-English summary — calibrated for dataset size and task difficulty>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "concerns": ["<concern 1>"],
  "next_steps": ["<concrete, specific action>", "<concrete, specific action>"]
}

=== GRADING RUBRIC ===
(Always calibrate for dataset size — small datasets naturally score lower.)

For datasets >= 1000 samples:
  A: F1 >= 0.90  B: F1 0.80-0.89  C: F1 0.65-0.79  D: F1 0.50-0.64  F: F1 < 0.50

For datasets 200-999 samples:
  A: F1 >= 0.85  B: F1 0.72-0.84  C: F1 0.58-0.71  D: F1 0.45-0.57  F: F1 < 0.45

For datasets < 200 samples:
  A: F1 >= 0.80  B: F1 0.65-0.79  C: F1 0.50-0.64  D: F1 0.38-0.49  F: F1 < 0.38

=== CONCERN PATTERNS ===
- accuracy >> F1 (gap > 0.10): class imbalance — mention it
- per-class F1 variance > 0.20: some classes are much harder — name them
- dataset < 100 samples: always mention high metric variance
- num_classes > 10: note difficulty of fine-grained classification
- training_approach = "full_finetune" on < 500 samples: overfitting risk
- final_train_loss very low but F1 low: possible overfitting
- warnings from training: incorporate them into concerns/next_steps

=== NEXT STEPS ===
Be specific — mention actual actions (e.g., "add 50 more examples for class 'X'",
not "add more data"). Reference the actual class names and metrics when relevant."""


def _build_prompt(context: AgentContext) -> str:
    tr = context.training_result
    spec = context.task_spec
    profile = context.data_profile

    # Handle skipped training (no GPU libs)
    if tr.get("status") == "skipped":
        return json.dumps({
            "status": "training_skipped",
            "reason": tr.get("reason", ""),
            "task_type": spec.get("task_type", "unknown"),
            "recipe": tr.get("recipe", {}),
        })

    label_names = tr.get("label_names", spec.get("label_names") or [])
    per_class = tr.get("per_class_f1", {})
    label_distribution = profile.get("label_distribution", {})
    issues = profile.get("issues", [])
    all_warnings = tr.get("warnings", [])

    return json.dumps({
        "task_type":          spec.get("task_type", "text_classification"),
        "dataset": {
            "total_samples":   (tr.get("train_samples", 0) + tr.get("eval_samples", 0)),
            "train_samples":   tr.get("train_samples", 0),
            "eval_samples":    tr.get("eval_samples", 0),
            "num_classes":     tr.get("num_labels", len(label_names)),
            "label_names":     label_names,
            "label_distribution": label_distribution,
            "dataset_issues":  issues,
        },
        "model": {
            "base_model":         tr.get("base_model", "unknown"),
            "training_approach":  tr.get("training_approach", "unknown"),
            "device":             tr.get("device", "cpu"),
            "num_epochs":         tr.get("num_epochs_completed", 0),
            "final_train_loss":   tr.get("final_train_loss"),
            "training_time_secs": tr.get("training_time_seconds", 0),
        },
        "metrics": {
            "accuracy":     tr.get("accuracy"),
            "f1_weighted":  tr.get("f1"),
            "precision":    tr.get("precision"),
            "recall":       tr.get("recall"),
            "per_class_f1": per_class,
        },
        "warnings_from_training": all_warnings,
    }, indent=2)


class EvalAgent(BaseAgent):
    name  = "Eval"
    model = SONNET  # Needs contextual reasoning to interpret metrics + calibrate grade

    async def run(self, context: AgentContext) -> AgentResult:
        tr = context.training_result

        # ── No training result at all ─────────────────────────────────────────
        if not tr:
            return AgentResult(
                agent_name=self.name,
                success=False,
                output={},
                message="No training result available to evaluate.",
                next_agent=None,
            )

        # ── Training was skipped (no GPU libs) ────────────────────────────────
        if tr.get("status") == "skipped":
            context.eval_result = {
                "evaluation_grade": "—",
                "summary": (
                    "Training was skipped because GPU libraries are not installed in this environment. "
                    "Your recipe is saved — run ModelForge locally or with GPU support to train and evaluate."
                ),
                "strengths": ["Pipeline completed validation and model selection successfully."],
                "concerns":  ["No trained model — cannot evaluate performance."],
                "next_steps": [
                    "Install PyTorch and run `uvicorn main:app` locally with the full requirements.txt.",
                    "GPU cloud training (Modal) is on the ModelForge roadmap.",
                ],
            }
            return AgentResult(
                agent_name=self.name,
                success=True,
                output=context.eval_result,
                message=(
                    "No training was executed — evaluation skipped.\n"
                    "Your model recipe is ready. Run locally with GPU support to train."
                ),
                next_agent="Deploy",
            )

        # ── Call Claude for calibrated evaluation ─────────────────────────────
        # System prompt cached — grading rubric + concern patterns are ~800 tokens,
        # repeated on every run. Cache hits cut cost significantly at scale.
        prompt = _build_prompt(context)
        try:
            raw = await self._chat(
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                cache_system=True,
            )
            eval_model, _ = self._parse_llm_json(raw, EvalReport)
            report = eval_model.model_dump() if eval_model else _fallback_report(tr)
        except Exception as exc:
            logger.error("EvalAgent: Claude call failed: %s", exc, exc_info=True)
            report = _fallback_report(tr)

        # ── Merge raw metrics into output so Supabase gets everything ─────────
        eval_output: dict = {
            # Raw metrics (for run detail page charts)
            "accuracy":     tr.get("accuracy"),
            "f1":           tr.get("f1"),
            "precision":    tr.get("precision"),
            "recall":       tr.get("recall"),
            "per_class_f1": tr.get("per_class_f1", {}),
            "num_labels":   tr.get("num_labels"),
            "label_names":  tr.get("label_names", []),
            "train_samples": tr.get("train_samples"),
            "eval_samples":  tr.get("eval_samples"),
            # Claude evaluation
            "evaluation_grade": report.get("evaluation_grade", "—"),
            "summary":          report.get("summary", ""),
            "strengths":        report.get("strengths", []),
            "concerns":         report.get("concerns", []),
            "next_steps":       report.get("next_steps", []),
        }
        context.eval_result = eval_output

        # ── Build user-facing message ─────────────────────────────────────────
        grade = report.get("evaluation_grade", "—")
        summary = report.get("summary", "")
        concerns = report.get("concerns", [])
        next_steps = report.get("next_steps", [])

        acc_pct = f"{tr['accuracy'] * 100:.1f}%" if tr.get("accuracy") is not None else "—"
        f1_val  = f"{tr['f1']:.3f}" if tr.get("f1") is not None else "—"

        parts = [
            f"**Grade: {grade}** — Accuracy: {acc_pct} | F1: {f1_val}",
            summary,
        ]
        if concerns:
            parts.append("**Concerns:** " + " | ".join(concerns))
        if next_steps:
            parts.append("**Next steps:** " + " | ".join(next_steps))

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=eval_output,
            message="\n".join(parts),
            next_agent=None,
        )


def _fallback_report(tr: dict) -> dict:
    """Deterministic fallback when Claude is unavailable."""
    f1 = tr.get("f1", 0.0) or 0.0
    total = (tr.get("train_samples") or 0) + (tr.get("eval_samples") or 0)

    if total >= 1000:
        thresholds = [(0.90, "A"), (0.80, "B"), (0.65, "C"), (0.50, "D")]
    elif total >= 200:
        thresholds = [(0.85, "A"), (0.72, "B"), (0.58, "C"), (0.45, "D")]
    else:
        thresholds = [(0.80, "A"), (0.65, "B"), (0.50, "C"), (0.38, "D")]

    grade = "F"
    for threshold, g in thresholds:
        if f1 >= threshold:
            grade = g
            break

    acc_pct = f"{tr['accuracy'] * 100:.1f}%" if tr.get("accuracy") else "—"
    return {
        "evaluation_grade": grade,
        "summary": (
            f"The model achieved {acc_pct} accuracy and {f1:.3f} weighted F1 "
            f"on {tr.get('eval_samples', '?')} test samples."
        ),
        "strengths": ["Training completed successfully."],
        "concerns":  ["Detailed evaluation unavailable — Claude API may be temporarily unreachable."],
        "next_steps": [
            "Re-run the pipeline to get a full evaluation report.",
            "Inspect per-class F1 scores in the run detail page.",
        ],
    }
