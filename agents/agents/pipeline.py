from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator

from .base import AgentContext, AgentResult, StageMetrics
from .intent import IntentAgent
from .data import DataAgent
from .clean_agent import CleanAgent
from .model import ModelAgent
from .train_agent import TrainAgent
from .eval_agent import EvalAgent
from .deploy_agent import DeployAgent
from .cache import recipe_cache
from .memory import episodic_memory

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self) -> None:
        self.intent = IntentAgent()
        self.data   = DataAgent()
        self.clean  = CleanAgent()
        self.model  = ModelAgent()
        self.train  = TrainAgent()
        self.eval   = EvalAgent()
        self.deploy = DeployAgent()

    async def run_streaming(
        self,
        user_intent: str,
        dataset_path: str | None = None,
        hyperparameter_overrides: dict | None = None,
        hf_token: str | None = None,
        # A3: checkpoint/resume ─────────────────────────────────────────────
        run_id: str | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[AgentResult, AgentContext]]:
        """
        Stream (AgentResult, AgentContext) tuples as the pipeline progresses.

        initial_context: if provided (loaded from a checkpoint), fields are
          restored into the AgentContext so already-completed stages are skipped.
          The caller is responsible for loading the checkpoint; the pipeline
          just honours it.

        The final event emitted is always a "pipeline_summary" result that
        aggregates per-stage token/cost/latency metrics from every LLM call.
        """
        context = AgentContext(
            run_id=run_id or str(uuid.uuid4()),
            user_intent=user_intent,
            dataset_path=dataset_path,
            hyperparameter_overrides=hyperparameter_overrides or {},
            hf_token=hf_token,
        )

        # Restore checkpoint fields if we're resuming
        if initial_context:
            _restore_context(context, initial_context)
            logger.info(
                "[%s] Resuming pipeline — skipping completed stages: %s",
                context.run_id, context.completed_stages,
            )

        agents = [
            self.intent, self.data, self.clean, self.model,
            self.train, self.eval, self.deploy,
        ]
        last_result: AgentResult | None = None
        # Accumulate StageMetrics from every LLM-calling agent
        all_stage_metrics: list[StageMetrics] = []
        pipeline_failed = False

        for agent in agents:
            # Skip stages already recorded in completed_stages
            if agent.name in context.completed_stages:
                logger.info("[%s] Skipping %s (already completed)", context.run_id, agent.name)
                continue

            try:
                async for result in agent.run_stream(context):
                    last_result = result
                    yield result, context
                    if not result.success:
                        pipeline_failed = True
                        break  # hard failure

            except Exception as exc:
                logger.error(
                    "Unhandled exception in %s agent: %s",
                    agent.name, exc, exc_info=True,
                )
                pipeline_failed = True
                yield AgentResult(
                    agent_name=agent.name,
                    success=False,
                    output={},
                    message=(
                        f"An unexpected error occurred in the {agent.name} agent. "
                        "Please try again."
                    ),
                    next_agent=None,
                ), context

            # Collect this agent's last LLM call metrics (None for deterministic agents)
            if hasattr(agent, "last_stage_metrics") and agent.last_stage_metrics is not None:
                all_stage_metrics.append(agent.last_stage_metrics)

            if pipeline_failed:
                break

            # ── After EvalAgent: write successful recipe to cache + memory ──────
            if agent.name == "Eval" and last_result and last_result.success:
                grade     = context.eval_result.get("evaluation_grade", "")
                eval_f1   = float(context.eval_result.get("f1") or 0)
                recipe    = context.model_recipe
                task_type = context.task_spec.get("task_type", "text_classification")
                if recipe and grade:
                    recipe_cache.set(context.data_profile, task_type, recipe, grade)
                    try:
                        episodic_memory.memorize(
                            context.data_profile, task_type, recipe, grade, eval_f1
                        )
                    except Exception as exc:
                        logger.debug("EpisodicMemory memorize error: %s", exc)

            # Mark stage complete so the checkpoint knows where we are
            if last_result and last_result.success:
                if agent.name not in context.completed_stages:
                    context.completed_stages.append(agent.name)

            # Stop if the last result didn't request a next agent
            if last_result is None or last_result.next_agent is None:
                break

        # ── Emit pipeline summary ────────────────────────────────────────────
        # Always emitted (success or failure) so the UI can display cost/latency.
        summary = _build_pipeline_summary(all_stage_metrics)
        logger.info(
            "[%s] Pipeline done — total cost=$%.6f tokens=%d cache_hit=%.1f%%",
            context.run_id,
            summary["total_cost_usd"],
            summary["total_input_tokens"] + summary["total_output_tokens"],
            summary["overall_cache_hit_ratio"] * 100,
        )
        yield AgentResult(
            agent_name="Pipeline",
            success=not pipeline_failed,
            output={"type": "pipeline_summary", **summary},
            message=(
                f"Pipeline {'completed' if not pipeline_failed else 'stopped'}. "
                f"Total cost: ${summary['total_cost_usd']:.4f} · "
                f"Cache hit: {summary['overall_cache_hit_ratio'] * 100:.0f}%"
            ),
            next_agent=None,
            metadata={"stage_metrics": [_metrics_to_dict(m) for m in all_stage_metrics]},
        ), context

    def context_snapshot(self, context: AgentContext) -> dict[str, Any]:
        """
        Return a JSON-serialisable snapshot of the context for checkpointing.
        Excludes hf_token (sensitive) and any non-serialisable fields.
        """
        d = asdict(context)
        d.pop("hf_token", None)  # never persist secrets
        return d


# ── Helpers ──────────────────────────────────────────────────────────────────

def summarize_pipeline_context(context: "AgentContext") -> dict[str, Any]:
    """
    Return a compact, JSON-safe summary of pipeline context for downstream LLM prompts.

    Goals:
      - Total JSON < 5 KB regardless of dataset size
      - Secrets (hf_token) never included
      - Large arrays (epoch_metrics, full label_distribution) are truncated
      - EvalAgent and DeployAgent receive exactly what they need — nothing more

    The raw context fields are NOT modified; this returns a new dict.
    """
    profile = context.data_profile
    tr      = context.training_result
    spec    = context.task_spec
    ev      = context.eval_result

    # Truncate label_distribution to top-10 by count (keeps prompt compact for >10 class tasks)
    label_dist = profile.get("label_distribution", {})
    if len(label_dist) > _SUMMARY_MAX_CLASSES:
        top_classes = dict(
            sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:_SUMMARY_MAX_CLASSES]
        )
        remaining = len(label_dist) - _SUMMARY_MAX_CLASSES
        top_classes[f"… and {remaining} more classes"] = sum(
            v for k, v in label_dist.items()
            if k not in top_classes
        )
        label_dist = top_classes

    # Loss history: only last 5 entries (sufficient for trend analysis)
    epoch_metrics = tr.get("epoch_metrics", [])
    loss_history_tail = epoch_metrics[-5:] if epoch_metrics else []

    return {
        "task_spec": {
            "task_type":    spec.get("task_type"),
            "num_labels":   spec.get("num_labels"),
            "label_names":  spec.get("label_names"),
            "input_column": spec.get("input_column"),
            "label_column": spec.get("label_column"),
        },
        "data_profile": {
            "num_rows":              profile.get("num_rows"),
            "num_classes":           profile.get("num_classes"),
            "label_distribution":    label_dist,
            "label_noise_estimate":  profile.get("label_noise_estimate", 0.0),
            "label_noise_count":     profile.get("label_noise_count", 0),
            "text_quality_score":    profile.get("text_quality_score", 1.0),
            "issues":                profile.get("issues", []),
        },
        "training_result": {
            "base_model":           tr.get("base_model"),
            "training_approach":    tr.get("training_approach"),
            "device":               tr.get("device"),
            "num_epochs_completed": tr.get("num_epochs_completed"),
            "final_train_loss":     tr.get("final_train_loss"),
            "training_time_seconds": tr.get("training_time_seconds"),
            "accuracy":             tr.get("accuracy"),
            "f1":                   tr.get("f1"),
            "precision":            tr.get("precision"),
            "recall":               tr.get("recall"),
            "ece":                  tr.get("ece"),
            "per_class_f1":         _trim_per_class_f1(tr.get("per_class_f1", {})),
            "num_labels":           tr.get("num_labels"),
            "label_names":          (tr.get("label_names") or [])[:_SUMMARY_MAX_CLASSES],
            "train_samples":        tr.get("train_samples"),
            "eval_samples":         tr.get("eval_samples"),
            "warnings":             tr.get("warnings", []),
            "loss_history_tail":    loss_history_tail,  # last 5 epochs only
        },
        "eval_result": {
            "evaluation_grade": ev.get("evaluation_grade"),
            "summary":          ev.get("summary", ""),
            "concerns":         ev.get("concerns", []),
            "next_steps":       ev.get("next_steps", []),
        } if ev else {},
    }


# Truncate label_distribution to this many entries in LLM prompts (saves tokens for many-class tasks)
_SUMMARY_MAX_CLASSES = 10


def _build_pipeline_summary(metrics: list[StageMetrics]) -> dict[str, Any]:
    """Aggregate per-stage StageMetrics into a pipeline-level cost/token summary."""
    if not metrics:
        return {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_write_tokens": 0,
            "overall_cache_hit_ratio": 0.0,
            "total_latency_ms": 0.0,
            "llm_stages_called": 0,
            "per_stage": [],
        }

    total_input   = sum(m.input_tokens for m in metrics)
    total_output  = sum(m.output_tokens for m in metrics)
    total_cr      = sum(m.cache_read_tokens for m in metrics)
    total_cw      = sum(m.cache_write_tokens for m in metrics)
    total_cost    = sum(m.estimated_cost_usd for m in metrics)
    total_latency = sum(m.latency_ms for m in metrics)

    return {
        "total_cost_usd": round(total_cost, 8),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cr,
        "total_cache_write_tokens": total_cw,
        "overall_cache_hit_ratio": round(total_cr / max(total_input, 1), 4),
        "total_latency_ms": round(total_latency, 1),
        "llm_stages_called": len(metrics),
        "per_stage": [_metrics_to_dict(m) for m in metrics],
    }


def _trim_per_class_f1(per_class: dict[str, float]) -> dict[str, float]:
    """Return at most 10 per-class F1 entries: worst 5 + best 5.
    These are most diagnostic for the eval agent — weakest classes drive concerns,
    strongest drive strengths.  For ≤ 10 classes, returns the full dict."""
    if len(per_class) <= _SUMMARY_MAX_CLASSES:
        return per_class
    sorted_items = sorted(per_class.items(), key=lambda x: x[1])
    worst = sorted_items[:5]
    best  = sorted_items[-5:]
    seen  = {k for k, _ in worst}
    combined = dict(worst + [item for item in best if item[0] not in seen])
    return combined


def _metrics_to_dict(m: StageMetrics) -> dict[str, Any]:
    return {
        "agent": m.agent_name,
        "model": m.model,
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "cache_read_tokens": m.cache_read_tokens,
        "cache_write_tokens": m.cache_write_tokens,
        "latency_ms": m.latency_ms,
        "estimated_cost_usd": m.estimated_cost_usd,
        "cache_hit_ratio": m.cache_hit_ratio,
        "timestamp": m.timestamp,
    }


def _restore_context(context: AgentContext, snapshot: dict[str, Any]) -> None:
    """
    Copy fields from a checkpoint snapshot back into a fresh AgentContext.
    Only updates fields that are present in the snapshot and non-empty.
    Preserves the live run_id / dataset_path / hf_token set by the caller.
    """
    _safe_fields = {
        "task_spec", "data_profile", "model_recipe",
        "training_result", "eval_result", "deploy_result",
        "completed_stages",
    }
    for field in _safe_fields:
        value = snapshot.get(field)
        if value:
            setattr(context, field, value)
