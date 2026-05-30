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
