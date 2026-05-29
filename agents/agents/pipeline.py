from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator

from .base import AgentContext, AgentResult
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
                        return  # Hard failure — stop pipeline

            except Exception as exc:
                logger.error(
                    "Unhandled exception in %s agent: %s",
                    agent.name, exc, exc_info=True,
                )
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
                return

            # Mark stage complete so the checkpoint knows where we are
            if last_result and last_result.success:
                if agent.name not in context.completed_stages:
                    context.completed_stages.append(agent.name)

            # Stop if the last result didn't request a next agent
            if last_result is None or last_result.next_agent is None:
                return

    def context_snapshot(self, context: AgentContext) -> dict[str, Any]:
        """
        Return a JSON-serialisable snapshot of the context for checkpointing.
        Excludes hf_token (sensitive) and any non-serialisable fields.
        """
        d = asdict(context)
        d.pop("hf_token", None)  # never persist secrets
        return d


# ── Helpers ──────────────────────────────────────────────────────────────────

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
