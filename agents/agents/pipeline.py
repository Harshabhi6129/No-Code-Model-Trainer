from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

from .base import AgentContext, AgentResult
from .intent import IntentAgent
from .data import DataAgent
from .model import ModelAgent
from .train_agent import TrainAgent
from .eval_agent import EvalAgent
from .deploy_agent import DeployAgent

logger = logging.getLogger(__name__)

# Full 5-agent pipeline: Intent → Data → Model → Train → Eval
_AGENT_ORDER = ["intent", "data", "model", "train", "eval"]


class TrainingPipeline:
    def __init__(self) -> None:
        self.intent  = IntentAgent()
        self.data    = DataAgent()
        self.model   = ModelAgent()
        self.train   = TrainAgent()
        self.eval    = EvalAgent()
        self.deploy  = DeployAgent()

    async def run_streaming(
        self,
        user_intent: str,
        dataset_path: str | None = None,
        hyperparameter_overrides: dict | None = None,
    ) -> AsyncIterator[AgentResult]:
        context = AgentContext(
            run_id=str(uuid.uuid4()),
            user_intent=user_intent,
            dataset_path=dataset_path,
            hyperparameter_overrides=hyperparameter_overrides or {},
        )

        agents = [self.intent, self.data, self.model, self.train, self.eval, self.deploy]
        last_result: AgentResult | None = None

        for agent in agents:
            try:
                # All agents expose run_stream(); TrainAgent overrides it
                # to emit keepalive events during long training runs.
                async for result in agent.run_stream(context):
                    last_result = result
                    yield result
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
                )
                return

            # After all results from this agent: check whether to continue
            if last_result is None or last_result.next_agent is None:
                return
