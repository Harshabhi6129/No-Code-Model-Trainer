from __future__ import annotations
import uuid
from typing import AsyncIterator
from .base import AgentContext, AgentResult
from .intent import IntentAgent
from .data import DataAgent
from .model import ModelAgent


class TrainingPipeline:
    def __init__(self) -> None:
        self.intent = IntentAgent()
        self.data = DataAgent()
        self.model = ModelAgent()

    async def run_streaming(
        self, user_intent: str, dataset_path: str | None = None,
    ) -> AsyncIterator[AgentResult]:
        context = AgentContext(run_id=str(uuid.uuid4()), user_intent=user_intent, dataset_path=dataset_path)
        for agent in [self.intent, self.data, self.model]:
            result = await agent.run(context)
            yield result
            if not result.success or result.next_agent is None:
                return
