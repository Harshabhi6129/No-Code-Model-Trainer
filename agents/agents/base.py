from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

# Readable from env so model can be swapped without a code change
MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip() or "claude-sonnet-4-6"


@dataclass
class AgentContext:
    run_id: str
    user_intent: str
    dataset_path: str | None = None
    task_spec: dict[str, Any] = field(default_factory=dict)
    data_profile: dict[str, Any] = field(default_factory=dict)
    model_recipe: dict[str, Any] = field(default_factory=dict)
    training_result: dict[str, Any] = field(default_factory=dict)
    eval_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    output: dict[str, Any]
    message: str
    next_agent: str | None = None


class BaseAgent(abc.ABC):
    name: str

    def __init__(self, client: "anthropic.AsyncAnthropic | None" = None) -> None:
        if client is None:
            import anthropic as _anthropic
            # AsyncAnthropic so _chat never blocks the event loop
            client = _anthropic.AsyncAnthropic()
        self.client = client

    @abc.abstractmethod
    async def run(self, context: AgentContext) -> AgentResult: ...

    async def stream(self, context: AgentContext) -> AsyncIterator[str]:
        result = await self.run(context)
        yield result.message

    async def _chat(self, system: str, messages: list[dict[str, Any]]) -> str:
        response = await self.client.messages.create(
            model=MODEL, max_tokens=2048, system=system, messages=messages,
        )
        return response.content[0].text  # type: ignore[union-attr]
