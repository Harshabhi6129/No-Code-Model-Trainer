from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

MODEL = "claude-sonnet-4-6"


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

    def __init__(self, client: "anthropic.Anthropic | None" = None) -> None:
        if client is None:
            import anthropic as _anthropic
            client = _anthropic.Anthropic()
        self.client = client

    @abc.abstractmethod
    async def run(self, context: AgentContext) -> AgentResult: ...

    async def stream(self, context: AgentContext) -> AsyncIterator[str]:
        result = await self.run(context)
        yield result.message

    def _chat(self, system: str, messages: list[dict[str, Any]]) -> str:
        response = self.client.messages.create(
            model=MODEL, max_tokens=2048, system=system, messages=messages,
        )
        return response.content[0].text  # type: ignore[union-attr]
