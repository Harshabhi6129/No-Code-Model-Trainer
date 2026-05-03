from __future__ import annotations

import abc
import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

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
    deploy_result: dict[str, Any] = field(default_factory=dict)
    hyperparameter_overrides: dict[str, Any] = field(default_factory=dict)


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

    async def run_stream(self, context: AgentContext) -> AsyncIterator[AgentResult]:
        """Default: single result. Override in agents that emit progress events."""
        yield await self.run(context)

    async def stream(self, context: AgentContext) -> AsyncIterator[str]:
        result = await self.run(context)
        yield result.message

    async def _chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> str:
        """
        Call Claude with automatic retry on rate-limit and connection errors.
        max_tokens defaults to 4096; set higher for long-form outputs (model cards).
        """
        import anthropic as _anthropic

        for attempt in range(3):
            try:
                response = await self.client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                return response.content[0].text  # type: ignore[union-attr]
            except (_anthropic.RateLimitError, _anthropic.APIConnectionError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    "%s: Claude API %s on attempt %d — retrying in %ds",
                    self.name, type(exc).__name__, attempt + 1, wait,
                )
                await asyncio.sleep(wait)
            except Exception:
                raise

        raise RuntimeError(f"{self.name}: Claude API unavailable after 3 retries")
