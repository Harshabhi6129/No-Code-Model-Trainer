from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, ValidationError

_M = TypeVar("_M", bound=BaseModel)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)

# ── Model tier constants ────────────────────────────────────────────────────
# Route each agent to the cheapest model capable of its task.
# Research: intelligent routing achieves ~85% cost reduction with <5% quality delta.

HAIKU  = "claude-haiku-4-5-20251001"  # fast + cheap: formatting, routing, validation
SONNET = "claude-sonnet-4-6"          # balanced: reasoning, analysis, narration
OPUS   = "claude-opus-4-7"            # powerful: reserved for complex planning (future)

# Legacy env-var still works as a global override (useful for testing)
_ENV_OVERRIDE: str = os.getenv("ANTHROPIC_MODEL", "").strip()


# ── Pipeline context ────────────────────────────────────────────────────────

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
    hf_token: str | None = None
    # Tracks completed stages for checkpoint/resume (Phase A3)
    completed_stages: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    output: dict[str, Any]
    message: str
    next_agent: str | None = None


# ── Base agent ──────────────────────────────────────────────────────────────

class BaseAgent(abc.ABC):
    name: str
    # Each subclass declares its tier. SONNET is the sensible default.
    model: str = SONNET

    def __init__(self, client: "anthropic.AsyncAnthropic | None" = None) -> None:
        if client is None:
            import anthropic as _anthropic
            client = _anthropic.AsyncAnthropic()
        self.client = client
        # Env override wins (useful for CI / cost testing)
        self._resolved_model = _ENV_OVERRIDE or self.model

    @abc.abstractmethod
    async def run(self, context: AgentContext) -> AgentResult: ...

    async def run_stream(self, context: AgentContext) -> AsyncIterator[AgentResult]:
        """Default: single result. TrainAgent overrides to emit live progress."""
        yield await self.run(context)

    async def stream(self, context: AgentContext) -> AsyncIterator[str]:
        result = await self.run(context)
        yield result.message

    # ── Core LLM call with caching + tiering ────────────────────────────────

    async def _chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> str:
        """
        Call the agent's assigned model tier with:
          • Prompt caching on the system prompt (up to 90% cost reduction on hits)
          • Per-agent model tiering (HAIKU / SONNET / OPUS)
          • Exponential backoff on transient API errors

        cache_system=True wraps the system string in Anthropic's cache_control
        list format. Prompts shorter than the provider's cache minimum (~1 024 tokens
        for Sonnet) are accepted without error — they just won't be cached.

        Set cache_system=False for one-shot calls with short, unique system prompts.
        """
        import anthropic as _anthropic

        # System prompt caching: list[TextBlockParam] with cache_control
        if cache_system:
            system_param: Any = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_param = system

        for attempt in range(3):
            try:
                response = await self.client.messages.create(
                    model=self._resolved_model,
                    max_tokens=max_tokens,
                    system=system_param,
                    messages=messages,
                )

                # Emit cache hit ratio at DEBUG level for cost observability
                usage = getattr(response, "usage", None)
                if usage and logger.isEnabledFor(logging.DEBUG):
                    cached = getattr(usage, "cache_read_input_tokens", 0) or 0
                    total  = getattr(usage, "input_tokens", 0) or 1
                    logger.debug(
                        "%s [%s] cache: %d/%d tokens (%.0f%% hit)",
                        self.name, self._resolved_model, cached, total,
                        100 * cached / total,
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

    # ── Shared JSON parse + Pydantic validate ────────────────────────────────

    def _parse_llm_json(
        self,
        raw: str,
        schema: type[_M],
        context_label: str = "",
    ) -> tuple[_M | None, str]:
        """
        Parse raw LLM output as JSON and validate against a Pydantic schema.

        Returns (validated_model, "")  on success.
        Returns (None, error_message)  on parse or validation failure.

        Centralises the parse→validate flow so each agent has zero boilerplate.
        The caller decides whether to retry, fall back, or surface the error.
        """
        text = raw.strip()
        # Strip accidental markdown fences (```json ... ```)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        label = f"{self.name}{f' ({context_label})' if context_label else ''}"

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"{label}: JSON parse failed — {exc}"
            logger.warning(msg)
            return None, msg

        try:
            model = schema.model_validate(data)
            return model, ""
        except ValidationError as exc:
            errs = "; ".join(
                f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            msg = f"{label}: schema validation failed — {errs}"
            logger.warning(msg)
            return None, msg
