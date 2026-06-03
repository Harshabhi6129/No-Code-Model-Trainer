"""
IntentAgent unit tests — covering gaps not addressed by test_hitl.py
(clarification routing) or test_model_routing.py (Haiku/Sonnet routing).

Covers:
  • Malformed / non-JSON LLM response → AgentResult(success=False)
  • Partial JSON (missing required fields) → AgentResult(success=False)
  • context.task_spec is populated on success
  • NER intent → task_type="token_classification", label_names extracted
  • Result message contains task_type on success
  • base_model_hint appears in result output
  • System prompt is ≥ 1 024 tokens (cache eligibility)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.base import AgentContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_response(text: str) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 300
    usage.output_tokens = 60
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.usage = usage
    resp.content = [content]
    return resp


def _valid_json(
    task_type: str = "text_classification",
    num_labels: int = 3,
    label_names: list[str] | None = None,
    input_col: str = "text",
    label_col: str = "label",
    base_model: str = "distilbert-base-uncased",
    confidence: float = 0.92,
    clarification: str | None = None,
) -> str:
    return json.dumps({
        "task_type": task_type,
        "num_labels": num_labels,
        "label_names": label_names or ["a", "b", "c"],
        "input_column": input_col,
        "label_column": label_col,
        "base_model_hint": base_model,
        "confidence": confidence,
        "clarification_needed": clarification,
    })


def _make_agent(response_text: str):
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=_fake_response(response_text))
    from agents.intent import IntentAgent
    return IntentAgent(client=mock_client)


# ── JSON parse failures ───────────────────────────────────────────────────────

class TestIntentAgentParseFailures:
    @pytest.mark.asyncio
    async def test_plain_text_response_returns_failure(self):
        """LLM returns prose instead of JSON → AgentResult(success=False)."""
        agent = _make_agent("Sure! I'll classify your emails.")
        ctx = AgentContext(run_id="r1", user_intent="classify emails")
        result = await agent.run(ctx)

        assert result.success is False
        assert ctx.task_spec == {} or ctx.task_spec is None or not ctx.task_spec

    @pytest.mark.asyncio
    async def test_empty_response_returns_failure(self):
        """Empty LLM response → AgentResult(success=False)."""
        agent = _make_agent("")
        ctx = AgentContext(run_id="r2", user_intent="classify emails")
        result = await agent.run(ctx)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_partial_json_missing_task_type_returns_failure(self):
        """JSON missing required 'task_type' field → parse/validation fails."""
        bad_json = json.dumps({
            "num_labels": 2,
            "confidence": 0.9,
        })
        agent = _make_agent(bad_json)
        ctx = AgentContext(run_id="r3", user_intent="classify emails")
        result = await agent.run(ctx)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json_is_tolerated_or_fails_gracefully(self):
        """
        LLM sometimes wraps JSON in ```json ... ``` fences.
        The agent should either strip them (works) or fail gracefully (success=False).
        Either is acceptable — it must NOT raise an unhandled exception.
        """
        wrapped = "```json\n" + _valid_json() + "\n```"
        agent = _make_agent(wrapped)
        ctx = AgentContext(run_id="r4", user_intent="classify support tickets")
        result = await agent.run(ctx)  # must not raise

        # Either success (agent strips fences) or graceful failure — not an exception
        assert isinstance(result.success, bool)


# ── Successful parse ──────────────────────────────────────────────────────────

class TestIntentAgentSuccessfulParse:
    @pytest.mark.asyncio
    async def test_context_task_spec_populated(self):
        """After a successful run, context.task_spec must be populated."""
        agent = _make_agent(_valid_json())
        ctx = AgentContext(run_id="r5", user_intent="classify support tickets by urgency")
        result = await agent.run(ctx)

        assert result.success is True
        assert ctx.task_spec is not None
        assert ctx.task_spec.get("task_type") == "text_classification"

    @pytest.mark.asyncio
    async def test_result_output_contains_task_type(self):
        """result.output must carry back the full parsed spec dict."""
        agent = _make_agent(_valid_json(task_type="text_classification"))
        ctx = AgentContext(run_id="r6", user_intent="classify sentiment")
        result = await agent.run(ctx)

        assert result.output.get("task_type") == "text_classification"

    @pytest.mark.asyncio
    async def test_result_output_contains_base_model_hint(self):
        """base_model_hint must appear in result.output."""
        agent = _make_agent(_valid_json(base_model="roberta-base"))
        ctx = AgentContext(run_id="r7", user_intent="classify product reviews")
        result = await agent.run(ctx)

        assert result.output.get("base_model_hint") == "roberta-base"

    @pytest.mark.asyncio
    async def test_success_message_mentions_task_type(self):
        """On success, the message should describe the task type."""
        agent = _make_agent(_valid_json(task_type="text_classification"))
        ctx = AgentContext(run_id="r8", user_intent="classify emails")
        result = await agent.run(ctx)

        assert result.success is True
        # Message should mention the task label
        assert "classification" in result.message.lower() or "text" in result.message.lower()

    @pytest.mark.asyncio
    async def test_next_agent_is_data_on_high_confidence(self):
        """High confidence → next_agent == 'Data'."""
        agent = _make_agent(_valid_json(confidence=0.9))
        ctx = AgentContext(run_id="r9", user_intent="classify support tickets")
        result = await agent.run(ctx)

        assert result.next_agent == "Data"


# ── NER task type ─────────────────────────────────────────────────────────────

class TestIntentAgentNER:
    @pytest.mark.asyncio
    async def test_ner_task_type_parsed(self):
        """NER intent → task_type='token_classification' stored in context."""
        agent = _make_agent(_valid_json(
            task_type="token_classification",
            label_names=["PER", "ORG", "LOC"],
            input_col="tokens",
            label_col="ner_tags",
            base_model="dslim/bert-base-NER",
            confidence=0.9,
        ))
        ctx = AgentContext(run_id="r10", user_intent="extract named entities from news articles")
        result = await agent.run(ctx)

        assert result.success is True
        assert ctx.task_spec["task_type"] == "token_classification"
        assert ctx.task_spec["label_names"] == ["PER", "ORG", "LOC"]
        assert ctx.task_spec["input_column"] == "tokens"
        assert ctx.task_spec["label_column"] == "ner_tags"

    @pytest.mark.asyncio
    async def test_ner_routes_to_data_agent(self):
        """Token classification with high confidence should also route to Data."""
        agent = _make_agent(_valid_json(
            task_type="token_classification",
            confidence=0.88,
        ))
        ctx = AgentContext(run_id="r11", user_intent="NER on clinical notes")
        result = await agent.run(ctx)

        assert result.next_agent == "Data"


# ── System prompt cache eligibility ──────────────────────────────────────────

class TestIntentAgentSystemPrompt:
    def test_system_prompt_meets_cache_minimum(self):
        """
        Anthropic's prompt caching requires ≥ 1 024 tokens in the system prompt.
        We use a character-count proxy: 1 token ≈ 4 characters (conservative
        lower-bound for English text).  1 024 tokens ≈ 4 096 characters.
        """
        from agents.intent import SYSTEM

        # 4 096 chars ≈ 1 024 tokens at 4 chars/token (conservative lower bound).
        min_chars = 4_096
        assert len(SYSTEM) >= min_chars, (
            f"SYSTEM prompt is only {len(SYSTEM)} characters (~{len(SYSTEM)//4} tokens) — "
            f"under the 1 024-token cache minimum. Expand it so cache_system=True "
            "actually reduces API costs."
        )
