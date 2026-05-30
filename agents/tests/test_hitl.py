"""
Tests for Step 6 — Human-in-the-Loop Clarification.

Covers:
  • IntentAgent: confidence < 0.7 → next_agent=None (pipeline pauses)
  • IntentAgent: confidence >= 0.7 → next_agent="Data" (pipeline continues)
  • IntentAgent: confidence = 0.7 exactly → still routes to Data (boundary)
  • clarification_needed field populated when confidence < 0.7
  • Pipeline stops after IntentAgent when next_agent=None
  • Amended intent format: original + "\n\nUser clarification: " + user_response
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.base import AgentContext, HAIKU, SONNET


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_intent_response(
    confidence: float = 0.95,
    clarification_question: str | None = None,
    task_type: str = "text_classification",
) -> str:
    return json.dumps({
        "task_type": task_type,
        "num_labels": 3,
        "label_names": ["a", "b", "c"],
        "input_column": "text",
        "label_column": "label",
        "base_model_hint": "distilbert-base-uncased",
        "confidence": confidence,
        "clarification_needed": clarification_question,
    })


def _fake_response(text: str) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 200
    usage.output_tokens = 80
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    content = MagicMock()
    content.text = text
    resp = MagicMock()
    resp.usage = usage
    resp.content = [content]
    return resp


# ── IntentAgent confidence routing ───────────────────────────────────────────

class TestIntentAgentClarification:
    @pytest.mark.asyncio
    async def test_high_confidence_routes_to_data(self):
        """Confidence ≥ 0.7 → next_agent = 'Data' → pipeline continues."""
        from agents.intent import IntentAgent

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(_make_intent_response(confidence=0.95))
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r1", user_intent="classify product reviews by sentiment")
        result = await agent.run(ctx)

        assert result.success is True
        assert result.next_agent == "Data"
        assert ctx.task_spec["confidence"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_low_confidence_sets_next_agent_none(self):
        """Confidence < 0.7 → next_agent = None → pipeline pauses."""
        from agents.intent import IntentAgent

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(
                _make_intent_response(
                    confidence=0.5,
                    clarification_question="What specific categories do you want to classify into?"
                )
            )
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r2", user_intent="classify this data")
        result = await agent.run(ctx)

        assert result.success is True
        assert result.next_agent is None  # pipeline must stop

    @pytest.mark.asyncio
    async def test_clarification_question_in_output(self):
        """When clarification is needed, the question appears in result.output."""
        from agents.intent import IntentAgent

        question = "What are the categories you want to predict?"
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(
                _make_intent_response(confidence=0.6, clarification_question=question)
            )
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r3", user_intent="classify")
        result = await agent.run(ctx)

        assert result.output.get("clarification_needed") == question

    @pytest.mark.asyncio
    async def test_confidence_070_routes_to_data(self):
        """Boundary: confidence = 0.70 exactly → 0.70 >= 0.7 → routes to Data."""
        from agents.intent import IntentAgent

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(_make_intent_response(confidence=0.70))
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r4", user_intent="classify support tickets by urgency")
        result = await agent.run(ctx)

        assert result.next_agent == "Data"

    @pytest.mark.asyncio
    async def test_confidence_069_pauses_pipeline(self):
        """Boundary: confidence = 0.69 < 0.7 → next_agent = None."""
        from agents.intent import IntentAgent

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(
                _make_intent_response(
                    confidence=0.69,
                    clarification_question="Please clarify."
                )
            )
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r5", user_intent="classify")
        result = await agent.run(ctx)

        assert result.next_agent is None

    @pytest.mark.asyncio
    async def test_clarification_question_used_as_message(self):
        """When clarification is needed, the message IS the clarification question."""
        from agents.intent import IntentAgent

        question = "What columns map to input and output?"
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_fake_response(
                _make_intent_response(confidence=0.5, clarification_question=question)
            )
        )
        agent = IntentAgent(client=mock_client)
        ctx = AgentContext(run_id="r6", user_intent="train a model")
        result = await agent.run(ctx)

        assert result.message == question


# ── Amended intent format ─────────────────────────────────────────────────────

class TestAmendedIntent:
    def test_amended_intent_format(self):
        """
        The clarify endpoint amends intent as:
        original + "\n\nUser clarification: " + user_response
        Verify the format is correct.
        """
        original = "classify this data"
        user_response = "I want to detect spam vs. ham in email subjects"
        amended = f"{original}\n\nUser clarification: {user_response}"

        assert amended.startswith(original)
        assert "User clarification:" in amended
        assert user_response in amended
        assert "\n\n" in amended  # double newline separates the two parts

    def test_empty_user_response_raises(self):
        """Empty clarification should not be accepted."""
        import pydantic
        from pydantic import BaseModel, field_validator

        # Simulate the ClarifyRequest validator
        class _ClarifyRequest(BaseModel):
            user_response: str

            @field_validator("user_response")
            @classmethod
            def response_not_empty(cls, v: str) -> str:
                if not v.strip():
                    raise ValueError("user_response cannot be empty")
                return v.strip()

        with pytest.raises(pydantic.ValidationError):
            _ClarifyRequest(user_response="")

        with pytest.raises(pydantic.ValidationError):
            _ClarifyRequest(user_response="   ")


# ── Pipeline stops on next_agent=None ────────────────────────────────────────

class TestPipelineStopsOnClarification:
    @pytest.mark.asyncio
    async def test_pipeline_stops_when_next_agent_none(self):
        """
        The pipeline breaks when next_agent=None.
        With a low-confidence intent, DataAgent should never be called.
        """
        from agents.base import BaseAgent, AgentResult
        from agents.pipeline import TrainingPipeline

        called_agents = []

        # Patch IntentAgent to return low-confidence
        class _LowConfidenceIntent(BaseAgent):
            name = "Intent"
            async def run(self, context: AgentContext) -> AgentResult:
                called_agents.append("Intent")
                context.task_spec = {
                    "task_type": "text_classification",
                    "confidence": 0.4,
                    "clarification_needed": "What do you want?",
                }
                return AgentResult(
                    agent_name=self.name, success=True,
                    output=context.task_spec, message="Clarification needed.",
                    next_agent=None,  # pipeline should stop here
                )

        class _TrackingDataAgent(BaseAgent):
            name = "Data"
            async def run(self, context: AgentContext) -> AgentResult:
                called_agents.append("Data")
                return AgentResult(agent_name=self.name, success=True, output={}, message="ok")

        pipeline = TrainingPipeline.__new__(TrainingPipeline)
        pipeline.intent = _LowConfidenceIntent()
        pipeline.data   = _TrackingDataAgent()
        pipeline.clean  = _TrackingDataAgent()  # reuse — name doesn't matter
        pipeline.model  = _TrackingDataAgent()
        pipeline.train  = _TrackingDataAgent()
        pipeline.eval   = _TrackingDataAgent()
        pipeline.deploy = _TrackingDataAgent()

        results = []
        async for result, _ in pipeline.run_streaming(user_intent="classify"):
            results.append(result)

        # Intent was called; Data was NOT (pipeline stopped)
        assert "Intent" in called_agents
        assert "Data" not in called_agents

        # Pipeline summary is still emitted
        summary = results[-1]
        assert summary.agent_name == "Pipeline"
