"""
Tests for Step 1 — Observability Foundation.

Covers:
  • StageMetrics fields and cost computation
  • BaseAgent._chat() populates last_stage_metrics correctly
  • Cache hit ratio computed without division-by-zero on cold cache
  • pipeline._build_pipeline_summary() aggregates correctly
  • TrainingInsightsAnalyzer detects overfitting, divergence, stagnation
  • TrainingInsightsAnalyzer handles empty / single-entry epoch_metrics safely
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.base import (
    HAIKU,
    SONNET,
    OPUS,
    StageMetrics,
    _compute_cost,
    _COST_PER_M,
    _DEFAULT_COST,
)
from agents.pipeline import _build_pipeline_summary, _metrics_to_dict
from agents.training_insights import TrainingInsightsAnalyzer, _slope


# ── Cost computation ──────────────────────────────────────────────────────────

class TestComputeCost:
    def test_haiku_zero_tokens_is_zero(self):
        assert _compute_cost(HAIKU, 0, 0, 0, 0) == 0.0

    def test_sonnet_input_only(self):
        # 1M input tokens at $3.00/M → $3.00
        cost = _compute_cost(SONNET, 1_000_000, 0, 0, 0)
        assert abs(cost - 3.0) < 1e-6

    def test_sonnet_output_only(self):
        # 1M output tokens at $15.00/M → $15.00
        cost = _compute_cost(SONNET, 0, 1_000_000, 0, 0)
        assert abs(cost - 15.0) < 1e-6

    def test_sonnet_cache_read_cheaper_than_input(self):
        # Cache read ($0.30/M) < input ($3.00/M)
        cache_cost = _compute_cost(SONNET, 0, 0, 1_000_000, 0)
        input_cost  = _compute_cost(SONNET, 1_000_000, 0, 0, 0)
        assert cache_cost < input_cost

    def test_haiku_cheaper_than_sonnet(self):
        cost_haiku  = _compute_cost(HAIKU,  10_000, 2_000, 0, 0)
        cost_sonnet = _compute_cost(SONNET, 10_000, 2_000, 0, 0)
        assert cost_haiku < cost_sonnet

    def test_opus_most_expensive(self):
        cost_haiku  = _compute_cost(HAIKU,  10_000, 2_000, 0, 0)
        cost_sonnet = _compute_cost(SONNET, 10_000, 2_000, 0, 0)
        cost_opus   = _compute_cost(OPUS,   10_000, 2_000, 0, 0)
        assert cost_haiku < cost_sonnet < cost_opus

    def test_unknown_model_falls_back_to_sonnet_rates(self):
        cost_unknown = _compute_cost("unknown-model", 1_000_000, 0, 0, 0)
        cost_sonnet  = _compute_cost(SONNET,           1_000_000, 0, 0, 0)
        assert abs(cost_unknown - cost_sonnet) < 1e-9


# ── StageMetrics via BaseAgent._chat() ───────────────────────────────────────

def _make_fake_response(
    input_tokens: int = 500,
    output_tokens: int = 100,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    text: str = '{"result": "ok"}',
):
    """Build a minimal mock that looks like an Anthropic Messages response."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read_tokens
    usage.cache_creation_input_tokens = cache_creation_tokens

    content = MagicMock()
    content.text = text

    response = MagicMock()
    response.usage = usage
    response.content = [content]
    return response


@pytest.mark.asyncio
async def test_stage_metrics_populated_after_chat():
    """_chat() must populate last_stage_metrics with correct token counts."""
    from agents.base import BaseAgent, AgentContext, AgentResult

    class _DummyAgent(BaseAgent):
        name = "Dummy"
        model = SONNET

        async def run(self, context: AgentContext) -> AgentResult:
            await self._chat("sys", [{"role": "user", "content": "hi"}])
            return AgentResult(agent_name=self.name, success=True, output={}, message="ok")

    fake_response = _make_fake_response(
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=800,
        cache_creation_tokens=50,
    )

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=fake_response)

    agent = _DummyAgent(client=mock_client)
    ctx = AgentContext(run_id="t1", user_intent="test")
    await agent.run(ctx)

    m = agent.last_stage_metrics
    assert m is not None
    assert m.agent_name == "Dummy"
    assert m.model == SONNET
    assert m.input_tokens == 1000
    assert m.output_tokens == 200
    assert m.cache_read_tokens == 800
    assert m.cache_write_tokens == 50
    assert m.latency_ms >= 0  # mock returns instantly; real calls will be > 0
    assert m.estimated_cost_usd > 0
    assert m.cache_hit_ratio == pytest.approx(800 / 1000)


@pytest.mark.asyncio
async def test_cache_hit_ratio_zero_on_cold_cache():
    """cache_hit_ratio must be 0.0 when cache_read_input_tokens is 0 (cold call)."""
    from agents.base import BaseAgent, AgentContext, AgentResult

    class _DummyAgent(BaseAgent):
        name = "Dummy"
        model = SONNET

        async def run(self, context: AgentContext) -> AgentResult:
            await self._chat("sys", [{"role": "user", "content": "hi"}])
            return AgentResult(agent_name=self.name, success=True, output={}, message="ok")

    fake_response = _make_fake_response(input_tokens=400, output_tokens=80, cache_read_tokens=0)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=fake_response)

    agent = _DummyAgent(client=mock_client)
    ctx = AgentContext(run_id="t2", user_intent="test")
    await agent.run(ctx)

    assert agent.last_stage_metrics is not None
    assert agent.last_stage_metrics.cache_hit_ratio == 0.0


@pytest.mark.asyncio
async def test_no_division_by_zero_on_zero_input_tokens():
    """cache_hit_ratio must not raise ZeroDivisionError when input_tokens is 0."""
    from agents.base import BaseAgent, AgentContext, AgentResult

    class _DummyAgent(BaseAgent):
        name = "Dummy"
        model = HAIKU

        async def run(self, context: AgentContext) -> AgentResult:
            await self._chat("sys", [{"role": "user", "content": "hi"}])
            return AgentResult(agent_name=self.name, success=True, output={}, message="ok")

    fake_response = _make_fake_response(input_tokens=0, output_tokens=10, cache_read_tokens=0)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=fake_response)

    agent = _DummyAgent(client=mock_client)
    ctx = AgentContext(run_id="t3", user_intent="test")
    await agent.run(ctx)

    assert agent.last_stage_metrics is not None
    # Should be 0/max(0,1) = 0.0 — no ZeroDivisionError
    assert agent.last_stage_metrics.cache_hit_ratio == 0.0


# ── _build_pipeline_summary ───────────────────────────────────────────────────

class TestBuildPipelineSummary:
    def _make_metrics(
        self,
        agent_name: str,
        input_t: int,
        output_t: int,
        cache_read: int = 0,
        cost: float = 0.0,
    ) -> StageMetrics:
        return StageMetrics(
            agent_name=agent_name,
            model=SONNET,
            input_tokens=input_t,
            output_tokens=output_t,
            cache_read_tokens=cache_read,
            cache_write_tokens=0,
            latency_ms=123.0,
            timestamp="2026-01-01T00:00:00Z",
            estimated_cost_usd=cost,
            cache_hit_ratio=cache_read / max(input_t, 1),
        )

    def test_empty_metrics_returns_zeros(self):
        result = _build_pipeline_summary([])
        assert result["total_cost_usd"] == 0.0
        assert result["total_input_tokens"] == 0
        assert result["llm_stages_called"] == 0
        assert result["per_stage"] == []

    def test_sums_tokens_correctly(self):
        metrics = [
            self._make_metrics("Intent",  input_t=300, output_t=50),
            self._make_metrics("Model",   input_t=700, output_t=150),
            self._make_metrics("Eval",    input_t=500, output_t=200),
        ]
        result = _build_pipeline_summary(metrics)
        assert result["total_input_tokens"] == 1500
        assert result["total_output_tokens"] == 400
        assert result["llm_stages_called"] == 3

    def test_cache_hit_ratio_aggregated_correctly(self):
        # 600 cache reads out of 1000 total input = 60%
        metrics = [
            self._make_metrics("Intent", input_t=500, output_t=50, cache_read=300),
            self._make_metrics("Eval",   input_t=500, output_t=100, cache_read=300),
        ]
        result = _build_pipeline_summary(metrics)
        assert result["overall_cache_hit_ratio"] == pytest.approx(600 / 1000)

    def test_total_cost_summed(self):
        metrics = [
            self._make_metrics("Intent", input_t=100, output_t=20, cost=0.001),
            self._make_metrics("Eval",   input_t=200, output_t=50, cost=0.004),
        ]
        result = _build_pipeline_summary(metrics)
        assert result["total_cost_usd"] == pytest.approx(0.005, rel=1e-4)

    def test_per_stage_length_matches_metrics(self):
        metrics = [self._make_metrics(f"Agent{i}", 100, 20) for i in range(4)]
        result = _build_pipeline_summary(metrics)
        assert len(result["per_stage"]) == 4


# ── TrainingInsightsAnalyzer ─────────────────────────────────────────────────

class TestTrainingInsightsAnalyzer:
    def setup_method(self):
        self.analyzer = TrainingInsightsAnalyzer()

    def _make_epochs(self, losses: list[float], eval_losses: list[float] | None = None) -> list[dict]:
        epochs = []
        for i, loss in enumerate(losses):
            entry: dict[str, Any] = {"epoch": i, "step": i, "loss": loss}
            if eval_losses and i < len(eval_losses):
                entry["eval_loss"] = eval_losses[i]
            epochs.append(entry)
        return epochs

    def test_empty_metrics_returns_no_issues(self):
        ins = self.analyzer.analyze([])
        assert not ins.overfitting_detected
        assert not ins.divergence_detected
        assert not ins.stagnation_detected
        assert ins.warnings == []

    def test_single_epoch_no_false_positives(self):
        ins = self.analyzer.analyze(self._make_epochs([0.5]))
        assert not ins.overfitting_detected
        assert not ins.divergence_detected
        assert not ins.stagnation_detected

    def test_nan_loss_triggers_divergence(self):
        ins = self.analyzer.analyze(self._make_epochs([0.5, 0.4, float("nan")]))
        assert ins.divergence_detected
        assert any("NaN" in w or "Inf" in w for w in ins.warnings)

    def test_inf_loss_triggers_divergence(self):
        ins = self.analyzer.analyze(self._make_epochs([0.5, float("inf")]))
        assert ins.divergence_detected

    def test_exploding_loss_triggers_divergence(self):
        # Loss above threshold (100.0)
        ins = self.analyzer.analyze(self._make_epochs([0.5, 0.4, 0.3, 150.0]))
        assert ins.divergence_detected

    def test_normal_decreasing_loss_no_divergence(self):
        ins = self.analyzer.analyze(self._make_epochs([1.0, 0.8, 0.6, 0.4, 0.2]))
        assert not ins.divergence_detected

    def test_overfitting_detected(self):
        # Train loss going down, eval loss going up — needs ≥4 epochs
        train = [1.0, 0.8, 0.6, 0.4, 0.3]
        val   = [1.0, 1.1, 1.2, 1.4, 1.6]
        ins = self.analyzer.analyze(self._make_epochs(train, val))
        assert ins.overfitting_detected

    def test_both_losses_decreasing_no_overfit(self):
        train = [1.0, 0.8, 0.6, 0.4, 0.3]
        val   = [1.1, 0.9, 0.7, 0.5, 0.4]
        ins = self.analyzer.analyze(self._make_epochs(train, val))
        assert not ins.overfitting_detected

    def test_stagnation_detected(self):
        # All losses essentially flat
        flat = [0.500, 0.500, 0.500, 0.500, 0.500]
        ins = self.analyzer.analyze(self._make_epochs(flat))
        assert ins.stagnation_detected

    def test_no_stagnation_with_decreasing_loss(self):
        decreasing = [1.0, 0.8, 0.6, 0.4, 0.2]
        ins = self.analyzer.analyze(self._make_epochs(decreasing))
        assert not ins.stagnation_detected

    def test_suggestions_populated_when_warnings_exist(self):
        flat = [0.500, 0.500, 0.500, 0.500, 0.500]
        ins = self.analyzer.analyze(self._make_epochs(flat))
        assert len(ins.suggestions) > 0

    def test_to_dict_has_all_expected_keys(self):
        ins = self.analyzer.analyze([])
        d = ins.to_dict()
        assert "overfitting_detected" in d
        assert "divergence_detected" in d
        assert "stagnation_detected" in d
        assert "warnings" in d
        assert "suggestions" in d


# ── slope helper ─────────────────────────────────────────────────────────────

class TestSlope:
    def test_flat_returns_zero(self):
        assert _slope([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_perfectly_increasing(self):
        assert _slope([0.0, 1.0, 2.0, 3.0]) > 0.0

    def test_perfectly_decreasing(self):
        assert _slope([3.0, 2.0, 1.0, 0.0]) < 0.0

    def test_single_element_returns_zero(self):
        assert _slope([42.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _slope([]) == 0.0
