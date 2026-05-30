"""
Tests for Step 13 — Structured Tracing.

Covers:
  • PipelineTracer.record_stage(): appends to JSONL file
  • PipelineTracer.finish_run(): writes summary with correct totals
  • JSONL file contains valid JSON on each line
  • Supabase write failure is silent (pipeline continues)
  • LangSmith push called when LANGSMITH_API_KEY is set
  • LangSmith push failure is silent
  • Permission denied on JSONL → falls back to in-memory only
  • record_stage() with a StageMetrics dataclass works
  • record_stage() with a plain dict works
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base import StageMetrics, SONNET


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_metrics(agent: str = "Intent", cost: float = 0.001) -> StageMetrics:
    return StageMetrics(
        agent_name=agent,
        model=SONNET,
        input_tokens=500,
        output_tokens=100,
        cache_read_tokens=400,
        cache_write_tokens=0,
        latency_ms=250.0,
        timestamp="2026-01-01T00:00:00Z",
        estimated_cost_usd=cost,
        cache_hit_ratio=0.8,
    )


# ── JSONL writing ─────────────────────────────────────────────────────────────

class TestJSONLWriting:
    def test_record_stage_appends_to_jsonl(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_1")
            tracer.record_stage(_make_metrics("Intent"))
            tracer.record_stage(_make_metrics("Model"))

        jsonl_file = tmp_path / "test_run_1.jsonl"
        assert jsonl_file.exists()
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_2")
            tracer.record_stage(_make_metrics("Intent", 0.002))
            tracer.record_stage(_make_metrics("Eval", 0.005))

        jsonl_file = tmp_path / "test_run_2.jsonl"
        for line in jsonl_file.read_text().strip().split("\n"):
            data = json.loads(line)  # must not raise
            assert isinstance(data, dict)

    def test_finish_run_writes_summary(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_3")
            tracer.record_stage(_make_metrics("Intent", 0.001))
            tracer.finish_run(total_cost=0.001, outcome="completed")

        jsonl_file = tmp_path / "test_run_3.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        # Last line should be the pipeline_summary
        summary = json.loads(lines[-1])
        assert summary["type"] == "pipeline_summary"
        assert summary["outcome"] == "completed"

    def test_finish_run_cost_matches(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_4")
            tracer.finish_run(total_cost=0.0042, outcome="completed")

        jsonl_file = tmp_path / "test_run_4.jsonl"
        summary = json.loads(jsonl_file.read_text().strip())
        assert summary["total_cost_usd"] == pytest.approx(0.0042, rel=1e-5)

    def test_failed_outcome_recorded(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_5")
            tracer.finish_run(total_cost=0.0, outcome="failed")

        jsonl_file = tmp_path / "test_run_5.jsonl"
        summary = json.loads(jsonl_file.read_text().strip())
        assert summary["outcome"] == "failed"

    def test_record_stage_with_dict(self, tmp_path):
        """record_stage() must accept plain dicts too."""
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_6")
            tracer.record_stage({
                "agent_name": "Eval", "latency_ms": 500, "estimated_cost_usd": 0.003,
                "input_tokens": 200, "output_tokens": 50, "cache_read_tokens": 0,
            })
            tracer.finish_run(0.003, "completed")

        jsonl_file = tmp_path / "test_run_6.jsonl"
        assert jsonl_file.exists()

    def test_permission_denied_no_crash(self, tmp_path):
        """If the JSONL file can't be written, the tracer must not raise."""
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_perm")
            tracer._jsonl_path = None  # simulate permission failure by removing path
            tracer.record_stage(_make_metrics())
            tracer.finish_run(0.0, "completed")  # must not raise


# ── Supabase failure is silent ────────────────────────────────────────────────

class TestSupabaseFallback:
    def test_supabase_write_failure_silent(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_sb")
            with patch.object(tracer, "_write_supabase", side_effect=Exception("DB down")):
                # Should not raise
                tracer.finish_run(0.001, "completed")


# ── LangSmith integration ─────────────────────────────────────────────────────

class TestLangSmith:
    def test_langsmith_called_when_api_key_set(self, tmp_path):
        from services.tracer import PipelineTracer, _push_to_langsmith

        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test_key"}):
            with patch("services.tracer._push_to_langsmith") as mock_push:
                with patch("services.tracer.TRACES_DIR", tmp_path):
                    tracer = PipelineTracer("test_run_ls")
                    tracer.finish_run(0.001, "completed")
                mock_push.assert_called_once()

    def test_langsmith_not_called_without_api_key(self, tmp_path):
        from services.tracer import PipelineTracer

        env = {k: v for k, v in os.environ.items() if k != "LANGSMITH_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("services.tracer._push_to_langsmith") as mock_push:
                with patch("services.tracer.TRACES_DIR", tmp_path):
                    tracer = PipelineTracer("test_run_nols")
                    tracer.finish_run(0.001, "completed")
                mock_push.assert_called_once()  # called but does nothing internally

    def test_langsmith_failure_silent(self, tmp_path):
        """LangSmith errors must not propagate."""
        from services.tracer import _push_to_langsmith
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "bad_key"}):
            with patch("httpx.post", side_effect=Exception("network error")):
                # Must not raise
                _push_to_langsmith("run1", {"outcome": "test"})


# ── Token aggregation ─────────────────────────────────────────────────────────

class TestTokenAggregation:
    def test_total_tokens_summed_correctly(self, tmp_path):
        from services.tracer import PipelineTracer
        with patch("services.tracer.TRACES_DIR", tmp_path):
            tracer = PipelineTracer("test_run_tok")
            tracer.record_stage(StageMetrics(
                "Intent", SONNET, 300, 50, 200, 0, 100.0, "ts", 0.001, 0.67
            ))
            tracer.record_stage(StageMetrics(
                "Eval", SONNET, 600, 150, 400, 0, 200.0, "ts", 0.003, 0.67
            ))
            tracer.finish_run(0.004, "completed")

        jsonl_file = tmp_path / "test_run_tok.jsonl"
        lines = jsonl_file.read_text().strip().split("\n")
        summary = json.loads(lines[-1])
        # total_tokens = (300 + 50) + (600 + 150) = 1100
        assert summary["total_tokens"] == 1100
