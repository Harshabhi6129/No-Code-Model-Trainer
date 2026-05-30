"""
Structured pipeline tracer for ModelForge.

Emits per-call StageMetrics and per-run pipeline summaries to:
  1. Local JSONL file (agents/traces/{run_id}.jsonl) — always
  2. Supabase traces table — if SUPABASE_URL/KEY are set
  3. LangSmith — if LANGSMITH_API_KEY is set (optional, best-effort)

Design rules:
  - Tracing NEVER blocks training. All writes are best-effort; failures are
    logged at DEBUG level and the pipeline continues.
  - The tracer is injected into AgentContext (or used as a module-level
    singleton) so all agents can record without circular imports.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TRACES_DIR = Path(os.getenv("TRACES_DIR", str(Path(__file__).parent.parent.parent / "agents" / "traces")))
_SUPABASE_TABLE = "traces"


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _supabase_client():
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


# ── LangSmith helper ──────────────────────────────────────────────────────────

def _push_to_langsmith(run_id: str, payload: dict[str, Any]) -> None:
    """Push a run summary to LangSmith. Best-effort, silent on failure."""
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return
    try:
        import httpx
        resp = httpx.post(
            "https://api.smith.langchain.com/runs",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "id":        run_id,
                "name":      "modelforge-pipeline",
                "run_type":  "chain",
                "inputs":    {"run_id": run_id},
                "outputs":   {"summary": payload},
                "end_time":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            timeout=5.0,
        )
        if resp.status_code not in (200, 201, 202):
            logger.debug("LangSmith push returned %d", resp.status_code)
    except Exception as exc:
        logger.debug("LangSmith push failed: %s", exc)


# ── PipelineTracer ────────────────────────────────────────────────────────────

class PipelineTracer:
    """
    Per-run tracer. One instance per pipeline run.

    Usage (in pipeline.py):
        tracer = PipelineTracer(run_id)
        # ... agents emit StageMetrics via BaseAgent._chat() ...
        tracer.record_stage(stage_metrics)
        tracer.finish_run(total_cost, outcome)
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._started_at = time.time()
        self._stage_records: list[dict[str, Any]] = []
        self._jsonl_path: Path | None = None

        # Ensure traces directory exists
        try:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = TRACES_DIR / f"{run_id}.jsonl"
        except Exception as exc:
            logger.debug("PipelineTracer: could not create traces dir: %s", exc)

    def record_stage(self, stage_metrics: Any) -> None:
        """
        Record one StageMetrics entry.
        Accepts the StageMetrics dataclass from agents.base or any dict.
        """
        try:
            if hasattr(stage_metrics, "__dataclass_fields__"):
                record = asdict(stage_metrics)
            elif isinstance(stage_metrics, dict):
                record = stage_metrics
            else:
                record = {
                    "agent_name": str(getattr(stage_metrics, "agent_name", "?")),
                    "latency_ms": float(getattr(stage_metrics, "latency_ms", 0)),
                    "estimated_cost_usd": float(getattr(stage_metrics, "estimated_cost_usd", 0)),
                }
            record["run_id"] = self.run_id
            self._stage_records.append(record)
            self._append_jsonl({"type": "stage", **record})
        except Exception as exc:
            logger.debug("PipelineTracer.record_stage error: %s", exc)

    def finish_run(self, total_cost: float, outcome: str) -> None:
        """
        Emit the run-level summary to JSONL + Supabase + LangSmith.

        outcome: "completed" | "failed" | "cancelled"
        """
        elapsed = time.time() - self._started_at
        n_stages = len(self._stage_records)
        total_tokens = sum(
            r.get("input_tokens", 0) + r.get("output_tokens", 0)
            for r in self._stage_records
        )
        total_cache = sum(r.get("cache_read_tokens", 0) for r in self._stage_records)
        total_input = sum(r.get("input_tokens", 0) for r in self._stage_records)
        cache_hit_ratio = round(total_cache / max(total_input, 1), 4)

        summary = {
            "run_id":          self.run_id,
            "outcome":         outcome,
            "total_cost_usd":  round(total_cost, 8),
            "total_tokens":    total_tokens,
            "cache_hit_ratio": cache_hit_ratio,
            "llm_stages":      n_stages,
            "elapsed_seconds": round(elapsed, 2),
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_stage":       self._stage_records,
        }

        self._append_jsonl({"type": "pipeline_summary", **summary})
        try:
            self._write_supabase(summary)
        except Exception as exc:
            logger.debug("PipelineTracer Supabase finish_run error: %s", exc)
        try:
            _push_to_langsmith(self.run_id, summary)
        except Exception as exc:
            logger.debug("PipelineTracer LangSmith finish_run error: %s", exc)

        logger.info(
            "[%s] Trace complete — cost=$%.6f tokens=%d cache=%.0f%% outcome=%s",
            self.run_id, total_cost, total_tokens, cache_hit_ratio * 100, outcome,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if self._jsonl_path is None:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except PermissionError:
            logger.debug("PipelineTracer: permission denied writing to %s", self._jsonl_path)
        except Exception as exc:
            logger.debug("PipelineTracer._append_jsonl error: %s", exc)

    def _write_supabase(self, summary: dict[str, Any]) -> None:
        client = _supabase_client()
        if client is None:
            return
        try:
            client.table(_SUPABASE_TABLE).upsert({
                "run_id":          summary["run_id"],
                "outcome":         summary["outcome"],
                "total_cost_usd":  summary["total_cost_usd"],
                "total_tokens":    summary["total_tokens"],
                "cache_hit_ratio": summary["cache_hit_ratio"],
                "llm_stages":      summary["llm_stages"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }, on_conflict="run_id").execute()
        except Exception as exc:
            logger.debug("PipelineTracer Supabase write error: %s", exc)
