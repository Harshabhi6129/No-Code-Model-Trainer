"""
Persist training events to the Supabase run_events table.
Uses the service role key to bypass RLS — backend writes are trusted.
Fails silently if SUPABASE_SERVICE_ROLE_KEY is not configured so that
missing credentials never crash the training pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client = None
_client_init_attempted = False


def _get_client():
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client

    _client_init_attempted = True
    url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        logger.debug("run_event_writer: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set — events will not be persisted")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("run_event_writer: Supabase client initialized")
    except ImportError:
        logger.warning("run_event_writer: supabase package not installed — pip install supabase")
    except Exception as exc:
        logger.warning("run_event_writer: failed to init Supabase client: %s", exc)

    return _client


async def write_run_event(
    run_id: str,
    event_type: str,
    data: dict[str, Any],
) -> bool:
    """
    Insert one row into run_events. event_type must be one of:
    'agent' | 'progress' | 'metric' | 'log' | 'error' | 'done'

    Returns True on success, False on any failure (never raises).
    """
    client = _get_client()
    if client is None:
        return False

    valid_types = {"agent", "progress", "metric", "log", "error", "done"}
    if event_type not in valid_types:
        logger.warning("write_run_event: invalid event_type %r — skipping", event_type)
        return False

    # Sanitize: ensure data is JSON-serialisable (drop any non-serialisable values)
    safe_data: dict[str, Any] = {}
    for k, v in data.items():
        try:
            import json
            json.dumps(v)
            safe_data[k] = v
        except (TypeError, ValueError):
            safe_data[k] = str(v)

    try:
        await asyncio.to_thread(
            lambda: client.table("run_events").insert({
                "run_id": run_id,
                "event_type": event_type,
                "data": safe_data,
            }).execute()
        )
        return True
    except Exception as exc:
        logger.warning("write_run_event failed for run %s: %s", run_id, exc)
        return False


async def write_agent_event(run_id: str, agent_name: str, success: bool, message: str, output: dict) -> bool:
    """Convenience wrapper for agent pipeline events."""
    return await write_run_event(run_id, "agent", {
        "agent": agent_name,
        "success": success,
        "message": message,
        "output": {k: v for k, v in output.items() if k != "final"},
    })


async def write_metric_event(run_id: str, epoch: int, metrics: dict[str, Any]) -> bool:
    """Convenience wrapper for per-epoch training metric events."""
    return await write_run_event(run_id, "metric", {
        "epoch": epoch,
        **metrics,
    })


async def write_pipeline_checkpoint(
    run_id: str,
    completed_stages: list[str],
    checkpoint_data: dict[str, Any],
) -> bool:
    """
    Persist an AgentContext snapshot to the runs table after each stage.
    Uses a targeted PATCH so we only update the two checkpoint columns —
    other run fields (status, metrics, etc.) are untouched.

    Fails silently if Supabase is not configured — the pipeline continues.
    """
    client = _get_client()
    if client is None:
        return False

    # Sanitize: only keep JSON-serialisable values
    import json
    safe: dict[str, Any] = {}
    for k, v in checkpoint_data.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)

    try:
        await asyncio.to_thread(
            lambda: client.table("runs").update({
                "completed_stages":    completed_stages,
                "pipeline_checkpoint": safe,
            }).eq("id", run_id).execute()
        )
        return True
    except Exception as exc:
        logger.warning("write_pipeline_checkpoint failed for run %s: %s", run_id, exc)
        return False


async def load_pipeline_checkpoint(run_id: str) -> dict[str, Any] | None:
    """
    Load a previously saved AgentContext snapshot from the runs table.
    Returns None if the run doesn't exist, has no checkpoint, or Supabase
    is unavailable — callers treat None as "start fresh".
    """
    client = _get_client()
    if client is None:
        return None

    try:
        resp = await asyncio.to_thread(
            lambda: client.table("runs")
            .select("pipeline_checkpoint,completed_stages")
            .eq("id", run_id)
            .single()
            .execute()
        )
        row = resp.data
        if not row:
            return None
        checkpoint = row.get("pipeline_checkpoint") or {}
        stages     = row.get("completed_stages") or []
        if not checkpoint and not stages:
            return None
        checkpoint["completed_stages"] = stages
        return checkpoint
    except Exception as exc:
        logger.warning("load_pipeline_checkpoint failed for run %s: %s", run_id, exc)
        return None
