"""ModelForge API — FastAPI entrypoint."""
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, field_validator
from typing import Literal

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ModelForge API", version="0.3.0")

_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path(os.getenv("UPLOAD_DIR", str(Path(__file__).parent / "uploads")))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Known runs directory — artifact_path must stay within this tree
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(Path(__file__).parent.parent / "agents" / "runs")))

# In-memory file registry: file_id → absolute Path.
# Rebuilt from .meta.json sidecars on startup so it survives server restarts.
_FILE_REGISTRY: dict[str, Path] = {}

# Pending clarifications: run_id → original ChatRequest params.
# Populated when IntentAgent returns confidence < 0.7 and asks a clarifying question.
# Consumed (and removed) by POST /clarify/{run_id}.
_pending_clarifications: dict[str, dict] = {}


def _rebuild_registry_from_disk() -> None:
    """Scan UPLOADS_DIR for .meta.json sidecars and restore the in-memory registry."""
    for meta_file in UPLOADS_DIR.glob("*.meta.json"):
        try:
            import json as _json
            meta = _json.loads(meta_file.read_text())
            file_id = meta.get("file_id")
            data_path = Path(meta.get("data_path", ""))
            if file_id and data_path.exists():
                _FILE_REGISTRY[file_id] = data_path
        except Exception as exc:
            logger.warning("Could not restore registry entry from %s: %s", meta_file, exc)

    if _FILE_REGISTRY:
        logger.info("Restored %d file(s) to registry from disk", len(_FILE_REGISTRY))


_rebuild_registry_from_disk()

from services.socket_manager import manager as socket_manager


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await socket_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        socket_manager.disconnect(job_id, websocket)
    except Exception as exc:
        logger.warning("WebSocket error for job %s: %s", job_id, exc)
        socket_manager.disconnect(job_id, websocket)


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a dataset file. Returns metadata including sample rows, class
    distribution, and data quality warnings — all without hitting the GPU.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(400, "Invalid filename")

    allowed = {".csv", ".json", ".jsonl", ".txt"}
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type {suffix}. Allowed: {', '.join(allowed)}")

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{file_id}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    _FILE_REGISTRY[file_id] = dest

    # Write sidecar so the registry survives server restarts
    meta_file = UPLOADS_DIR / f"{file_id}.meta.json"
    meta_file.write_text(json.dumps({
        "file_id": file_id,
        "original_name": safe_name,
        "data_path": str(dest),
    }))

    try:
        df = pd.read_csv(dest) if suffix == ".csv" else pd.read_json(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
        _FILE_REGISTRY.pop(file_id, None)
        raise HTTPException(422, f"Could not parse file: {exc}") from exc

    text_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and df[c].dropna().str.len().mean() > 10
    ]
    label_cols = [
        c for c in df.columns
        if c.lower() in {"label", "target", "class", "sentiment", "category"}
    ]

    # Class distribution for the first detected label column
    class_distribution: dict[str, int] = {}
    if label_cols:
        class_distribution = df[label_cols[0]].astype(str).value_counts().to_dict()

    # Text length stats for the first detected text column
    text_length_stats: dict[str, float] = {}
    if text_cols:
        lengths = df[text_cols[0]].dropna().astype(str).str.len()
        text_length_stats = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "p50": round(float(lengths.quantile(0.50)), 1),
            "p90": round(float(lengths.quantile(0.90)), 1),
            "p99": round(float(lengths.quantile(0.99)), 1),
        }

    # Quick data quality warnings (pure Python, no GPU)
    data_warnings: list[str] = []
    total = len(df)
    if total < 50:
        data_warnings.append(f"Very small dataset ({total} rows) — results may be unreliable.")
    elif total < 200:
        data_warnings.append(f"Small dataset ({total} rows) — consider data augmentation.")

    if label_cols and class_distribution:
        counts = list(class_distribution.values())
        if len(counts) > 1:
            imbalance = max(counts) / max(min(counts), 1)
            if imbalance > 10:
                data_warnings.append(f"Severe class imbalance ({imbalance:.0f}:1 ratio) — weighted loss will be applied automatically.")
            elif imbalance > 3:
                data_warnings.append(f"Moderate class imbalance ({imbalance:.1f}:1 ratio) — consider oversampling.")

    if text_cols and text_length_stats:
        if text_length_stats.get("p90", 0) > 400:
            data_warnings.append("Many texts exceed 400 chars — they will be truncated to model max_length.")
        if text_length_stats.get("mean", 100) < 10:
            data_warnings.append("Average text is very short — there may be insufficient signal for training.")

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        data_warnings.append(f"{dup_count} duplicate row(s) detected — CleanAgent will remove them.")

    null_count = int(df[text_cols[0]].isna().sum()) if text_cols else 0
    if null_count > 0:
        data_warnings.append(f"{null_count} row(s) with empty text detected — will be removed before training.")

    # Build histogram of text lengths (10 bins)
    text_length_histogram: list[dict] = []
    if text_cols and len(df) > 0:
        lengths_series = df[text_cols[0]].dropna().astype(str).str.len()
        try:
            hist, bin_edges = _compute_histogram(lengths_series.tolist(), bins=10)
            text_length_histogram = [
                {"bin_start": int(bin_edges[i]), "bin_end": int(bin_edges[i + 1]), "count": int(hist[i])}
                for i in range(len(hist))
            ]
        except Exception:
            pass

    return {
        "file_id": file_id,
        "filename": safe_name,
        "rows": len(df),
        "columns": list(df.columns),
        "text_columns": text_cols,
        "label_columns": label_cols,
        "sample_rows": df.sample(min(5, len(df)), random_state=42).to_dict("records"),
        "unique_labels": (
            df[label_cols[0]].dropna().unique().tolist() if label_cols else []
        ),
        "class_distribution": class_distribution,
        "text_length_stats": text_length_stats,
        "text_length_histogram": text_length_histogram,
        "data_warnings": data_warnings,
        "duplicate_count": dup_count,
        "null_count": null_count,
    }


def _validate_file_id(file_id: str) -> str:
    """Security: ensure file_id is a valid UUID before touching the filesystem."""
    try:
        uuid.UUID(file_id)
    except ValueError:
        raise HTTPException(400, "Invalid file_id format.")
    return file_id


class RenameDatasetRequest(BaseModel):
    filename: str

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("filename cannot be empty")
        if "/" in v or "\\" in v or ".." in v:
            raise ValueError("filename contains invalid characters")
        return v


@app.delete("/datasets/{file_id}")
async def delete_dataset(file_id: str) -> dict[str, str]:
    """Remove an uploaded dataset and its registry entry."""
    _validate_file_id(file_id)

    if file_id not in _FILE_REGISTRY:
        raise HTTPException(404, f"Dataset {file_id} not found.")

    data_path = _FILE_REGISTRY.pop(file_id)
    data_path.unlink(missing_ok=True)

    meta_file = UPLOADS_DIR / f"{file_id}.meta.json"
    meta_file.unlink(missing_ok=True)

    return {"status": "deleted", "file_id": file_id}


@app.patch("/datasets/{file_id}")
async def rename_dataset(file_id: str, req: RenameDatasetRequest) -> dict[str, str]:
    """Rename a dataset (updates the stored display name, not the physical file)."""
    _validate_file_id(file_id)

    if file_id not in _FILE_REGISTRY:
        raise HTTPException(404, f"Dataset {file_id} not found.")

    meta_file = UPLOADS_DIR / f"{file_id}.meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            meta["original_name"] = req.filename
            meta_file.write_text(json.dumps(meta))
        except Exception as exc:
            logger.warning("Could not update meta for %s: %s", file_id, exc)

    return {"status": "renamed", "file_id": file_id, "filename": req.filename}


def _compute_histogram(values: list, bins: int = 10) -> tuple[list[int], list[float]]:
    """Pure-Python histogram computation (avoids numpy import at module level)."""
    if not values:
        return [], []
    mn, mx = min(values), max(values)
    if mn == mx:
        return [len(values)], [float(mn), float(mx) + 1]
    step = (mx - mn) / bins
    edges = [mn + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / step), bins - 1)
        counts[idx] += 1
    return counts, edges


class ChatRequest(BaseModel):
    message: str
    file_id: str | None = None
    run_id: str | None = None
    hyperparameter_overrides: dict[str, Any] = {}
    hf_token: str | None = None
    # A3: set to a previous run_id to resume from its last checkpoint
    resume_from_run_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


def _resolve_file_id(file_id: str | None) -> str | None:
    if not file_id:
        return None
    path = _FILE_REGISTRY.get(file_id)
    if path is None:
        raise HTTPException(404, "Dataset not found. Please re-upload your file.")
    return str(path)


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream agent responses as Server-Sent Events.

    Pass resume_from_run_id to continue a pipeline that failed mid-way —
    completed stages (Intent, Data, Clean, Model) are restored from the
    checkpoint and skipped, so only the remaining stages execute.
    """
    dataset_path = _resolve_file_id(req.file_id)
    _agents_import()

    async def event_stream():
        try:
            from agents.pipeline import TrainingPipeline
            from services.run_event_writer import (
                write_agent_event,
                write_pipeline_checkpoint,
                load_pipeline_checkpoint,
            )

            # A3: load checkpoint when resuming
            initial_context: dict | None = None
            if req.resume_from_run_id:
                initial_context = await load_pipeline_checkpoint(req.resume_from_run_id)
                if initial_context:
                    logger.info(
                        "Resuming run %s from checkpoint (completed: %s)",
                        req.resume_from_run_id,
                        initial_context.get("completed_stages", []),
                    )

            pipeline = TrainingPipeline()
            async for result, context in pipeline.run_streaming(
                user_intent=req.message,
                dataset_path=dataset_path,
                hyperparameter_overrides=req.hyperparameter_overrides,
                hf_token=req.hf_token,
                run_id=req.run_id,
                initial_context=initial_context,
            ):
                data = json.dumps({
                    "agent":   result.agent_name,
                    "success": result.success,
                    "message": result.message,
                    "output":  result.output,
                    "metadata": result.metadata,
                })
                yield f"data: {data}\n\n"

                # ── HITL: pause pipeline when IntentAgent needs clarification ──
                # IntentAgent returns next_agent=None with clarification_needed set
                # when confidence < 0.7. Store params so /clarify/{run_id} can resume.
                if (
                    result.agent_name == "Intent"
                    and result.success
                    and result.output.get("clarification_needed")
                    and req.run_id
                ):
                    _pending_clarifications[req.run_id] = {
                        "message":                  req.message,
                        "file_id":                  req.file_id,
                        "hyperparameter_overrides": req.hyperparameter_overrides,
                        "hf_token":                 req.hf_token,
                        "clarification_question":   result.output["clarification_needed"],
                    }
                    logger.info(
                        "[%s] Clarification needed — run paused awaiting user response",
                        req.run_id,
                    )

                if req.run_id:
                    # Always persist the agent event
                    await write_agent_event(
                        run_id=req.run_id,
                        agent_name=result.agent_name,
                        success=result.success,
                        message=result.message,
                        output=result.output,
                    )
                    # A3: after each successful stage, checkpoint the full context
                    # so a resume can skip already-completed work
                    if result.success:
                        await write_pipeline_checkpoint(
                            run_id=req.run_id,
                            completed_stages=list(context.completed_stages),
                            checkpoint_data=pipeline.context_snapshot(context),
                        )

                if not result.success:
                    break

        except Exception as exc:
            logger.error("Agent pipeline error: %s", exc, exc_info=True)
            err_payload = json.dumps({
                "agent": "System", "success": False,
                "message": "An internal error occurred. Please try again.", "output": {},
            })
            yield f"data: {err_payload}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ClarifyRequest(BaseModel):
    user_response: str

    @field_validator("user_response")
    @classmethod
    def response_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_response cannot be empty")
        return v.strip()


@app.post("/clarify/{run_id}")
async def clarify_intent(run_id: str, req: ClarifyRequest) -> StreamingResponse:
    """
    Resume a pipeline that paused waiting for user clarification.

    The original intent is amended with the user's response and the pipeline
    restarts from IntentAgent with the enriched intent. Returns SSE.
    """
    params = _pending_clarifications.pop(run_id, None)
    if params is None:
        raise HTTPException(
            404,
            f"No pending clarification found for run_id={run_id}. "
            "The session may have timed out or already been resumed."
        )

    original_intent   = params["message"]
    amended_intent    = (
        f"{original_intent}\n\n"
        f"User clarification: {req.user_response}"
    )
    file_id           = params.get("file_id")
    overrides         = params.get("hyperparameter_overrides", {})
    hf_token          = params.get("hf_token")
    dataset_path      = _resolve_file_id(file_id)
    _agents_import()

    async def clarify_stream():
        try:
            from agents.pipeline import TrainingPipeline
            from services.run_event_writer import write_agent_event, write_pipeline_checkpoint

            pipeline = TrainingPipeline()
            async for result, context in pipeline.run_streaming(
                user_intent=amended_intent,
                dataset_path=dataset_path,
                hyperparameter_overrides=overrides,
                hf_token=hf_token,
                run_id=run_id,
            ):
                data = json.dumps({
                    "agent":   result.agent_name,
                    "success": result.success,
                    "message": result.message,
                    "output":  result.output,
                    "metadata": result.metadata,
                })
                yield f"data: {data}\n\n"

                await write_agent_event(
                    run_id=run_id,
                    agent_name=result.agent_name,
                    success=result.success,
                    message=result.message,
                    output=result.output,
                )
                if result.success:
                    await write_pipeline_checkpoint(
                        run_id=run_id,
                        completed_stages=list(context.completed_stages),
                        checkpoint_data=pipeline.context_snapshot(context),
                    )
                if not result.success:
                    break

        except Exception as exc:
            logger.error("Clarify pipeline error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'agent': 'System', 'success': False, 'message': str(exc), 'output': {}})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(clarify_stream(), media_type="text/event-stream")


@app.post("/train")
async def start_training_job_deprecated() -> None:
    """
    Deprecated legacy training endpoint.
    Use POST /chat with the agent pipeline instead.
    """
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use POST /chat with the agent pipeline for training.",
    )


def _agents_import():
    agents_path = Path(__file__).parent.parent / "agents"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))


@app.post("/train/{run_id}/cancel")
async def cancel_training(run_id: str) -> dict[str, str]:
    """Signal an active training run to stop at the next step boundary."""
    _agents_import()
    from agents.train_agent import cancel_run

    found = cancel_run(run_id)
    if not found:
        raise HTTPException(404, f"No active training run found for run_id={run_id}")
    return {"status": "cancelling", "run_id": run_id}


@app.post("/train/{run_id}/pause")
async def pause_training(run_id: str) -> dict[str, str]:
    """Block training at the next step boundary."""
    _agents_import()
    from agents.train_agent import pause_run, is_paused

    if is_paused(run_id):
        raise HTTPException(409, "Training is already paused.")
    found = pause_run(run_id)
    if not found:
        raise HTTPException(404, f"No active training run found for run_id={run_id}")
    return {"status": "paused", "run_id": run_id}


@app.post("/train/{run_id}/resume")
async def resume_training(run_id: str) -> dict[str, str]:
    """Unblock a paused training run."""
    _agents_import()
    from agents.train_agent import resume_run, is_paused

    if not is_paused(run_id):
        raise HTTPException(409, "Training is not paused.")
    found = resume_run(run_id)
    if not found:
        raise HTTPException(404, f"No active training run found for run_id={run_id}")
    return {"status": "resumed", "run_id": run_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> dict[str, Any]:
    """Get the last known status for a WebSocket training job."""
    from services.socket_manager import manager
    state = manager.get_state(job_id)
    if state is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return state


class InferRequest(BaseModel):
    run_id: str
    text: str
    artifact_path: str
    label_names: list[str] = []

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()

    @field_validator("run_id")
    @classmethod
    def run_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("run_id cannot be empty")
        return v.strip()

    @field_validator("artifact_path")
    @classmethod
    def artifact_path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("artifact_path cannot be empty")
        return v.strip()


def _validate_artifact_path(artifact_path: str, run_id: str) -> Path:
    """
    Security check: artifact_path must resolve to a path within RUNS_DIR.
    Prevents path traversal attacks where a user supplies an arbitrary filesystem path.
    """
    resolved = Path(artifact_path).resolve()
    runs_resolved = RUNS_DIR.resolve()

    # Also allow paths relative to the agents directory (agent pipeline output)
    agents_runs = (Path(__file__).parent.parent / "agents" / "runs").resolve()

    if not (str(resolved).startswith(str(runs_resolved)) or
            str(resolved).startswith(str(agents_runs))):
        logger.warning(
            "Rejected artifact_path outside RUNS_DIR: run_id=%s path=%s",
            run_id, artifact_path,
        )
        raise HTTPException(422, "Invalid model artifact path.")

    return resolved


@app.post("/infer")
async def run_inference(req: InferRequest) -> dict[str, Any]:
    """Run classification inference on a trained model."""
    import asyncio
    from services.inference_cache import cache

    if len(req.text) > 2000:
        raise HTTPException(
            400,
            f"Text is too long ({len(req.text)} chars). Maximum 2000 characters.",
        )

    # Security: validate artifact_path is within the known runs directory
    _validate_artifact_path(req.artifact_path, req.run_id)

    try:
        result = await cache.predict(
            run_id=req.run_id,
            text=req.text,
            artifact_path=req.artifact_path,
            label_names=req.label_names,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Inference runtime error for run %s: %s", req.run_id, exc, exc_info=True)
        raise HTTPException(500, "Model inference failed. Please try again.") from exc
    except Exception as exc:
        logger.error("Inference error for run %s: %s", req.run_id, exc, exc_info=True)
        raise HTTPException(500, "An unexpected error occurred during inference.") from exc


class ExportRequest(BaseModel):
    run_id: str
    artifact_path: str
    format: Literal["onnx", "torchscript"] = "onnx"
    opset_version: int = 14
    optimize: bool = True

    @field_validator("run_id", "artifact_path")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v.strip()


@app.post("/export")
async def export_model_endpoint(req: ExportRequest) -> FileResponse:
    """Convert a trained model to ONNX or TorchScript and return as a download."""
    import asyncio
    import shutil

    validated_path = _validate_artifact_path(req.artifact_path, req.run_id)

    from services.model_exporter import export_model

    try:
        out_file: Path = await asyncio.to_thread(
            export_model,
            artifact_path=str(validated_path),
            export_format=req.format,
            opset_version=req.opset_version,
            optimize=req.optimize,
        )
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Export failed for run %s: %s", req.run_id, exc, exc_info=True)
        raise HTTPException(500, f"Export failed: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected export error for run %s: %s", req.run_id, exc, exc_info=True)
        raise HTTPException(500, "An unexpected error occurred during export.") from exc

    suffix   = ".onnx" if req.format == "onnx" else ".pt"
    filename = f"model_{req.run_id[:8]}{suffix}"
    tmp_parent = out_file.parent

    def cleanup() -> None:
        shutil.rmtree(tmp_parent, ignore_errors=True)

    return FileResponse(
        path=str(out_file),
        media_type="application/octet-stream",
        filename=filename,
        background=BackgroundTask(cleanup),
    )


@app.get("/models")
async def get_models(
    task_type: str | None = None,
    category: str | None = None,
    max_params_m: int | None = None,
    provider: str | None = None,
    lora_only: bool = False,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """Return filtered model catalog."""
    _agents_import()
    from agents.model_catalog import filter_catalog

    results = filter_catalog(
        task_type=task_type,
        category=category,
        max_params_m=max_params_m,
        provider=provider,
        lora_only=lora_only,
    )

    if q:
        ql = q.lower()
        results = [
            m for m in results
            if ql in m["display_name"].lower()
            or ql in m["description"].lower()
            or ql in m.get("best_for", "").lower()
            or ql in m["model_id"].lower()
            or any(ql in tag for tag in m.get("tags", []))
        ]

    return results


@app.get("/runs/{run_id}/script")
async def download_training_script(run_id: str) -> StreamingResponse:
    """
    Generate and download a standalone Python training script for a completed run.

    Loads the pipeline checkpoint from Supabase (task_spec + model_recipe +
    training_result), generates a copy-paste-runnable script via code_generator,
    and returns it as a .py file download.
    """
    from services.run_event_writer import load_pipeline_checkpoint

    checkpoint = await load_pipeline_checkpoint(run_id)
    if not checkpoint:
        raise HTTPException(
            404,
            f"No pipeline checkpoint found for run_id={run_id}. "
            "The run may not have completed or Supabase is not configured."
        )

    task_spec       = checkpoint.get("task_spec") or {}
    data_profile    = checkpoint.get("data_profile") or {}
    model_recipe    = checkpoint.get("model_recipe") or {}
    training_result = checkpoint.get("training_result") or {}

    if not model_recipe or not training_result:
        raise HTTPException(
            422,
            "Run did not complete training — no model recipe or training result available."
        )

    _agents_import()
    from agents.services.code_generator import generate_training_script, CodeGenerationError

    try:
        script = generate_training_script(
            task_spec=task_spec,
            data_profile=data_profile,
            model_recipe=model_recipe,
            training_result=training_result,
        )
    except CodeGenerationError as exc:
        logger.error("Script generation failed for run %s: %s", run_id, exc)
        raise HTTPException(500, f"Could not generate training script: {exc}") from exc

    filename = f"train_{run_id[:8]}.py"
    return StreamingResponse(
        iter([script]),
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
