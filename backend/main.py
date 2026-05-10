"""ModelForge API — FastAPI entrypoint."""
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

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
    """Stream agent responses as Server-Sent Events."""
    dataset_path = _resolve_file_id(req.file_id)

    agents_path = Path(__file__).parent.parent / "agents"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))

    async def event_stream():
        try:
            from agents.pipeline import TrainingPipeline
            from services.run_event_writer import write_agent_event

            pipeline = TrainingPipeline()
            async for result in pipeline.run_streaming(
                user_intent=req.message,
                dataset_path=dataset_path,
                hyperparameter_overrides=req.hyperparameter_overrides,
                hf_token=req.hf_token,
            ):
                data = json.dumps({
                    "agent": result.agent_name,
                    "success": result.success,
                    "message": result.message,
                    "output": result.output,
                })
                yield f"data: {data}\n\n"

                # Persist agent events to DB when run_id is provided
                if req.run_id and result.output.get("final", True):
                    await write_agent_event(
                        run_id=req.run_id,
                        agent_name=result.agent_name,
                        success=result.success,
                        message=result.message,
                        output=result.output,
                    )

                if not result.success:
                    break
        except Exception as exc:
            logger.error("Agent pipeline error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'agent': 'System', 'success': False, 'message': 'An internal error occurred. Please try again.', 'output': {}})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@app.post("/train/{run_id}/cancel")
async def cancel_training(run_id: str) -> dict[str, str]:
    """Signal an active training run to stop at the next step boundary."""
    agents_path = Path(__file__).parent.parent / "agents"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))

    from agents.train_agent import cancel_run

    found = cancel_run(run_id)
    if not found:
        raise HTTPException(404, f"No active training run found for run_id={run_id}")
    return {"status": "cancelling", "run_id": run_id}


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
    agents_path = Path(__file__).parent.parent / "agents"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
