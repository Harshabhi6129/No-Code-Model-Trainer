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

app = FastAPI(title="ModelForge API", version="0.2.0")

# Read allowed origins from env; fall back to localhost for local dev
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    # Wildcard subdomains need allow_origin_regex, not allow_origins
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve uploads dir relative to this file so it's stable regardless of cwd
UPLOADS_DIR = Path(os.getenv("UPLOAD_DIR", str(Path(__file__).parent / "uploads")))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory file registry: file_id → absolute Path.
# Stage 3: replace with a DB lookup so registrations survive restarts.
_FILE_REGISTRY: dict[str, Path] = {}

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
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Sanitize: strip any path separators the client might inject
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(400, "Invalid filename")

    allowed = {".csv", ".json", ".jsonl", ".txt"}
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type {suffix}. Allowed: {', '.join(allowed)}")

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{file_id}{suffix}"  # no user-supplied name in path
    content = await file.read()
    dest.write_bytes(content)
    _FILE_REGISTRY[file_id] = dest

    try:
        df = pd.read_csv(dest) if suffix == ".csv" else pd.read_json(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
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

    # Never return the server-side path — clients get only the opaque file_id
    return {
        "file_id": file_id,
        "filename": safe_name,
        "rows": len(df),
        "columns": list(df.columns),
        "text_columns": text_cols,
        "label_columns": label_cols,
        "sample": df.head(3).to_dict("records"),
        "unique_labels": (
            df[label_cols[0]].dropna().unique().tolist() if label_cols else []
        ),
    }


class ChatRequest(BaseModel):
    message: str
    file_id: str | None = None  # opaque id — backend resolves to path
    run_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


def _resolve_file_id(file_id: str | None) -> str | None:
    """Return the absolute path for a file_id, or None if not provided."""
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
            pipeline = TrainingPipeline()
            async for result in pipeline.run_streaming(
                user_intent=req.message,
                dataset_path=dataset_path,
            ):
                data = json.dumps({
                    "agent": result.agent_name,
                    "success": result.success,
                    "message": result.message,
                    "output": result.output,
                })
                yield f"data: {data}\n\n"
                if not result.success:
                    break
        except Exception as exc:
            logger.error("Agent pipeline error: %s", exc, exc_info=True)
            # Return a generic message — full details stay in server logs
            yield f"data: {json.dumps({'agent': 'System', 'success': False, 'message': 'An internal error occurred. Please try again.', 'output': {}})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class TrainRequest(BaseModel):
    file_id: str  # opaque id only — no client-supplied paths
    model_id: str = "distilbert-base-uncased"
    task_type: str = "text_classification"
    parameters: dict[str, Any] = {}
    use_cpu: bool = False

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id cannot be empty")
        return v.strip()


@app.post("/train")
async def start_training_job(req: TrainRequest) -> dict[str, str]:
    import asyncio
    from services.trainer import start_training

    file_path = _FILE_REGISTRY.get(req.file_id)
    if file_path is None:
        raise HTTPException(404, "Dataset not found. Please re-upload your file.")

    # Confirm the resolved path is still within uploads dir (defence-in-depth)
    if not str(file_path).startswith(str(UPLOADS_DIR)):
        raise HTTPException(400, "Invalid dataset reference.")

    job_id = str(uuid.uuid4())
    config = {
        "model_id": req.model_id,
        "task_type": req.task_type,
        "parameters": req.parameters,
        "use_cpu": req.use_cpu,
    }

    asyncio.create_task(start_training(config, str(file_path), client_id=job_id))
    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> dict[str, Any]:
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


@app.post("/infer")
async def run_inference(req: InferRequest) -> dict[str, Any]:
    """Run classification inference on a trained model."""
    import asyncio
    from services.inference_cache import cache

    if len(req.text) > 2000:
        raise HTTPException(
            400,
            f"Text is too long ({len(req.text)} chars). Maximum 2000 characters. "
            "Very long text is automatically truncated to 512 tokens by the model."
        )

    if not req.artifact_path:
        raise HTTPException(422, "No trained model artifact found for this run.")

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
