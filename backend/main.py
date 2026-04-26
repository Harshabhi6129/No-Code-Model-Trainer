"""ModelForge API — FastAPI entrypoint."""
import logging
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

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

    allowed = {".csv", ".json", ".jsonl", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type {suffix}. Allowed: {', '.join(allowed)}")

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{file_id}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)

    try:
        df = pd.read_csv(dest) if suffix == ".csv" else pd.read_json(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not parse file: {exc}") from exc

    text_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and df[c].dropna().str.len().mean() > 10
    ]
    label_cols = [
        c for c in df.columns
        if c.lower() in {"label", "target", "class", "sentiment", "category"}
    ]

    return {
        "file_id": file_id,
        "file_path": str(dest),
        "filename": file.filename,
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
    dataset_path: str | None = None
    run_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty")
        return v.strip()


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream agent responses as Server-Sent Events."""
    agents_path = Path(__file__).parent.parent / "agents"
    if str(agents_path) not in sys.path:
        sys.path.insert(0, str(agents_path))

    async def event_stream():
        try:
            from agents.pipeline import TrainingPipeline
            pipeline = TrainingPipeline()
            async for result in pipeline.run_streaming(
                user_intent=req.message,
                dataset_path=req.dataset_path,
            ):
                import json
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
            import json
            logger.error("Agent pipeline error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'agent': 'System', 'success': False, 'message': str(exc), 'output': {}})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class TrainRequest(BaseModel):
    file_id: str
    file_path: str
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

    if not Path(req.file_path).exists():
        raise HTTPException(404, f"Dataset file not found: {req.file_path}")

    job_id = str(uuid.uuid4())
    config = {
        "model_id": req.model_id,
        "task_type": req.task_type,
        "parameters": req.parameters,
        "use_cpu": req.use_cpu,
    }

    asyncio.create_task(start_training(config, req.file_path, client_id=job_id))
    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> dict[str, Any]:
    from services.socket_manager import manager
    state = manager.get_state(job_id)
    if state is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return state


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}
