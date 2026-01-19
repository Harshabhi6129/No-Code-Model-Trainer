from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import json
import uuid
from pathlib import Path
from typing import Dict, Any
import asyncio
from datetime import datetime

app = FastAPI(title="ML Training Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
uploads_dir = Path("uploads")
runs_dir = Path("runs")
uploads_dir.mkdir(exist_ok=True)
runs_dir.mkdir(exist_ok=True)

# In-memory storage for simplicity
active_connections: Dict[str, WebSocket] = {}
training_states: Dict[str, Dict] = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        active_connections.pop(session_id, None)

async def broadcast(session_id: str, data: Dict):
    if session_id in active_connections:
        try:
            await active_connections[session_id].send_text(json.dumps(data))
        except:
            active_connections.pop(session_id, None)

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze dataset"""
    file_id = str(uuid.uuid4())
    file_path = uploads_dir / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Analyze dataset
    df = pd.read_csv(file_path)
    
    # Detect task type and columns
    text_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 10]
    label_cols = [col for col in df.columns if col.lower() in ['label', 'target', 'class', 'sentiment']]
    
    stats = {
        "file_id": file_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "text_columns": text_cols,
        "label_columns": label_cols,
        "sample_data": df.head(5).to_dict('records'),
        "task_type": "classification" if label_cols else "text-generation",
        "unique_labels": df[label_cols[0]].unique().tolist() if label_cols else [],
        "avg_text_length": int(df[text_cols[0]].str.len().mean()) if text_cols else 0
    }
    
    return stats

@app.post("/models")
async def get_model_recommendations(payload: Dict[str, Any]):
    """Get AI-powered model recommendations"""
    task_type = payload.get("task_type", "classification")
    dataset_size = payload.get("dataset_size", 1000)
    
    # Simple model recommendations based on task and size
    models = {
        "classification": [
            {"id": "distilbert-base-uncased", "name": "DistilBERT", "speed": "fast", "accuracy": "good"},
            {"id": "bert-base-uncased", "name": "BERT Base", "speed": "medium", "accuracy": "excellent"},
            {"id": "roberta-base", "name": "RoBERTa", "speed": "medium", "accuracy": "excellent"}
        ],
        "text-generation": [
            {"id": "gpt2", "name": "GPT-2", "speed": "medium", "accuracy": "good"},
            {"id": "gpt2-medium", "name": "GPT-2 Medium", "speed": "slow", "accuracy": "excellent"},
            {"id": "t5-small", "name": "T5 Small", "speed": "fast", "accuracy": "good"}
        ]
    }
    
    recommended = models.get(task_type, models["classification"])
    
    # Add dynamic parameters for each model
    for model in recommended:
        if model["id"] in ["gpt2", "gpt2-medium"]:
            model["params"] = {
                "learning_rate": {"type": "float", "default": 5e-5, "min": 1e-6, "max": 1e-3},
                "batch_size": {"type": "int", "default": 8, "options": [4, 8, 16]},
                "epochs": {"type": "int", "default": 3, "min": 1, "max": 10},
                "max_length": {"type": "int", "default": 512, "min": 128, "max": 1024}
            }
        elif model["id"] == "t5-small":
            model["params"] = {
                "learning_rate": {"type": "float", "default": 3e-4, "min": 1e-6, "max": 1e-3},
                "batch_size": {"type": "int", "default": 16, "options": [8, 16, 32]},
                "epochs": {"type": "int", "default": 5, "min": 1, "max": 10},
                "max_length": {"type": "int", "default": 256, "min": 64, "max": 512}
            }
        else:  # BERT models
            model["params"] = {
                "learning_rate": {"type": "float", "default": 2e-5, "min": 1e-6, "max": 1e-3},
                "batch_size": {"type": "int", "default": 16 if dataset_size < 10000 else 32, "options": [8, 16, 32, 64]},
                "epochs": {"type": "int", "default": 3, "min": 1, "max": 10},
                "warmup_steps": {"type": "int", "default": 500, "min": 0, "max": 2000}
            }
    
    return {
        "models": recommended,
        "recommendation": recommended[0]["id"],
        "reasoning": f"DistilBERT recommended for {task_type} with {dataset_size} samples - good balance of speed and accuracy"
    }

@app.post("/train")
async def start_training(payload: Dict[str, Any]):
    """Start model training"""
    session_id = str(uuid.uuid4())
    
    # Store training config
    training_states[session_id] = {
        "status": "starting",
        "progress": 0,
        "epoch": 0,
        "loss": 0,
        "accuracy": 0,
        "config": payload,
        "start_time": datetime.now().isoformat()
    }
    
    # Start training simulation (replace with actual training)
    asyncio.create_task(simulate_training(session_id))
    
    return {"session_id": session_id, "status": "started"}

async def simulate_training(session_id: str):
    """Simulate training progress with realistic metrics"""
    import random
    import math
    
    state = training_states[session_id]
    epochs = state["config"].get("epochs", 3)
    
    for epoch in range(epochs):
        state["epoch"] = epoch + 1
        state["status"] = "training"
        
        # Simulate epoch progress
        for step in range(100):
            # Realistic loss curve (decreasing with noise)
            base_loss = 2.0 * math.exp(-epoch * 0.5 - step * 0.01)
            loss = base_loss + random.uniform(-0.1, 0.1)
            
            # Realistic accuracy curve (increasing with plateau)
            base_acc = 0.5 + 0.4 * (1 - math.exp(-epoch * 0.8 - step * 0.02))
            accuracy = base_acc + random.uniform(-0.02, 0.02)
            
            state.update({
                "progress": (epoch * 100 + step + 1) / (epochs * 100) * 100,
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "step": step + 1
            })
            
            await broadcast(session_id, state.copy())
            await asyncio.sleep(0.1)  # 100ms updates
    
    # Training complete
    state["status"] = "completed"
    state["progress"] = 100
    
    # Save final model info
    model_path = runs_dir / f"{session_id}_model.bin"
    model_path.touch()  # Create dummy model file
    
    state["model_path"] = str(model_path)
    await broadcast(session_id, state.copy())

@app.get("/status/{session_id}")
async def get_training_status(session_id: str):
    """Get current training status"""
    return training_states.get(session_id, {"error": "Session not found"})

@app.get("/export/{session_id}")
async def export_model(session_id: str):
    """Export trained model"""
    if session_id not in training_states:
        return {"error": "Session not found"}
    
    state = training_states[session_id]
    if state["status"] != "completed":
        return {"error": "Training not completed"}
    
    # Create export package
    export_data = {
        "model_id": session_id,
        "config": state["config"],
        "final_metrics": {
            "loss": state["loss"],
            "accuracy": state["accuracy"],
            "epochs": state["epoch"]
        },
        "training_time": state["start_time"],
        "export_time": datetime.now().isoformat()
    }
    
    return export_data

@app.post("/models")
async def get_models(request: dict):
    task_type = request.get("task_type")
    dataset_size = request.get("dataset_size")
    
    if task_type == "classification":
        models = [
            {
                "id": "bert-base",
                "name": "BERT Base",
                "speed": "medium",
                "accuracy": "good",
                "params": {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "default": 2e-5},
                    "epochs": {"type": "int", "min": 1, "max": 10, "default": 3},
                    "batch_size": {"type": "int", "options": [8, 16, 32], "default": 16}
                }
            },
            {
                "id": "roberta-base",
                "name": "RoBERTa Base",
                "speed": "medium",
                "accuracy": "excellent",
                "params": {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "default": 1e-5},
                    "epochs": {"type": "int", "min": 1, "max": 10, "default": 4},
                    "batch_size": {"type": "int", "options": [8, 16, 32], "default": 16}
                }
            }
        ]
    elif task_type == "text-generation":
        models = [
            {
                "id": "gpt2",
                "name": "GPT-2",
                "speed": "fast",
                "accuracy": "good",
                "params": {
                    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "default": 5e-5},
                    "epochs": {"type": "int", "min": 1, "max": 10, "default": 3},
                    "batch_size": {"type": "int", "options": [4, 8, 16], "default": 8}
                }
            }
        ]
    else:
        models = []
    
    return {"models": models}

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "ML Training Platform API"}
