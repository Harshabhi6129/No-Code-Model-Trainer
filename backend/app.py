# app.py
import os
import shutil
import uuid
import time
import json
import itertools
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from dataset_validator import validate_csv
from model_params import get_model_candidates, get_model_params
from hparam_suggester import suggest_hparams
from training_runner import start_training, get_training_state, save_checkpoint
from exporter import export_run
from ws_broker import ws_router, publish
from resource_monitor import start_resource_monitoring, stop_resource_monitoring
from llm_service import analyze_training_progress, generate_training_insights
from report_generator import generate_training_report, generate_model_package
from hyperopt_runner import start_hyperparameter_optimization, SEARCH_SPACES

# --- FastAPI App Setup ---
app = FastAPI(title="No-Code HuggingFace Fine-Tuning Platform")
from fastapi import WebSocket
from ws_broker import register, unregister

@app.websocket("/ws/{run_id}")
async def ws_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    await register(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except Exception:
        pass
    finally:
        await unregister(run_id, websocket)

# Enable CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add WebSocket router
app.include_router(ws_router)

# Ensure required folders exist
Path("uploads").mkdir(exist_ok=True)
Path("runs").mkdir(exist_ok=True)
Path("exports").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ----------------------
# 1️⃣ Dataset Validation
# ----------------------
@app.post("/validate-dataset")
async def validate_dataset(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = Path("uploads") / f"{file_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = validate_csv(file_path)
    return JSONResponse(result)

# ----------------------
# 2️⃣ Model Candidates
# ----------------------
@app.get("/model-candidates")
async def model_candidates(task: str):
    """
    Suggests best models for a given NLP task.
    Returns:
        {
            "transformers": [...],
            "finetuned": [...],
            "advice": "string"
        }
    """
    return JSONResponse(get_model_candidates(task))

# ----------------------
# 3️⃣ Model Hyperparameters
# ----------------------
@app.get("/model-params")
async def model_params(model_id: str):
    """
    Returns a JSON schema for the hyperparameter form.
    """
    return JSONResponse(get_model_params(model_id))

# ----------------------
# 4️⃣ Suggest Hyperparameters
# ----------------------
@app.post("/suggest-hparams")
async def suggest_params(payload: dict):
    """
    Suggest hyperparameters based on dataset & model context.
    """
    model_id = payload.get("model_id", "bert-base-uncased")
    stats = payload.get("stats", {"row_count": 1000, "num_labels": 2, "avg_length": 64})
    return JSONResponse(suggest_hparams(model_id, stats))

@app.post("/suggest-task-model")
async def suggest_task_model(payload: dict):
    """
    Suggest task type and models based on use case description.
    """
    use_case = payload.get("use_case", "")
    
    # Simple task classification
    task_type = "classification"
    if any(word in use_case.lower() for word in ["sentiment", "classify", "category"]):
        task_type = "classification"
    elif any(word in use_case.lower() for word in ["summarize", "summary"]):
        task_type = "summarization"
    elif any(word in use_case.lower() for word in ["translate", "translation"]):
        task_type = "translation"
    
    return JSONResponse({
        "task_type": task_type,
        "confidence": 0.9,
        "suggested_models": get_model_candidates(task_type),
        "reasoning": f"Detected {task_type} task based on keywords in description"
    })

# ----------------------
# 5️⃣ Start Training
# ----------------------
@app.post("/train")
async def train_model(payload: dict):
    """
    Starts training in a background thread.
    Returns run_id.
    """
    run_id = str(uuid.uuid4())
    start_training(run_id, payload)
    
    # Start resource monitoring
    start_resource_monitoring(run_id)
    
    return {"status": "started", "run_id": run_id}

# ----------------------
# 6️⃣ Training Control
# ----------------------
@app.post("/api/training/{run_id}/pause")
async def pause_training(run_id: str):
    """Pause training gracefully"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    if state.status != "running":
        return JSONResponse({"error": f"Cannot pause training in {state.status} state"}, status_code=400)
    
    state.pause()
    return {"status": "paused", "epoch": state.current_epoch, "batch": state.current_batch}

@app.post("/api/training/{run_id}/resume")
async def resume_training(run_id: str):
    """Resume paused training"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    if state.status != "paused":
        return JSONResponse({"error": f"Cannot resume training in {state.status} state"}, status_code=400)
    
    state.resume()
    return {"status": "resumed", "epoch": state.current_epoch, "batch": state.current_batch}

@app.post("/api/training/{run_id}/stop")
async def stop_training(run_id: str):
    """Stop training permanently"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    if state.status in ["stopped", "completed", "failed"]:
        return JSONResponse({"error": f"Training already in {state.status} state"}, status_code=400)
    
    state.stop()
    
    # Stop resource monitoring
    stop_resource_monitoring(run_id)
    
    return {"status": "stopped"}

@app.post("/api/training/{run_id}/update-params")
async def update_training_params(
    run_id: str,
    learning_rate: float = None,
    weight_decay: float = None,
    dropout: float = None,
    gradient_clip: float = None
):
    """Update training parameters during training"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    if state.status != "running":
        return JSONResponse({"error": f"Cannot update params in {state.status} state"}, status_code=400)
    
    updates = {}
    if learning_rate is not None:
        if not 0 < learning_rate < 1:
            return JSONResponse({"error": "Learning rate must be between 0 and 1"}, status_code=400)
        updates["learning_rate"] = learning_rate
    
    if weight_decay is not None:
        if not 0 <= weight_decay < 1:
            return JSONResponse({"error": "Weight decay must be between 0 and 1"}, status_code=400)
        updates["weight_decay"] = weight_decay
    
    if dropout is not None:
        if not 0 <= dropout < 1:
            return JSONResponse({"error": "Dropout must be between 0 and 1"}, status_code=400)
        updates["dropout"] = dropout
    
    if gradient_clip is not None:
        if gradient_clip <= 0:
            return JSONResponse({"error": "Gradient clip must be positive"}, status_code=400)
        updates["gradient_clip"] = gradient_clip
    
    state.update_params(**updates)
    publish(run_id, {"event": "params_updated", "params": updates})
    
    return {"status": "updated", "params": updates}

@app.get("/api/training/{run_id}/status")
async def get_training_status(run_id: str):
    """Get current training status"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    return {
        "status": state.status,
        "epoch": state.current_epoch,
        "batch": state.current_batch,
        "total_batches": state.total_batches,
        "elapsed_time": time.time() - state.start_time
    }

@app.post("/api/training/{run_id}/analyze")
async def analyze_training(run_id: str, training_history: list):
    """Analyze training progress and provide intelligent suggestions"""
    state = get_training_state(run_id)
    if not state:
        return JSONResponse({"error": "Training job not found"}, status_code=404)
    
    analysis = analyze_training_progress(training_history, state.current_epoch)
    return JSONResponse(analysis)

@app.get("/api/training/{run_id}/insights")
async def get_training_insights(run_id: str):
    """Generate training insights summary"""
    # Load training metrics from run directory
    run_dir = Path("runs") / run_id
    metrics_file = run_dir / "metrics.json"
    
    if not metrics_file.exists():
        return JSONResponse({"error": "No metrics found for this run"}, status_code=404)
    
    try:
        import json
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Calculate summary statistics
        val_losses = [m.get("val_loss") for m in metrics if m.get("val_loss")]
        val_accs = [m.get("val_accuracy") for m in metrics if m.get("val_accuracy")]
        
        summary = {
            "total_epochs": len(metrics),
            "best_val_loss": min(val_losses) if val_losses else None,
            "best_val_acc": max(val_accs) if val_accs else None,
            "training_time": metrics[-1].get("time_elapsed", 0) / 60 if metrics else 0,
            "status": "completed"
        }
        
        insights = generate_training_insights(summary)
        return {"insights": insights, "summary": summary}
        
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate insights: {str(e)}"}, status_code=500)

# ----------------------
# 7️⃣ Export & Reporting
# ----------------------
@app.get("/export/{run_id}")
async def export_model(run_id: str):
    """
    Returns a zip of the trained model and artifacts.
    """
    zip_path = export_run(run_id)
    return FileResponse(zip_path, filename=f"{run_id}.zip")

@app.get("/api/training/{run_id}/report")
async def generate_report(run_id: str):
    """
    Generate comprehensive training report with charts and analysis
    """
    try:
        report = generate_training_report(run_id)
        
        # Save report to file
        report_path = Path("reports") / f"{run_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return JSONResponse(report)
        
    except FileNotFoundError:
        return JSONResponse({"error": "Training run not found or no metrics available"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Failed to generate report: {str(e)}"}, status_code=500)

@app.get("/api/training/{run_id}/export-package")
async def export_model_package(run_id: str):
    """
    Export complete model package with inference code and documentation
    """
    try:
        package_path = generate_model_package(run_id)
        return FileResponse(
            package_path, 
            filename=f"{run_id}_complete_package.zip",
            media_type="application/zip"
        )
    except Exception as e:
        return JSONResponse({"error": f"Failed to create model package: {str(e)}"}, status_code=500)

@app.get("/api/training/list")
async def list_training_runs():
    """
    List all available training runs with basic info
    """
    runs_dir = Path("runs")
    runs = []
    
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            run_info = {
                "run_id": run_dir.name,
                "created_at": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
                "has_metrics": (run_dir / "metrics.json").exists(),
                "has_model": (run_dir / "pytorch_model.bin").exists()
            }
            
            # Add basic metrics if available
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    if metrics:
                        run_info["total_epochs"] = len(metrics)
                        run_info["final_loss"] = metrics[-1].get("train_loss")
                        run_info["best_val_acc"] = max([m.get("val_accuracy", 0) for m in metrics])
                except Exception:
                    pass
                    
            runs.append(run_info)
    
    # Sort by creation time (newest first)
    runs.sort(key=lambda x: x["created_at"], reverse=True)
    return JSONResponse(runs)

@app.delete("/api/training/{run_id}")
async def delete_training_run(run_id: str):
    """
    Delete a training run and all associated files
    """
    try:
        import shutil
        
        # Delete run directory
        run_dir = Path("runs") / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            
        # Delete checkpoints
        checkpoint_dir = Path("checkpoints") / run_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            
        # Delete exports
        export_files = list(Path("exports").glob(f"{run_id}*"))
        for file_path in export_files:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                
        # Delete reports
        report_files = list(Path("reports").glob(f"{run_id}*"))
        for file_path in report_files:
            file_path.unlink()
            
        return {"status": "deleted", "run_id": run_id}
        
    except Exception as e:
        return JSONResponse({"error": f"Failed to delete run: {str(e)}"}, status_code=500)

# ----------------------
# 8️⃣ Hyperparameter Optimization
# ----------------------
@app.post("/api/hyperopt/start")
async def start_hyperopt(payload: dict):
    """
    Start hyperparameter optimization
    """
    run_id = str(uuid.uuid4())
    
    # Validate payload
    required_fields = ["base_config", "search_space"]
    for field in required_fields:
        if field not in payload:
            return JSONResponse({"error": f"Missing required field: {field}"}, status_code=400)
    
    # Start optimization
    start_hyperparameter_optimization(run_id, payload)
    
    return {"status": "started", "run_id": run_id, "optimization_type": "hyperparameter"}

@app.get("/api/hyperopt/search-spaces")
async def get_search_spaces():
    """
    Get predefined search spaces for different scenarios
    """
    return JSONResponse({
        "search_spaces": SEARCH_SPACES,
        "descriptions": {
            "classification_basic": "Basic search space for text classification tasks",
            "classification_advanced": "Advanced search space with more parameters",
            "small_dataset": "Optimized for datasets with <10k samples",
            "large_dataset": "Optimized for datasets with >100k samples"
        }
    })

@app.post("/api/hyperopt/suggest-space")
async def suggest_search_space(payload: dict):
    """
    Suggest optimal search space based on dataset characteristics
    """
    dataset_size = payload.get("dataset_size", 1000)
    task_type = payload.get("task_type", "classification")
    model_size = payload.get("model_size", "base")  # base, large, small
    
    # Select appropriate search space
    if dataset_size < 1000:
        space_key = "small_dataset"
    elif dataset_size > 50000:
        space_key = "large_dataset"
    else:
        space_key = "classification_basic"
    
    suggested_space = SEARCH_SPACES[space_key].copy()
    
    # Adjust based on model size
    if model_size == "large":
        suggested_space["learning_rate"] = [lr for lr in suggested_space["learning_rate"] if lr <= 3e-5]
        suggested_space["batch_size"] = [bs for bs in suggested_space["batch_size"] if bs <= 16]
    elif model_size == "small":
        suggested_space["learning_rate"] = [lr for lr in suggested_space["learning_rate"] if lr >= 2e-5]
        suggested_space["batch_size"] = [bs for bs in suggested_space["batch_size"] if bs >= 16]
    
    return JSONResponse({
        "suggested_space": suggested_space,
        "reasoning": f"Selected {space_key} based on dataset size {dataset_size} and {model_size} model",
        "estimated_trials": min(20, len(list(itertools.product(*suggested_space.values()))))
    })

# ----------------------
# 9️⃣ Health Check
# ----------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Backend is running!"}
