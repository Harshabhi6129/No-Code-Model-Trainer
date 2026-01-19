# backend/llm_service.py

import os, json, re, requests
from typing import Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
GOOGLE_API_KEY3 = os.getenv("GOOGLE_API_KEY3")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
JSON_RE        = re.compile(r"\{.*\}", re.S)

def _safe_json(text: str) -> Dict:
    match = JSON_RE.search(text or "")
    if not match:
        raise ValueError("LLM response contained no JSON")
    return json.loads(match.group(0))

@lru_cache(maxsize=128)
def _call_gemini(prompt: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("Missing API key")
    response = requests.post(
        f"{GEMINI_URL}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=25
    )
    response.raise_for_status()
    data = response.json()
    for path in (["candidates", 0, "content", "parts", 0, "text"],
                 ["responses", 0, "candidates", 0, "content", "parts", 0, "text"]):
        try:
            node = data
            for k in path:
                node = node[k]
            return str(node)
        except Exception:
            continue
    raise RuntimeError("Unrecognized Gemini JSON shape")

def query_llm_gemini(prompt: str) -> str:
    return _call_gemini(prompt, GOOGLE_API_KEY1).strip()

# ───────── Task Suggestion ─────────
class LLMResponse(BaseModel):
    task: str
    recommended_model: str
    reason: str

def suggest_models(task_prompt: str) -> LLMResponse:
    prompt = f"""
Suggest the best HuggingFace model for this NLP task:

{task_prompt}

Return ONLY this JSON:

{{
  "task": "<task-name>",
  "recommended_model": "<huggingface-model-id>",
  "reason": "<short explanation>"
}}
""".strip()
    raw = _call_gemini(prompt, GOOGLE_API_KEY1)
    return LLMResponse(**_safe_json(raw))

# ───────── Model Candidates ─────────
def build_model_candidate_prompt(task: str, stats: dict) -> str:
    sample = [str(r.get("text")) for r in stats.get("preview", [])][:3]
    return f"""
You are an NLP architect.

TASK: {task}
Sample texts: {sample}

Return JSON with keys:
  "transformers", "finetuned", "advice"

Example:
{{
  "transformers": ["distilbert-base-uncased", "bert-base-uncased"],
  "finetuned":    ["dslim/bert-base-NER", "finiteautomata/bertweet-base-sentiment-analysis"],
  "advice":       "Start with RoBERTa for accurate classification."
}}
""".strip()

def get_model_candidates(task: str, stats: dict) -> Dict:
    raw = _call_gemini(build_model_candidate_prompt(task, stats), GOOGLE_API_KEY1)
    return _safe_json(raw)

# ───────── Param Tooltips ─────────
def describe_params(model_id: str, param_names: list[str]) -> dict[str, str]:
    if not param_names:
        return {}
    prompt = f"""
You are an ML tutor.
Give <15-word descriptions for each of these {model_id} parameters:
{', '.join(param_names)}

Return JSON.
""".strip()
    try:
        return _safe_json(_call_gemini(prompt, GOOGLE_API_KEY2))
    except Exception:
        return {}

# ───────── Training Analysis ─────────
def analyze_training_progress(training_history: list, current_epoch: int) -> dict:
    """Analyze training metrics and provide intelligent suggestions"""
    if len(training_history) < 3:
        return {"status": "insufficient_data", "suggestions": [], "auto_actions": []}
    
    recent_metrics = training_history[-5:]  # Last 5 epochs
    
    # Extract key metrics
    train_losses = [m.get("train_loss", 0) for m in recent_metrics]
    val_losses = [m.get("val_loss", 0) for m in recent_metrics if m.get("val_loss")]
    train_accs = [m.get("train_accuracy", 0) for m in recent_metrics]
    val_accs = [m.get("val_accuracy", 0) for m in recent_metrics if m.get("val_accuracy")]
    grad_norms = [m.get("grad_norm", 0) for m in recent_metrics if m.get("grad_norm")]
    
    prompt = f"""
Analyze this ML training progress and provide expert recommendations:

Recent Training Metrics (last 5 epochs):
Train Losses: {train_losses}
Validation Losses: {val_losses}
Train Accuracies: {train_accs}
Validation Accuracies: {val_accs}
Gradient Norms: {grad_norms}
Current Epoch: {current_epoch}

Analyze and return JSON with:
{{
  "status": "healthy|overfitting|underfitting|diverging|plateau",
  "confidence": 0.0-1.0,
  "issues": ["list of detected issues"],
  "suggestions": ["actionable recommendations"],
  "auto_actions": [
    {{"action": "reduce_lr", "value": 0.00001, "reason": "explanation"}},
    {{"action": "early_stop", "reason": "explanation"}}
  ]
}}

Focus on:
1. Overfitting (val_loss increasing while train_loss decreasing)
2. Learning rate issues (loss plateaus, gradient explosions)
3. Training efficiency recommendations
""".strip()
    
    try:
        response = _call_gemini(prompt, GOOGLE_API_KEY2)
        analysis = _safe_json(response)
        
        # Add automatic pattern detection
        if len(val_losses) >= 3:
            # Check for overfitting
            if val_losses[-1] > val_losses[-2] > val_losses[-3] and train_losses[-1] < train_losses[-2]:
                analysis["issues"] = analysis.get("issues", []) + ["Validation loss increasing while training loss decreasing"]
                analysis["auto_actions"] = analysis.get("auto_actions", []) + [
                    {"action": "reduce_lr", "value": 0.5, "reason": "Reduce overfitting"}
                ]
        
        # Check gradient norms
        if grad_norms and max(grad_norms) > 10:
            analysis["issues"] = analysis.get("issues", []) + ["High gradient norms detected"]
            analysis["auto_actions"] = analysis.get("auto_actions", []) + [
                {"action": "gradient_clip", "value": 1.0, "reason": "Prevent gradient explosion"}
            ]
            
        return analysis
        
    except Exception as e:
        return {
            "status": "analysis_failed",
            "error": str(e),
            "suggestions": ["Unable to analyze training progress"],
            "auto_actions": []
        }

def generate_training_insights(metrics_summary: dict) -> str:
    """Generate human-readable training insights"""
    prompt = f"""
Generate a brief, encouraging training summary for a user:

Training Summary:
- Total Epochs: {metrics_summary.get('total_epochs', 0)}
- Best Validation Loss: {metrics_summary.get('best_val_loss', 'N/A')}
- Best Validation Accuracy: {metrics_summary.get('best_val_acc', 'N/A')}
- Training Time: {metrics_summary.get('training_time', 'N/A')} minutes
- Final Status: {metrics_summary.get('status', 'unknown')}

Write 2-3 sentences highlighting the key achievements and any notable patterns.
Be encouraging and focus on what went well.
""".strip()

    try:
        return _call_gemini(prompt, GOOGLE_API_KEY3)
    except Exception:
        return "Training completed successfully! Check the metrics dashboard for detailed results."
