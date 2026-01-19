# backend/llm_service.py

import os, json, re, requests
from typing import Dict, List
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Helper: Detect all available keys
def _load_api_keys() -> List[str]:
    keys = []
    # Primary key
    if os.getenv("GOOGLE_API_KEY"):
        keys.append(os.getenv("GOOGLE_API_KEY"))
    
    # Secondary keys (GOOGLE_API_KEY_2, _3, etc.)
    i = 2
    while True:
        k = os.getenv(f"GOOGLE_API_KEY_{i}") or os.getenv(f"GOOGLE_API_KEY{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    
    if not keys and GOOGLE_API_KEY:
         keys.append(GOOGLE_API_KEY)
         
    return keys

class KeyManager:
    def __init__(self):
        self.keys = _load_api_keys()
        self.current_idx = 0
        if not self.keys:
            print("WARNING: No Google API Keys found in .env")

    def get_next_key(self) -> str:
        if not self.keys:
            raise RuntimeError("No API keys available")
        
        # Round-robin
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key
        
    def get_all_keys(self) -> List[str]:
        return self.keys

# Global manager instance
_KEY_MANAGER = KeyManager()

JSON_RE        = re.compile(r"\{.*\}", re.S)
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def _safe_json(text: str) -> Dict:
    match = JSON_RE.search(text or "")
    if not match:
        raise ValueError("LLM response contained no JSON")
    return json.loads(match.group(0))

def _call_gemini(prompt: str, _unused_key: str = None) -> str:
    """
    Calls Gemini with automatic key rotation and retries.
    Ignores the `_unused_key` argument to maintain signature compatibility if needed,
    but prefers using the internal KeyManager.
    """
    max_retries = len(_KEY_MANAGER.get_all_keys())
    # Try at least once, up to matching the number of keys we have
    if max_retries == 0:
        raise RuntimeError("No API keys configured")
    
    last_error = None
    
    for attempt in range(max_retries):
        api_key = _KEY_MANAGER.get_next_key()
        try:
            response = requests.post(
                f"{GEMINI_URL}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=25
            )
            
            # If rate limit (429) or server error (5xx), raise to trigger retry
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
                
            data = response.json()
            
            # Check for error in JSON body even if 200 OK (common in some APIs, though standard HTTP should catch above)
            if "error" in data:
                 raise RuntimeError(f"Gemini API Error: {data['error']}")
                 
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
            
        except Exception as e:
            # print(f"Attempt {attempt+1} failed with key ...{api_key[-4:]}: {e}")
            last_error = e
            continue
            
    raise RuntimeError(f"All API keys failed. Last error: {last_error}")

def query_llm_gemini(prompt: str) -> str:
    return _call_gemini(prompt).strip()

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
    raw = _call_gemini(prompt)
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
    raw = _call_gemini(build_model_candidate_prompt(task, stats))
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
        return _safe_json(_call_gemini(prompt))
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
        response = _call_gemini(prompt)
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
        return _call_gemini(prompt)
    except Exception:
        return "Training completed successfully! Check the metrics dashboard for detailed results."

def generate_comprehensive_report(summary: dict, config: dict, metrics_sample: list) -> dict:
    """
    Generate a full training report with executive summary and actionable advice.
    """
    prompt = f"""
    You are a Senior ML Engineer writing a post-training report.
    
    TRAINING STATS:
    - Total Epochs: {summary.get('total_epochs')}
    - Final Train Loss: {summary.get('final_train_loss')}
    - Final Val Loss: {summary.get('final_val_loss')}
    - Best Val Accuracy: {summary.get('best_val_acc')}
    - Total Time: {summary.get('total_time', 0)/60:.1f} min
    
    CONFIGURATION:
    {json.dumps(config, indent=2)}
    
    METRICS SAMPLE (Start, Middle, End):
    {json.dumps(metrics_sample, indent=2)}
    
    Generate a JSON report with:
    {{
      "executive_summary": "2-3 sentences summarizing the overall outcome (Success/Failure) and key result.",
      "key_findings": [
        "Bullet point 1 about convergence speed",
        "Bullet point 2 about overfitting/underfitting patterns",
        "Bullet point 3 about stability"
      ],
      "tuning_advice": [
        "Specific suggestion for next run (e.g. 'Increase batch size to 32')",
        "Another suggestion (e.g. 'Use cosine schedule')"
      ]
    }}
    """.strip()
    
    try:
        response = _call_gemini(prompt)
        return _safe_json(response)
    except Exception as e:
        print(f"DEBUG: API Error: {e}")
        # Fallback structure
        return {
            "executive_summary": "Training completed. Please review the charts for details.",
            "key_findings": ["Automated analysis failed due to API error."],
            "tuning_advice": ["Check manual logs for insights."]
        }

# ───────── Dataset Intent Analysis ─────────
def analyze_dataset_intent(df_head: str, filename: str, user_description: str = "") -> dict:
    """
    Analyze dataset preview to determine task type, domain, and quality insights.
    """
    prompt = f"""
    Analyze this dataset sample to understand the user's goal.
    
    Filename: {filename}
    User Description: {user_description if user_description else "Not provided"}
    
    Data Sample:
    {df_head}
    
    Return ONLY JSON with this structure:
    {{
      "detected_task": "Text Classification | NER | Summarization | Translation | Question Answering",
      "confidence": 0.0-1.0,
      "domain": "Medical | Finance | Legal | Social Media | General",
      "summary": "Brief 1-sentence description of what this dataset is for",
      "recommendation_prompt": "A detailed prompt describing the task, suitable for asking an AI to recommend a model. E.g. 'I have a dataset of customer reviews labeled with sentiment. I want to train a model to classify them into positive, negative, and neutral.'",
      "potential_challenges": ["List of 2-3 potential issues like noise, length, etc."],
      "recommended_model_type": "DistilBERT | T5 | BART | etc."
    }}
    """.strip()
    
    try:
        response = _call_gemini(prompt)
        return _safe_json(response)
    except Exception as e:
        print(f"DEBUG: API Intention Analysis Failed: {e}")
        
        # Smart Heuristic Fallback
        # Check for common column names to guess the task
        detected_task = "Unknown"
        summary_text = "Analysis failed, using heuristic fallback."
        rec_model = "bert-base-uncased"
        
        lower_head = df_head.lower()
        
        if "label" in lower_head or "target" in lower_head or "sentiment" in lower_head:
            detected_task = "Text Classification"
            summary_text = "Dataset appears to contain text and labels, suitable for classification."
            rec_model = "distilbert-base-uncased"
        elif "summary" in lower_head or "summarize" in lower_head:
            detected_task = "Summarization"
            rec_model = "sshleifer/distilbart-cnn-12-6"
        elif "translate" in lower_head:
            detected_task = "Translation"
            rec_model = "t5-small"
        
        return {
            "detected_task": detected_task,
            "confidence": 0.4,
            "domain": "General (Heuristic)",
            "summary": summary_text,
            "recommendation_prompt": f"I have a dataset named {filename}. Please recommend a suitable model for {detected_task}.",
            "potential_challenges": ["Heuristic analysis only - please verify manually."],
            "recommended_model_type": rec_model
        }

# ───────── Model Registry Enrichment ─────────
def enrich_model_metadata(model_id: str, description: str) -> dict:
    """
    Generate pros, cons, and best use cases for a model using Gemini.
    """
    prompt = f"""
    Analyze this HuggingFace model: {model_id}
    Description: {description[:1000]}...

    Return ONLY JSON with this structure:
    {{
      "pros": ["list of 2-3 strengths"],
      "cons": ["list of 2-3 limitations"],
      "best_for": "Specific ideal use case (e.g. 'Short text classification', 'Medical NER')"
    }}
    """.strip()
    
    try:
        # Use API Key 2 to distribute load
        response = _call_gemini(prompt)
        return _safe_json(response)
    except Exception:
        return {
            "pros": ["Standard Transformer architecture", "Pre-trained on large corpus"],
            "cons": ["May require fine-tuning", "Resource intensive"],
            "best_for": "General NLP tasks"
        }
