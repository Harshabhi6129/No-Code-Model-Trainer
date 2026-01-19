# backend/model_params.py

from typing import Dict, Any, List
from transformers import AutoConfig


def get_model_candidates(task: str = "classification") -> List[Dict[str, Any]]:
    """Return list of recommended models for the task."""
    models = [
        {"id": "distilbert-base-uncased", "name": "DistilBERT", "size": "small", "speed": "fast"},
        {"id": "bert-base-uncased", "name": "BERT Base", "size": "medium", "speed": "medium"},
        {"id": "roberta-base", "name": "RoBERTa", "size": "medium", "speed": "medium"},
        {"id": "albert-base-v2", "name": "ALBERT", "size": "small", "speed": "fast"},
    ]
    return models


def get_model_params(model_id: str) -> Dict[str, Any]:
    """Return a minimal parameter schema for the model."""
    return {
        "title": "Training Config",
        "type": "object",
        "properties": {
            "num_epochs": {"type": "integer", "default": 3, "minimum": 1, "maximum": 100},
            "batch_size": {"type": "integer", "default": 8, "enum": [4, 8, 16, 32]},
            "learning_rate": {"type": "number", "default": 2e-5, "minimum": 1e-6, "maximum": 1e-3},
            "weight_decay": {"type": "number", "default": 0.01, "minimum": 0.0, "maximum": 0.1}
        }
    }
