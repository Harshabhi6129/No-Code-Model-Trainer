# backend/model_params.py

from typing import Dict, Any, List
from transformers import AutoConfig
from llm_param_schema import llm_schema
from llm_service import describe_params
from default_param_schemas import DEFAULT_PARAMS


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
    """Alias for get_param_schema."""
    return get_param_schema(model_id)


# ------------------------ add descriptions to schema ------------------------
def _with_descriptions(model_id: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        names = list(schema.get("properties", {}).keys())
        descs = describe_params(model_id, names)
        for k, v in descs.items():
            if k in schema["properties"]:
                schema["properties"][k]["description"] = v
    except Exception:
        pass
    return schema

# ------------------------ main param schema logic ------------------------
def get_param_schema(model_id: str) -> Dict[str, Any]:
    model_id = model_id.strip()

    # 1. Curated fallback schemas
    if model_id in DEFAULT_PARAMS:
        return _with_descriptions(model_id, DEFAULT_PARAMS[model_id])

    # 2. Generic HuggingFace transformer match
    if any(t in model_id.lower() for t in ["bert", "roberta", "electra", "distilbert", "albert"]):
        return _with_descriptions(model_id, DEFAULT_PARAMS["HF_GENERIC"])

    # 3. HuggingFace config check – if it loads, assume transformer
    try:
        AutoConfig.from_pretrained(model_id)
        return _with_descriptions(model_id, DEFAULT_PARAMS["HF_GENERIC"])
    except Exception:
        pass

    # 4. LLM fallback if all else fails
    try:
        return _with_descriptions(model_id, llm_schema(model_id))
    except Exception:
        pass

    # 5. Minimal fail-safe
    return {
        "title": "Training Config",
        "type": "object",
        "properties": {
            "epochs":      {"type": "integer", "default": 3},
            "batch_size":  {"type": "integer", "default": 8}
        }
    }
