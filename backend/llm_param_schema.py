# backend/llm_param_schema.py

import json, re, functools
from typing import Dict
from llm_service import query_llm_gemini

_PROMPT = """
You are a senior ML/NLP engineer.
Return ONLY a valid JSON-Schema (no markdown, no prose) describing the
top hyper-parameters a user should tune when training or fine-tuning
the HuggingFace model **{model_id}**.

Limit to 3–7 meaningful parameters. Example format:
{
  "title": "DistilBERT fine-tune",
  "type":  "object",
  "properties": {
    "epochs":        {"type":"integer","default":3,"minimum":1},
    "learning_rate": {"type":"number","default":2e-5,"minimum":1e-6},
    "batch_size":    {"type":"integer","default":8,"minimum":1}
  }
}
"""

JSON_RE = re.compile(r"\{.*\}", re.S)

@functools.lru_cache(maxsize=64)
def llm_schema(model_id: str) -> Dict:
    raw = query_llm_gemini(_PROMPT.format(model_id=model_id))
    match = JSON_RE.search(raw)
    if not match:
        raise ValueError("LLM response did not contain valid JSON.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Bad JSON from LLM: {e}") from e
