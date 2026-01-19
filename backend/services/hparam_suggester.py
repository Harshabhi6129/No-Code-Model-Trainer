"""
Very lightweight “agent” that recommends sensible defaults.
You can swap the heuristic logic with a LangChain chain or a Gemini
prompt later – the interface stays the same.
"""
from typing import Dict

from services.llm_service import _safe_json, _call_gemini   # reuse helpers

_PROMPT = """
You are a senior ML engineer.

Suggest *up to* 7 hyper-parameters and defaults for fine-tuning
the HuggingFace model **{model_id}** on a dataset with:

- {row_cnt} rows
- {class_cnt} target classes
- Average text length ≈ {avg_len} tokens

Return ONLY valid JSON *object* (no markdown) where keys are param names
and values are the recommended default numbers/booleans.
""".strip()


def suggest_hparams(model_id: str, stats: Dict) -> Dict:
    rows      = stats.get("row_count", 1000)
    classes   = stats.get("num_labels", 2)
    avg_len   = stats.get("avg_length", 64)

    # --- simple heuristic shortcut ---
    if rows < 10_000:
        guess = {
            "epochs"       : 4,
            "batch_size"   : 8,
            "learning_rate": 3e-5,
            "warmup_steps" : 200,
            "max_seq_length": min(256, max(64, avg_len * 2)),
            "use_peft"     : True,
        }
        return guess

    # --- fallback to Gemini for richer advice ---
    try:
        prompt = _PROMPT.format(
            model_id=model_id, row_cnt=rows, class_cnt=classes, avg_len=avg_len
        )
        return _safe_json(_call_gemini(prompt))
    except Exception:
        # final safety-net defaults
        return {
            "epochs"       : 3,
            "batch_size"   : 16,
            "learning_rate": 2e-5,
            "use_peft"     : False,
        }
