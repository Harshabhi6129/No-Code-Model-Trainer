# backend/default_param_schemas.py
DEFAULT_PARAMS = {
    "HF_GENERIC": {
        "title": "Transformer Fine-Tune",
        "type": "object",
        "properties": {
            "epochs":        {"type": "integer", "default": 3,   "minimum": 1},
            "batch_size":    {"type": "integer", "default": 16,  "minimum": 1},
            "learning_rate": {"type": "number",  "default": 3e-5,"minimum": 1e-6,"maximum": 1e-2},
            "warmup_steps":  {"type": "integer", "default": 500, "minimum": 0},
            "max_seq_length":{"type": "integer", "default": 128, "minimum": 32,"maximum": 1024},
            "weight_decay":  {"type": "number",  "default": 0.01,"minimum": 0,"maximum": 1},
            "use_peft":      {"type": "boolean", "default": False},
        },
        "required": ["epochs", "batch_size", "learning_rate"],
    }
}
