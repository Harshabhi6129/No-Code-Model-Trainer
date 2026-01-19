"""
Architecture-to-Parameter Mapping for Dynamic UI Generation.
Maps model architectures to their specific hyperparameters.
"""

# Universal parameters (apply to all models)
UNIVERSAL_PARAMS = {
    "learning_rate": {
        "type": "float",
        "default": 2e-5,
        "min": 1e-6,
        "max": 1e-3,
        "description": "Learning rate for optimizer"
    },
    "num_epochs": {
        "type": "int",
        "default": 3,
        "min": 1,
        "max": 10,
        "options": [1, 2, 3, 4, 5, 10],
        "description": "Number of training epochs"
    },
    "batch_size": {
        "type": "int",
        "default": 8,
        "min": 1,
        "max": 64,
        "options": [4, 8, 16, 32],
        "description": "Batch size for training"
    },
    "weight_decay": {
        "type": "float",
        "default": 0.01,
        "min": 0.0,
        "max": 0.1,
        "description": "Weight decay (L2 regularization)"
    }
}

# Architecture-specific parameters
ARCHITECTURE_PARAMS = {
    # Transformer-based models (BERT, RoBERTa, DistilBERT)
    "BertForSequenceClassification": {
        "max_seq_length": {
            "type": "int",
            "default": 512,
            "min": 64,
            "max": 512,
            "options": [128, 256, 512],
            "description": "Maximum sequence length"
        },
        "warmup_ratio": {
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 0.3,
            "description": "Warmup ratio for learning rate scheduler"
        }
    },
    "RobertaForSequenceClassification": {
        "max_seq_length": {
            "type": "int",
            "default": 512,
            "min": 64,
            "max": 512,
            "options": [128, 256, 512],
            "description": "Maximum sequence length"
        },
        "warmup_ratio": {
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 0.3,
            "description": "Warmup ratio for learning rate scheduler"
        }
    },
    "DistilBertForSequenceClassification": {
        "max_seq_length": {
            "type": "int",
            "default": 512,
            "min": 64,
            "max": 512,
            "options": [128, 256, 512],
            "description": "Maximum sequence length"
        }
    },
    
    # Causal LLMs (Llama, GPT)
    "LlamaForCausalLM": {
        "lora_r": {
            "type": "int",
            "default": 8,
            "min": 4,
            "max": 64,
            "options": [4, 8, 16, 32],
            "description": "LoRA rank"
        },
        "lora_alpha": {
            "type": "int",
            "default": 16,
            "min": 8,
            "max": 64,
            "options": [8, 16, 32],
            "description": "LoRA alpha"
        },
        "quantization_bit": {
            "type": "int",
            "default": 4,
            "options": [4, 8],
            "description": "Quantization bits (4-bit, 8-bit)"
        }
    },
    "GPT2LMHeadModel": {
        "max_seq_length": {
            "type": "int",
            "default": 1024,
            "min": 256,
            "max": 1024,
            "options": [256, 512, 1024],
            "description": "Maximum sequence length"
        }
    },
    
    # Vision models
    "ViTForImageClassification": {
        "image_size": {
            "type": "int",
            "default": 224,
            "options": [224, 384],
            "description": "Input image size"
        },
        "patch_size": {
            "type": "int",
            "default": 16,
            "options": [16, 32],
            "description": "Patch size for ViT"
        }
    },
    
    # Token classification (NER)
    "BertForTokenClassification": {
        "max_seq_length": {
            "type": "int",
            "default": 512,
            "min": 64,
            "max": 512,
            "options": [128, 256, 512],
            "description": "Maximum sequence length"
        },
        "label_smoothing": {
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 0.3,
            "description": "Label smoothing factor"
        }
    }
}

# Fallback for unknown architectures
DEFAULT_PARAMS = {
    "max_seq_length": {
        "type": "int",
        "default": 512,
        "min": 64,
        "max": 512,
        "options": [128, 256, 512],
        "description": "Maximum sequence length"
    }
}

def get_params_for_architecture(architecture: str) -> dict:
    """
    Get the parameter schema for a given architecture.
    Returns universal params + architecture-specific params.
    """
    # Start with universal params
    params = UNIVERSAL_PARAMS.copy()
    
    # Add architecture-specific params
    arch_params = ARCHITECTURE_PARAMS.get(architecture, DEFAULT_PARAMS)
    params.update(arch_params)
    
    return params

def get_params_from_architectures(architectures: list) -> dict:
    """
    Given a list of architectures (from config.json),
    return the merged parameter schema.
    """
    if not architectures:
        return {**UNIVERSAL_PARAMS, **DEFAULT_PARAMS}
    
    # Use the first architecture in the list
    primary_arch = architectures[0]
    return get_params_for_architecture(primary_arch)
