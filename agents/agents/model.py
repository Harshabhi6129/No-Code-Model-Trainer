"""
Model Agent — selects the best base model and training recipe.

Architecture (research-grounded):
  • SONNET tier: needs real reasoning over 55+ model catalog + dataset profile
  • System prompt CACHED: the full catalog is ~3 000 tokens; caching saves ~90%
    on every repeated run that hits the same prefix
  • Output validated via ModelRecipe (Pydantic): semantic guards on LR range,
    batch size, LoRA field consistency — catches hallucinations before training
  • Override path bypasses LLM entirely (Phase B1 will add Optuna warm-start here)
"""
from __future__ import annotations

import logging

from .base import BaseAgent, AgentContext, AgentResult, SONNET
from .model_catalog import catalog_summary_for_prompt
from .schemas import ModelRecipe

logger = logging.getLogger(__name__)

# ── System prompt (CACHED — full 55-model catalog exceeds 1 024-token minimum) ─

_SYSTEM_TEMPLATE = """You are the Model Agent for ModelForge. Your job is to select the best base model
and training recipe for this ML task.

=== AVAILABLE MODELS ===
{catalog}

=== TRAINING RECIPE PRIORS (research-backed defaults) ===
LoRA rank (lora_r):
  • 8–16 for classification / simple tasks  (sweet spot: 16)
  • 32 for instruction tuning / generation
  • Apply LoRA to ALL linear layers for best results

Learning rate:
  • LoRA / QLoRA: 1e-4 to 3e-4 with cosine schedule
  • Full fine-tune: 2e-5 to 5e-5
  • NEVER output a value outside [1e-7, 1e-2]

Batch size: prefer 16 or 32; use 8 for large models on small GPUs.
Epochs: 2–3 for >1 000 rows; 5 for <200 rows.
QLoRA (4-bit) is the memory-efficient default for LLMs; LoRA (16-bit) for encoders.

=== INSTRUCTIONS ===
• Choose ONLY model IDs from the catalog above — do NOT invent IDs.
• Prefer smaller models when dataset is small (<1 000 rows) or speed matters.
• Prefer higher-quality models (roberta-base, deberta-v3-small) when accuracy is priority.

Output ONLY valid JSON — no markdown fences, no extra commentary:
{{
  "base_model": "<model_id from catalog above>",
  "training_approach": "full_finetune" | "lora" | "qlora" | "embed_classify",
  "lora_r": <int or null>,
  "lora_alpha": <int or null>,
  "lora_target_modules": null,
  "learning_rate": <float in [1e-7, 1e-2]>,
  "num_epochs": <int 1-20>,
  "batch_size": <int, power of 2>,
  "max_length": <int 16-512>,
  "warmup_ratio": <float 0.0-0.5>,
  "weight_decay": <float 0.0-0.5>,
  "reasoning": "<1-2 sentences explaining your choices>"
}}"""


class ModelAgent(BaseAgent):
    name  = "Model"
    model = SONNET  # Needs real reasoning over 55+ models + dataset characteristics

    async def run(self, context: AgentContext) -> AgentResult:
        ovr = context.hyperparameter_overrides

        # User-pinned model: build recipe from overrides, skip LLM
        if ovr.get("model_id"):
            recipe = _override_recipe(ovr, context)
            # Still validate the override recipe — catches user-supplied bad values
            validated, err = self._parse_llm_json(
                __import__("json").dumps(recipe), ModelRecipe
            )
            if validated:
                context.model_recipe = validated.model_dump()
                return _success_result(context.model_recipe)
            # If validation fails on override, warn but proceed with raw recipe
            logger.warning("Override recipe validation issue: %s", err)
            context.model_recipe = recipe
            return _success_result(recipe)

        # Build system prompt with full catalog (will be cached by Anthropic)
        system = _SYSTEM_TEMPLATE.format(catalog=catalog_summary_for_prompt())

        import json
        prompt = json.dumps(
            {
                "task_spec":    context.task_spec,
                "data_profile": {
                    # Compact profile — only what model selection needs
                    # Avoids injecting raw data or oversized context
                    "num_rows":           context.data_profile.get("num_rows"),
                    "num_classes":        context.data_profile.get("num_classes"),
                    "avg_input_len":      context.data_profile.get("avg_input_len"),
                    "class_balance_ratio": context.data_profile.get("class_balance_ratio"),
                    "label_noise_estimate": context.data_profile.get("label_noise_estimate", 0.0),
                    "issues":             context.data_profile.get("issues", []),
                },
            },
            indent=2,
        )

        raw = await self._chat(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            cache_system=True,   # catalog is large — this is where caching pays off
        )

        recipe_model, err = self._parse_llm_json(raw, ModelRecipe)
        if recipe_model is None:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message=(
                    f"Could not produce a valid model recipe ({err}). "
                    "Please try again or select a model manually."
                ),
            )

        recipe = recipe_model.model_dump()
        context.model_recipe = recipe
        return _success_result(recipe)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _override_recipe(ovr: dict, context: AgentContext) -> dict:
    """Build a recipe from user overrides with research-backed defaults."""
    rows = context.data_profile.get("num_rows", 500)
    approach = ovr.get("training_approach", "full_finetune")
    lora_r = ovr.get("lora_r", 16)
    return {
        "base_model":          ovr["model_id"],
        "training_approach":   approach,
        "lora_r":              lora_r if approach in ("lora", "qlora") else None,
        "lora_alpha":          ovr.get("lora_alpha", lora_r * 2) if approach in ("lora", "qlora") else None,
        "lora_target_modules": None,
        "learning_rate":       ovr.get("learning_rate", 2e-4 if approach in ("lora", "qlora") else 2e-5),
        "num_epochs":          ovr.get("num_epochs", 3 if rows >= 200 else 5),
        "batch_size":          ovr.get("batch_size", 16),
        "max_length":          ovr.get("max_length", 128),
        "warmup_ratio":        ovr.get("warmup_ratio", 0.1),
        "weight_decay":        ovr.get("weight_decay", 0.01),
        "reasoning":           f"Model `{ovr['model_id']}` selected by user override.",
    }


def _success_result(recipe: dict) -> AgentResult:
    approach   = str(recipe.get("training_approach", "unknown")).replace("_", " ").upper()
    base_model = recipe.get("base_model", "unknown model")
    lora_info  = ""
    if recipe.get("lora_r"):
        lora_info = f" (r={recipe['lora_r']}, α={recipe.get('lora_alpha', '?')})"
    return AgentResult(
        agent_name="Model", success=True, output=recipe,
        message=(
            f"Selected **`{base_model}`** with **{approach}{lora_info}**.\n"
            f"{recipe.get('reasoning', '')}\n"
            f"Training for {recipe.get('num_epochs')} epochs, "
            f"lr={recipe.get('learning_rate'):.2e}, "
            f"batch={recipe.get('batch_size')}."
        ),
        next_agent="Train",
    )
