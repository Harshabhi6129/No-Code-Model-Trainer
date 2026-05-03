from __future__ import annotations
import json
from .base import BaseAgent, AgentContext, AgentResult
from .model_catalog import catalog_summary_for_prompt

_SYSTEM_TEMPLATE = """You are the Model Agent for ModelForge. Your job is to select the best base model
and training recipe for this ML task.

=== AVAILABLE MODELS ===
{catalog}

=== INSTRUCTIONS ===
Choose from the models listed above. Do NOT invent model IDs.
Prefer smaller, faster models when the dataset is small (<1 000 rows) or the user wants quick results.
Prefer higher-quality models (roberta-base, microsoft/deberta-v3-small) when accuracy is the priority.

Output ONLY valid JSON — no markdown, no commentary:
{{
  "base_model": "<model_id from catalog above>",
  "training_approach": "full_finetune" | "lora" | "qlora" | "embed_classify",
  "lora_r": <int or null>,
  "lora_alpha": <int or null>,
  "lora_target_modules": [<string>] or null,
  "learning_rate": <float>,
  "num_epochs": <int>,
  "batch_size": <int>,
  "max_length": <int>,
  "warmup_ratio": <float>,
  "weight_decay": <float>,
  "reasoning": "<1-2 sentences explaining your choices>"
}}"""

_REQUIRED = {"base_model", "training_approach", "num_epochs", "learning_rate", "batch_size"}


class ModelAgent(BaseAgent):
    name = "Model"

    async def run(self, context: AgentContext) -> AgentResult:
        ovr = context.hyperparameter_overrides

        # If the user has already pinned a model via overrides, skip LLM selection
        # and build a sensible default recipe around it.
        if ovr.get("model_id"):
            recipe = _override_recipe(ovr, context)
            context.model_recipe = recipe
            return _success_result(recipe)

        system = _SYSTEM_TEMPLATE.format(catalog=catalog_summary_for_prompt())
        prompt = json.dumps({
            "task_spec":    context.task_spec,
            "data_profile": context.data_profile,
        }, indent=2)

        raw = await self._chat(system=system, messages=[{"role": "user", "content": prompt}])

        # Strip accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            recipe = json.loads(raw)
        except json.JSONDecodeError:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message="Could not parse model recipe from Claude. Please try again.",
            )

        if not isinstance(recipe, dict):
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message="Model recipe was not a JSON object. Please try again.",
            )

        missing = _REQUIRED - recipe.keys()
        if missing:
            return AgentResult(
                agent_name=self.name, success=False, output=recipe,
                message=f"Model recipe missing fields: {', '.join(sorted(missing))}. Please try again.",
            )

        context.model_recipe = recipe
        return _success_result(recipe)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_recipe(ovr: dict, context: AgentContext) -> dict:
    """Build a recipe from user overrides, filling defaults from data_profile."""
    profile = context.data_profile
    rows    = profile.get("num_rows", 500)
    return {
        "base_model":           ovr["model_id"],
        "training_approach":    ovr.get("training_approach", "full_finetune"),
        "lora_r":               ovr.get("lora_r", 8),
        "lora_alpha":           ovr.get("lora_alpha", 16),
        "lora_target_modules":  None,
        "learning_rate":        ovr.get("learning_rate", 2e-5),
        "num_epochs":           ovr.get("num_epochs", 3 if rows >= 200 else 5),
        "batch_size":           ovr.get("batch_size", 16),
        "max_length":           ovr.get("max_length", 128),
        "warmup_ratio":         ovr.get("warmup_ratio", 0.1),
        "weight_decay":         ovr.get("weight_decay", 0.01),
        "reasoning":            f"Model `{ovr['model_id']}` selected by user override.",
    }


def _success_result(recipe: dict) -> AgentResult:
    approach   = str(recipe.get("training_approach", "unknown")).replace("_", " ").upper()
    base_model = recipe.get("base_model", "unknown model")
    return AgentResult(
        agent_name="Model", success=True, output=recipe,
        message=(
            f"Selected **`{base_model}`** with **{approach}**.\n"
            f"{recipe.get('reasoning', '')}\n"
            f"Training for {recipe.get('num_epochs')} epochs, "
            f"lr={recipe.get('learning_rate')}, batch_size={recipe.get('batch_size')}."
        ),
        next_agent="Train",
    )
