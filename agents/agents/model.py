from __future__ import annotations
import json
from .base import BaseAgent, AgentContext, AgentResult

SYSTEM = """You are the Model Agent for ModelForge. Select the best base model and training recipe
given the task spec and data profile. Consider: dataset size, task type, compute budget, and inference latency.

Output ONLY valid JSON:
{
  "base_model": "<HuggingFace model ID>",
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
  "reasoning": "<1-2 sentence explanation of choices>"
}"""

_REQUIRED = {"base_model", "training_approach", "num_epochs", "learning_rate", "batch_size"}


class ModelAgent(BaseAgent):
    name = "Model"

    async def run(self, context: AgentContext) -> AgentResult:
        prompt = json.dumps({"task_spec": context.task_spec, "data_profile": context.data_profile}, indent=2)
        raw = await self._chat(system=SYSTEM, messages=[{"role": "user", "content": prompt}])
        try:
            recipe = json.loads(raw)
        except json.JSONDecodeError:
            return AgentResult(agent_name=self.name, success=False, output={},
                               message="Could not determine training recipe. Please try again.")

        if not isinstance(recipe, dict):
            return AgentResult(agent_name=self.name, success=False, output={},
                               message="Model recipe was not a JSON object. Please try again.")

        missing = _REQUIRED - recipe.keys()
        if missing:
            return AgentResult(agent_name=self.name, success=False, output=recipe,
                               message=f"Model recipe missing required fields: {', '.join(missing)}. Please try again.")

        context.model_recipe = recipe
        approach = str(recipe.get("training_approach", "unknown")).replace("_", " ").upper()
        base_model = recipe.get("base_model", "unknown model")

        return AgentResult(
            agent_name=self.name, success=True, output=recipe,
            message=(
                f"Selected **`{base_model}`** with **{approach}**.\n"
                f"{recipe.get('reasoning', '')}\n"
                f"Training for {recipe.get('num_epochs')} epochs, "
                f"lr={recipe.get('learning_rate')}, batch_size={recipe.get('batch_size')}."
            ),
            next_agent="Train",
        )
