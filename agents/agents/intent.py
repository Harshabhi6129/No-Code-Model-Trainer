from __future__ import annotations
import json
from .base import BaseAgent, AgentContext, AgentResult

SYSTEM = """You are the Intent Agent for ModelForge, an AI model training platform.
Your job: translate a user's plain-English description into a precise ML task specification.

Output ONLY valid JSON with this schema:
{
  "task_type": "text_classification" | "token_classification" | "text_generation" | "llm_finetune" | "embedding" | "image_classification" | "audio",
  "num_labels": <int or null>,
  "label_names": [<string>, ...] or null,
  "input_column": "<likely CSV column name for input text>",
  "label_column": "<likely CSV column name for labels>",
  "base_model_hint": "<suggested HuggingFace model ID>",
  "confidence": <0.0 to 1.0>,
  "clarification_needed": "<question to ask user if confidence < 0.7, else null>"
}"""


class IntentAgent(BaseAgent):
    name = "Intent"

    async def run(self, context: AgentContext) -> AgentResult:
        raw = await self._chat(system=SYSTEM, messages=[{"role": "user", "content": context.user_intent}])
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError:
            return AgentResult(agent_name=self.name, success=False, output={},
                               message="Could not parse task specification. Please rephrase your request.")

        if not isinstance(spec, dict):
            return AgentResult(agent_name=self.name, success=False, output={},
                               message="Task specification was not a JSON object. Please rephrase your request.")

        context.task_spec = spec
        needs_clarification = spec.get("confidence", 1.0) < 0.7

        # Use .get() with fallbacks so missing keys never raise KeyError
        task_type = str(spec.get("task_type") or "unspecified").replace("_", " ")
        base_model = spec.get("base_model_hint") or "a suitable base model"

        return AgentResult(
            agent_name=self.name, success=True, output=spec,
            message=spec.get("clarification_needed") or
                    f"Got it — this is a **{task_type}** task. "
                    f"I'll use `{base_model}` as the base model.",
            next_agent=None if needs_clarification else "Data",
        )
