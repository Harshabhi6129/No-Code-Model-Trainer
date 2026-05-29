from __future__ import annotations

from .base import BaseAgent, AgentContext, AgentResult, SONNET
from .schemas import TaskSpec

# ── System prompt (cached — ~300 tokens, repeated on every run) ─────────────
# Note: below the 1 024-token cache minimum; cache_system still accepted without error.
# The EvalAgent and ModelAgent prompts exceed the minimum and benefit most from caching.

SYSTEM = """You are the Intent Agent for ModelForge, an AI model training platform.
Your job: translate a user's plain-English description into a precise ML task specification.

Output ONLY valid JSON — no markdown fences, no commentary:
{
  "task_type": "text_classification" | "token_classification" | "text_generation" | "llm_finetune" | "embedding" | "image_classification" | "audio",
  "num_labels": <int or null>,
  "label_names": [<string>, ...] or null,
  "input_column": "<most likely CSV column name for input text, e.g. 'text', 'review', 'sentence'>",
  "label_column": "<most likely CSV column name for labels, e.g. 'label', 'category', 'sentiment'>",
  "base_model_hint": "<HuggingFace model ID — prefer small/fast models unless user asks otherwise>",
  "confidence": <0.0 to 1.0>,
  "clarification_needed": "<question to ask user if confidence < 0.7, else null>"
}

Model hints by task:
  text_classification → "distilbert-base-uncased" (default), "bert-base-uncased", "roberta-base"
  token_classification → "distilbert-base-uncased", "bert-base-uncased"
  text_generation / llm_finetune → "microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  embedding → "sentence-transformers/all-MiniLM-L6-v2"
  image_classification → "google/vit-base-patch16-224"
"""


class IntentAgent(BaseAgent):
    name  = "Intent"
    model = SONNET  # Needs real NL reasoning to parse ambiguous user descriptions

    async def run(self, context: AgentContext) -> AgentResult:
        raw = await self._chat(
            system=SYSTEM,
            messages=[{"role": "user", "content": context.user_intent}],
            cache_system=True,
        )

        spec, err = self._parse_llm_json(raw, TaskSpec)
        if spec is None:
            return AgentResult(
                agent_name=self.name, success=False, output={},
                message=(
                    "Could not parse your task description into a machine learning specification. "
                    "Please rephrase — for example: 'classify customer support emails by urgency'."
                ),
            )

        # Store as plain dict (AgentContext uses dicts throughout)
        spec_dict = spec.model_dump()
        context.task_spec = spec_dict

        needs_clarification = spec.confidence < 0.7
        task_label = spec.task_type.replace("_", " ")
        model_hint = spec.base_model_hint

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=spec_dict,
            message=(
                spec.clarification_needed
                or f"Got it — this is a **{task_label}** task. "
                   f"I'll use `{model_hint}` as the base model."
            ),
            next_agent=None if needs_clarification else "Data",
        )
