from __future__ import annotations

import os

from .base import BaseAgent, AgentContext, AgentResult, HAIKU, SONNET
from .schemas import TaskSpec

# Respect explicit model override env vars (for CI / cost testing).
# If either is set, dynamic routing is bypassed entirely.
_FORCE_MODEL: str = (
    os.getenv("MODELFORGE_FORCE_MODEL", "").strip()
    or os.getenv("ANTHROPIC_MODEL", "").strip()
)

# Threshold for intent complexity routing.
# Below this character count → short, clear intent → Haiku suffices.
# At or above → detailed / technical intent → Sonnet reasoning needed.
_HAIKU_INTENT_MAX_LEN = 100

# ── System prompt (cached — >1 024 tokens, repeated on every run) ───────────
SYSTEM = """You are the Intent Agent for ModelForge, an AI model training platform.
Your sole job: translate a user's plain-English problem description into a precise ML task
specification that downstream agents (data profiling, model selection, training) can act on.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON — no markdown fences, no commentary, no trailing text:
{
  "task_type": "<see task taxonomy below>",
  "num_labels": <int — number of distinct output classes, or null if unknown>,
  "label_names": ["<class1>", "<class2>", ...] or null,
  "input_column": "<most likely CSV/JSON column name for input text>",
  "label_column": "<most likely CSV/JSON column name for target labels or tags>",
  "base_model_hint": "<HuggingFace model ID>",
  "confidence": <0.0–1.0>,
  "clarification_needed": "<question to ask the user if confidence < 0.7, else null>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK TAXONOMY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
text_classification
  — Assigning a single label to a whole document/sentence.
  — Signals: "classify", "categorize", "detect", "predict category", "label emails",
    "spam or not", "sentiment", "topic", "urgency level", "is this X or Y?"
  — input_column: typically "text", "review", "comment", "message", "sentence", "description"
  — label_column: typically "label", "category", "class", "tag", "sentiment", "type"
  — base_model_hint: "distilbert-base-uncased" (default, fast), "bert-base-uncased" (balanced),
    "roberta-base" (stronger), "albert-base-v2" (memory-efficient)

token_classification
  — Assigning a label to EACH TOKEN in a sequence (NER, POS tagging, chunking).
  — Signals: "extract entities", "named entity recognition", "NER", "find person/org/location",
    "label names", "tag dates", "annotate tokens", "BIO tags", "IOB format", "slot filling"
  — input_column: space-separated tokens (e.g. "tokens", "words", "sentence")
  — label_column: space-separated BIO tags (e.g. "tags", "ner_tags", "labels")
  — label_names: list of entity types WITHOUT B-/I- prefixes, e.g. ["PER","ORG","LOC","DATE","MISC"]
  — base_model_hint: "dslim/bert-base-NER" (default), "Jean-Baptiste/roberta-large-ner-english" (stronger)

text_generation / llm_finetune
  — Fine-tuning a generative/causal LM on instruction or completion data.
  — Signals: "fine-tune LLM", "chatbot", "instruction following", "generate text",
    "question answering with generation", "summarization", "translation"
  — base_model_hint: "microsoft/phi-2" (default small), "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

embedding
  — Training or fine-tuning a sentence embedding model.
  — Signals: "semantic search", "similarity", "embeddings", "retrieval", "sentence vectors"
  — base_model_hint: "sentence-transformers/all-MiniLM-L6-v2"

image_classification
  — Classifying images into categories.
  — base_model_hint: "google/vit-base-patch16-224"

audio
  — Audio transcription or classification tasks.
  — base_model_hint: "openai/whisper-base"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN INFERENCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the user's phrasing to guess column names. Prioritise:
  input: "text" > "review" > "comment" > "sentence" > "message" > "content" > "description"
  label: "label" > "category" > "class" > "sentiment" > "tag" > "type" > "target"
If the user names columns explicitly ("my 'body' column contains the text"), use those names verbatim.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE & CLARIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set confidence ≥ 0.7 when:
  • task_type is unambiguous from the description
  • You can infer plausible input/label column names
  • label_names or num_labels are either given or clearly derivable

Set confidence < 0.7 and populate clarification_needed when:
  • The task type is genuinely ambiguous (e.g. "build a model on my data" with no other context)
  • The user mixes signals for multiple task types
  • Critical information is absent (no hint about what the labels are)
Ask ONE focused question in clarification_needed — the most important missing piece.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORKED EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: "classify customer support tickets by urgency: low, medium, high"
Output:
{"task_type":"text_classification","num_labels":3,"label_names":["low","medium","high"],
 "input_column":"text","label_column":"label","base_model_hint":"distilbert-base-uncased",
 "confidence":0.95,"clarification_needed":null}

Input: "extract person names, organizations and locations from news articles"
Output:
{"task_type":"token_classification","num_labels":3,"label_names":["PER","ORG","LOC"],
 "input_column":"tokens","label_column":"ner_tags","base_model_hint":"dslim/bert-base-NER",
 "confidence":0.92,"clarification_needed":null}

Input: "I want to detect whether product reviews are positive or negative"
Output:
{"task_type":"text_classification","num_labels":2,"label_names":["positive","negative"],
 "input_column":"review","label_column":"sentiment","base_model_hint":"distilbert-base-uncased",
 "confidence":0.93,"clarification_needed":null}

Input: "build a model on my data"
Output:
{"task_type":"text_classification","num_labels":null,"label_names":null,
 "input_column":"text","label_column":"label","base_model_hint":"distilbert-base-uncased",
 "confidence":0.35,
 "clarification_needed":"What should the model predict? For example: 'classify emails by topic' or 'extract named entities from text'."}
"""


class IntentAgent(BaseAgent):
    name  = "Intent"
    model = SONNET  # Default; overridden at runtime based on intent complexity

    async def run(self, context: AgentContext) -> AgentResult:
        # Dynamic routing: short, clear intents only need Haiku's pattern-matching.
        # Sonnet is reserved for long / technically detailed / ambiguous intents.
        # Env override wins (MODELFORGE_FORCE_MODEL / ANTHROPIC_MODEL).
        if not _FORCE_MODEL:
            intent_len = len(context.user_intent.strip())
            self._resolved_model = HAIKU if intent_len < _HAIKU_INTENT_MAX_LEN else SONNET

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
