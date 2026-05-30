"""
Model Agent — selects base model + either a fixed recipe or an HPO search space.

Architecture (B1 — LLM warm-start + Optuna/TPE):
  SMALL DATASETS (<200 rows) or user overrides:
    → LLM outputs a ModelRecipe directly (same as before)
  LARGE DATASETS (≥200 rows), gradient-based approach, no overrides:
    → LLM outputs an HPOSearchSpace with LR range + LoRA-r choices
    → TrainAgent runs Optuna/TPE within that space (3-5 quick trials)
    → Best params used for full training

Research basis: LLMLingua + AgentHPO + SLLMBO show the LLM+TPE hybrid
beats pure-LLM HPO and matches Bayesian optimization on 9/14 tabular tasks.
The win comes from hybridization: LLM supplies warm-start priors from domain
knowledge; TPE supplies the actual exploration.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import BaseAgent, AgentContext, AgentResult, SONNET
from .model_catalog import catalog_summary_for_prompt
from .schemas import ModelRecipe, HPOSearchSpace
from .validators import validate_recipe_semantics
from .cache import recipe_cache
from .memory import episodic_memory, format_memory_exemplar

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────

_HPO_MIN_ROWS = 200          # below this, HPO overhead isn't worth it
_HPO_APPROACHES = {"lora", "qlora", "full_finetune"}  # embed_classify = no gradient HPO

# ── System prompts (both CACHED — large static context) ────────────────────

_RECIPE_SYSTEM = """You are the Model Agent for ModelForge. Select the best base model
and a fixed training recipe for this ML task.

=== AVAILABLE MODELS ===
{catalog}

=== TRAINING PRIORS (research-backed defaults) ===
LoRA rank: 8-16 for classification (sweet spot: 16); 32 for generation.
LoRA alpha: set to 2× lora_r.
Learning rate: 1e-4 to 3e-4 for LoRA; 2e-5 to 5e-5 for full fine-tune.
Batch size: 16 or 32. Epochs: 2-3 for >500 rows; 5 for <200 rows.
QLoRA (4-bit) is the memory-efficient default for LLMs.

=== INSTRUCTIONS ===
• Use ONLY model IDs from the catalog above.
• Prefer small/fast models for small datasets (<1 000 rows) or speed priority.
• Prefer roberta-base or deberta-v3-small for accuracy priority.

Output ONLY valid JSON — no markdown, no commentary:
{{
  "base_model": "<model_id from catalog>",
  "training_approach": "full_finetune"|"lora"|"qlora"|"embed_classify",
  "lora_r": <int or null>,
  "lora_alpha": <int or null>,
  "lora_target_modules": null,
  "learning_rate": <float in [1e-7, 1e-2]>,
  "num_epochs": <int 1-20>,
  "batch_size": <int, power of 2>,
  "max_length": <int 16-512>,
  "warmup_ratio": <float 0.0-0.5>,
  "weight_decay": <float 0.0-0.5>,
  "reasoning": "<1-2 sentences>"
}}"""

_HPO_SYSTEM = """You are the Model Agent for ModelForge. This dataset is large enough for
hyperparameter optimization. Instead of fixed values, output a SEARCH SPACE that
Optuna/TPE will explore in 3-5 quick one-epoch trials.

=== AVAILABLE MODELS ===
{catalog}

=== HOW TO SET SEARCH SPACE RANGES ===
Learning rate range guidance (log-scale search):
  LoRA/QLoRA on encoders (bert, distilbert, roberta):   lr_min=5e-5, lr_max=5e-4
  LoRA/QLoRA on large encoders (deberta, electra):       lr_min=1e-5, lr_max=2e-4
  Full fine-tune:                                        lr_min=1e-5, lr_max=5e-5
  Keep lr_max / lr_min ratio ≥ 5× to give Optuna room to search.

LoRA rank choices: [8, 16] for speed; [8, 16, 32] for quality.
n_trials: 3 for datasets 200-500 rows; 5 for 500+ rows (max budget: 5 min GPU).
epochs_per_trial: always 1 (proxy metric — fast).

num_epochs (in the outer object): the full-training epochs AFTER HPO finds best params.
batch_size, max_length, warmup_ratio, weight_decay: fixed values, not searched.

=== INSTRUCTIONS ===
• Use ONLY model IDs from the catalog above.
• Set ranges to reflect genuine uncertainty — don't make them trivially narrow.

Output ONLY valid JSON — no markdown, no commentary:
{{
  "base_model": "<model_id from catalog>",
  "training_approach": "lora"|"qlora"|"full_finetune",
  "hpo_config": {{
    "lr_min": <float>,
    "lr_max": <float>,
    "lora_r_choices": [8, 16] or [8, 16, 32],
    "n_trials": <int 3-5>,
    "epochs_per_trial": 1
  }},
  "batch_size": <int, power of 2>,
  "max_length": <int 16-512>,
  "warmup_ratio": <float>,
  "weight_decay": <float>,
  "num_epochs": <int — final training epochs>,
  "reasoning": "<1-2 sentences>"
}}"""


class ModelAgent(BaseAgent):
    name  = "Model"
    model = SONNET

    async def run(self, context: AgentContext) -> AgentResult:
        ovr     = context.hyperparameter_overrides
        profile = context.data_profile
        num_rows = int(profile.get("num_rows", 0))

        # ── Fast path: user has pinned a model ────────────────────────────────
        if ovr.get("model_id"):
            recipe = _override_recipe(ovr, context)
            validated, err = self._parse_llm_json(
                __import__("json").dumps(recipe), ModelRecipe
            )
            if validated:
                context.model_recipe = validated.model_dump()
            else:
                logger.warning("Override recipe validation issue: %s", err)
                context.model_recipe = recipe
            val_result = validate_recipe_semantics(context.model_recipe, profile)
            if not val_result.is_valid:
                return AgentResult(
                    agent_name=self.name, success=False, output={"validation_errors": val_result.errors},
                    message=(
                        "Recipe has semantic errors that would cause training to fail:\n"
                        + "\n".join(f"• {e}" for e in val_result.errors)
                    ),
                )
            result = _recipe_result(context.model_recipe)
            if val_result.warnings:
                result.metadata["validation_warnings"] = val_result.warnings
            return result

        # ── Cache check: skip LLM if a quality recipe for similar dataset exists ──
        task_type = str(context.task_spec.get("task_type", "text_classification"))
        cached = recipe_cache.get(profile, task_type)
        if cached:
            context.model_recipe = cached.model_recipe
            result = _recipe_result(context.model_recipe)
            result.metadata["cache_hit"] = True
            result.metadata["cached_grade"] = cached.eval_grade
            logger.info("ModelAgent: cache HIT — skipping LLM call (cached grade %s)", cached.eval_grade)
            return result

        catalog = catalog_summary_for_prompt()
        import json

        # ── Episodic memory: inject exemplar if similar past run exists ────────
        memory_exemplar: str | None = None
        try:
            memories = episodic_memory.recall(profile, task_type)
            if memories:
                memory_exemplar = format_memory_exemplar(memories[0])
                logger.info("ModelAgent: memory exemplar injected (grade %s)", memories[0].eval_grade)
        except Exception as exc:
            logger.debug("ModelAgent: memory recall failed: %s", exc)

        # Compact profile: only the signals that drive model + hyperparameter selection.
        # estimated_tokens_p95 is the key new field — it tells the LLM what max_length
        # to set so 95% of samples are covered without excessive padding.
        compact_profile = {
            "num_rows":              num_rows,
            "num_classes":           profile.get("num_classes"),
            "avg_input_len":         profile.get("avg_input_len"),
            "avg_word_count":        profile.get("avg_word_count"),
            "estimated_tokens_avg":  profile.get("estimated_tokens_avg"),
            "estimated_tokens_p95":  profile.get("estimated_tokens_p95"),
            "vocabulary_richness":   profile.get("vocabulary_richness"),
            "text_quality_score":    profile.get("text_quality_score"),
            "class_balance_ratio":   profile.get("class_balance_ratio"),
            "label_noise_estimate":  profile.get("label_noise_estimate", 0.0),
            "issues":                profile.get("issues", []),
        }
        payload: dict[str, Any] = {
            "task_spec":    context.task_spec,
            "data_profile": compact_profile,
        }
        if memory_exemplar:
            payload["memory_exemplar"] = memory_exemplar
        user_msg = json.dumps(payload, indent=2)

        # ── Choose output mode ────────────────────────────────────────────────
        use_hpo = num_rows >= _HPO_MIN_ROWS

        if use_hpo:
            system = _HPO_SYSTEM.format(catalog=catalog)
            raw = await self._chat(
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                cache_system=True,
            )
            space, err = self._parse_llm_json(raw, HPOSearchSpace)

            if space is None:
                logger.warning("HPOSearchSpace parse failed (%s) — falling back to recipe mode", err)
                use_hpo = False   # fall through to recipe mode
            else:
                # Store search space in model_recipe with an "hpo" marker
                space_dict = space.model_dump()
                space_dict["_hpo_mode"] = True  # sentinel read by TrainAgent
                context.model_recipe = space_dict
                approach = space.training_approach.replace("_", " ").upper()
                cfg = space.hpo_config
                return AgentResult(
                    agent_name=self.name, success=True, output=space_dict,
                    message=(
                        f"Selected **`{space.base_model}`** with **{approach}**.\n"
                        f"{space.reasoning}\n"
                        f"HPO search: {cfg.n_trials} trials × {cfg.epochs_per_trial} epoch — "
                        f"lr ∈ [{cfg.lr_min:.0e}, {cfg.lr_max:.0e}], "
                        f"LoRA-r ∈ {cfg.lora_r_choices}."
                    ),
                    next_agent="Train",
                )

        # ── Recipe mode (small dataset or HPO fallback) ───────────────────────
        system = _RECIPE_SYSTEM.format(catalog=catalog)
        raw = await self._chat(
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            cache_system=True,
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
        context.model_recipe = recipe_model.model_dump()
        val_result = validate_recipe_semantics(context.model_recipe, profile)
        if not val_result.is_valid:
            return AgentResult(
                agent_name=self.name, success=False,
                output={"validation_errors": val_result.errors},
                message=(
                    "The selected recipe has semantic errors:\n"
                    + "\n".join(f"• {e}" for e in val_result.errors)
                    + "\nRetrying with a different configuration is recommended."
                ),
            )
        result = _recipe_result(context.model_recipe)
        if val_result.warnings:
            result.metadata["validation_warnings"] = val_result.warnings
        return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _override_recipe(ovr: dict, context: AgentContext) -> dict:
    rows     = context.data_profile.get("num_rows", 500)
    approach = ovr.get("training_approach", "full_finetune")
    lora_r   = ovr.get("lora_r", 16)
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


def _recipe_result(recipe: dict) -> AgentResult:
    approach   = str(recipe.get("training_approach", "unknown")).replace("_", " ").upper()
    base_model = recipe.get("base_model", "unknown model")
    lora_info  = ""
    if recipe.get("lora_r"):
        lora_info = f" (r={recipe['lora_r']}, α={recipe.get('lora_alpha', '?')})"
    lr = recipe.get("learning_rate", 0)
    return AgentResult(
        agent_name="Model", success=True, output=recipe,
        message=(
            f"Selected **`{base_model}`** with **{approach}{lora_info}**.\n"
            f"{recipe.get('reasoning', '')}\n"
            f"Training for {recipe.get('num_epochs')} epochs, "
            f"lr={lr:.2e}, batch={recipe.get('batch_size')}."
        ),
        next_agent="Train",
    )
