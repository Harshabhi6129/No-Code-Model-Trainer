"""
Deploy Agent — completes the end-to-end pipeline:
  1. Generates a professional HuggingFace model card with Claude
  2. Pushes the trained model to HuggingFace Hub (private by default)
  3. Produces ready-to-use Python + pipeline code snippets
  4. Gracefully degrades on every failure mode with actionable messages

Degradation chain (skips cleanly at each level):
  no training result → skip
  ML libs not installed → skip (training was skipped upstream)
  model path missing → explain, suggest re-run
  huggingface_hub not installed → skip with install instructions
  HF_TOKEN not set → skip with manual push command
  name conflict → auto-increment slug suffix (-v2 … -v9)
  connection/rate-limit → retry 2× with 3s backoff
  model card generation fails → structured template fallback
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from .base import BaseAgent, AgentContext, AgentResult

logger = logging.getLogger(__name__)

_MAX_SLUG_LEN = 40
_HUB_RETRIES  = 2
_HUB_BACKOFF  = 3   # seconds


# ---------------------------------------------------------------------------
# Model card system prompt
# ---------------------------------------------------------------------------

_CARD_SYSTEM = """You are writing a professional HuggingFace model card (README.md) for a
fine-tuned NLP model. Output ONLY valid Markdown — no extra commentary.

The card MUST follow this exact structure:

```
---
language: en
license: apache-2.0
tags:
  - text-classification
  - modelforge
  - <task_type>
datasets:
  - custom
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
---

# <Model Name>

<1-sentence description of what the model does.>

## Model Details

| Property | Value |
|----------|-------|
| Base model | ... |
| Training approach | ... |
| Epochs | ... |
| Learning rate | ... |
| Device | ... |

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | ...% |
| Weighted F1 | ... |
| Precision | ... |
| Recall | ... |

### Per-Class F1

| Class | F1 |
|-------|----|
| ... | ... |

## Training Data

- **File**: ...
- **Samples**: ... (... train / ... test)
- **Classes** (...): ...

## Usage

### With HuggingFace Pipeline

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="<repo_id>")
result = classifier("Your input text here")
print(result)
```

### Manual Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("<repo_id>")
model = AutoModelForSequenceClassification.from_pretrained("<repo_id>")

inputs = tokenizer("Your input text", return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class = model.config.id2label[logits.argmax().item()]
print(predicted_class)
```

## Evaluation Summary

<2-3 sentences from the evaluation agent. Grade + calibrated interpretation.>

## Limitations

- This model was fine-tuned on a specific dataset; performance on out-of-distribution text may vary.
- ...
- Not intended for high-stakes decisions without human review.

## Training Configuration

Trained with [ModelForge](https://github.com/Harshabhi6129/No-Code-Model-Trainer) — a no-code AI training platform.
```

Fill in all placeholders using the provided JSON context. Do NOT add any text before or after the Markdown.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = _MAX_SLUG_LEN) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    s = s.strip("-")
    prefix = "modelforge-"
    body = s[: max_len - len(prefix)]
    return f"{prefix}{body}".rstrip("-")


def _has_hf_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return False


def _model_size_gb(path: str) -> float:
    """Approximate size of all files under path in GB."""
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 ** 3)


async def _push_to_hub(
    *,
    model_path: str,
    repo_id: str,
    model_card: str,
    token: str,
) -> str:
    """Push model folder + model card to HF Hub. Returns the resolved repo_id."""
    import huggingface_hub as hf

    api = hf.HfApi(token=token)

    for attempt in range(_HUB_RETRIES + 1):
        try:
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            await asyncio.to_thread(
                api.upload_folder,
                folder_path=model_path,
                repo_id=repo_id,
                ignore_patterns=["*.py", "*.sh"],
            )
            await asyncio.to_thread(
                api.upload_file,
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
            )
            return repo_id
        except hf.utils.RepositoryNotFoundError:
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "409" in msg:
                raise ValueError(f"Repo '{repo_id}' already exists with different settings.") from exc
            if attempt < _HUB_RETRIES:
                logger.warning("[DeployAgent] Push attempt %d failed (%s), retrying…", attempt + 1, exc)
                await asyncio.sleep(_HUB_BACKOFF)
            else:
                raise


def _find_unique_repo_id(base_slug: str, username: str, token: str) -> str:
    """Return a repo_id not already taken, appending -v2 … -v9 as needed."""
    import huggingface_hub as hf
    api = hf.HfApi(token=token)

    for suffix in [""] + [f"-v{i}" for i in range(2, 10)]:
        candidate = f"{username}/{base_slug}{suffix}"
        try:
            api.repo_info(repo_id=candidate)
            # Repo exists — try next suffix
        except Exception:
            return candidate
    return f"{username}/{base_slug}-{int(time.time())}"


def _template_card(context: AgentContext, repo_id: str) -> str:
    """Minimal fallback model card when Claude is unavailable."""
    tr = context.training_result
    spec = context.task_spec
    task = spec.get("task_type", "text-classification").replace("_", "-")
    acc = tr.get("accuracy")
    f1  = tr.get("f1")
    acc_str = f"{acc * 100:.1f}%" if acc is not None else "—"
    f1_str  = f"{f1:.3f}" if f1 is not None else "—"

    return f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - modelforge
  - {task}
datasets:
  - custom
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
---

# {repo_id.split("/")[-1]}

Fine-tuned {tr.get("base_model", "transformer")} model for {task}.
Trained with [ModelForge](https://github.com/Harshabhi6129/No-Code-Model-Trainer).

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | {acc_str} |
| Weighted F1 | {f1_str} |

## Usage

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="{repo_id}")
print(classifier("Your input text here"))
```
"""


def _build_snippets(repo_id: str) -> dict[str, str]:
    return {
        "python": (
            f'from transformers import AutoTokenizer, AutoModelForSequenceClassification\n'
            f'import torch\n\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            f'model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")\n\n'
            f'inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=512)\n'
            f'with torch.no_grad():\n'
            f'    logits = model(**inputs).logits\n'
            f'label = model.config.id2label[logits.argmax().item()]\n'
            f'print(label)\n'
        ),
        "pipeline": (
            f'from transformers import pipeline\n\n'
            f'classifier = pipeline("text-classification", model="{repo_id}")\n'
            f'result = classifier("Your text here")\n'
            f'print(result)\n'
        ),
    }


# ---------------------------------------------------------------------------
# Deploy Agent
# ---------------------------------------------------------------------------

class DeployAgent(BaseAgent):
    name = "Deploy"

    async def run(self, context: AgentContext) -> AgentResult:
        tr = context.training_result
        spec = context.task_spec
        user_intent = context.user_intent or ""

        # ── 1. No training result ────────────────────────────────────────────
        if not tr:
            context.deploy_result = {"status": "skipped", "reason": "no_training_result"}
            return AgentResult(
                agent_name=self.name, success=True,
                output=context.deploy_result,
                message="No training result — deploy skipped. Run the full pipeline to train a model first.",
                next_agent=None,
            )

        # ── 2. Training was skipped (no GPU libs) ────────────────────────────
        if tr.get("status") == "skipped":
            context.deploy_result = {"status": "skipped", "reason": "training_skipped"}
            return AgentResult(
                agent_name=self.name, success=True,
                output=context.deploy_result,
                message=(
                    "Training was skipped (GPU libraries not available), so there is nothing to deploy.\n"
                    "Run locally with GPU support and the full pipeline will train + deploy automatically."
                ),
                next_agent=None,
            )

        model_path = tr.get("model_path", "")

        # ── 3. Model files missing ───────────────────────────────────────────
        if not model_path or not Path(model_path).exists():
            context.deploy_result = {"status": "skipped", "reason": "model_path_missing", "model_path": model_path}
            return AgentResult(
                agent_name=self.name, success=True,
                output=context.deploy_result,
                message=(
                    f"Trained model files not found at `{model_path or '(none)'}`. "
                    "This can happen if the server restarted after training. "
                    "Re-run the pipeline to train and deploy in one step."
                ),
                next_agent=None,
            )

        # ── 4. huggingface_hub not installed ─────────────────────────────────
        if not _has_hf_hub():
            context.deploy_result = {"status": "skipped", "reason": "hf_hub_not_installed"}
            return AgentResult(
                agent_name=self.name, success=True,
                output=context.deploy_result,
                message=(
                    "The `huggingface_hub` package is not installed in this environment. "
                    "To deploy manually:\n"
                    "```bash\npip install huggingface_hub\n"
                    f"huggingface-cli upload {model_path} my-org/my-model\n```"
                ),
                next_agent=None,
            )

        # ── 5. HF Token not set ──────────────────────────────────────────────
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not hf_token:
            slug = _slugify(user_intent or spec.get("task_type", "model") or "model")
            context.deploy_result = {"status": "skipped", "reason": "no_hf_token", "suggested_repo_slug": slug}
            return AgentResult(
                agent_name=self.name, success=True,
                output=context.deploy_result,
                message=(
                    "No HuggingFace token found (`HF_TOKEN` env var not set).\n\n"
                    "To deploy your model manually:\n"
                    "```bash\n"
                    "export HF_TOKEN=hf_your_token_here\n"
                    f"huggingface-cli upload {model_path} your-username/{slug}\n"
                    "```\n\n"
                    "Get your token at: https://huggingface.co/settings/tokens\n"
                    "Then set `HF_TOKEN` in your `.env` file and re-run to deploy automatically."
                ),
                next_agent=None,
            )

        # ── 6. Resolve HF username ───────────────────────────────────────────
        try:
            import huggingface_hub as hf
            user_info = hf.HfApi(token=hf_token).whoami()
            username = user_info.get("name") or user_info.get("id", "user")
        except Exception as exc:
            context.deploy_result = {"status": "failed", "reason": f"hf_auth_failed: {exc}"}
            return AgentResult(
                agent_name=self.name, success=False,
                output=context.deploy_result,
                message=(
                    f"HuggingFace authentication failed: {exc}\n"
                    "Check that your `HF_TOKEN` is valid and has write permissions."
                ),
                next_agent=None,
            )

        # ── 7. Generate model card (Claude) ──────────────────────────────────
        base_slug = _slugify(user_intent or spec.get("task_type", "model") or "model")
        repo_id = _find_unique_repo_id(base_slug, username, hf_token)
        model_card = await self._generate_model_card(context, repo_id)

        # ── 8. Warn about large models ───────────────────────────────────────
        warnings: list[str] = []
        try:
            size_gb = _model_size_gb(model_path)
            if size_gb > 2:
                warnings.append(
                    f"Model is large ({size_gb:.1f} GB) — push may take several minutes. "
                    "Large models may require Git LFS on the HF repo."
                )
        except Exception:
            pass

        # ── 9. Push to Hub ───────────────────────────────────────────────────
        try:
            resolved_id = await _push_to_hub(
                model_path=model_path,
                repo_id=repo_id,
                model_card=model_card,
                token=hf_token,
            )
        except Exception as exc:
            logger.error("DeployAgent: Hub push failed: %s", exc, exc_info=True)
            context.deploy_result = {
                "status": "failed",
                "reason": str(exc),
                "model_card": model_card,
                "usage_snippet": _build_snippets(repo_id),
            }
            return AgentResult(
                agent_name=self.name, success=False,
                output=context.deploy_result,
                message=(
                    f"HuggingFace Hub push failed: {exc}\n"
                    "Your trained model is still saved locally at "
                    f"`{model_path}` — you can push it manually:\n"
                    f"```bash\nhuggingface-cli upload {model_path} {repo_id}\n```"
                ),
                next_agent=None,
            )

        # ── 10. Success ──────────────────────────────────────────────────────
        hf_url = f"https://huggingface.co/{resolved_id}"
        snippets = _build_snippets(resolved_id)
        context.deploy_result = {
            "status": "deployed",
            "hf_repo_id": resolved_id,
            "hf_url": hf_url,
            "model_card": model_card,
            "usage_snippet": snippets,
            "warnings": warnings,
        }

        warning_note = "\n**Note:** " + " | ".join(warnings) if warnings else ""
        return AgentResult(
            agent_name=self.name, success=True,
            output=context.deploy_result,
            message=(
                f"Model deployed to HuggingFace Hub!\n"
                f"**Repository:** [{resolved_id}]({hf_url})\n"
                f"Quick start:\n```python\n{snippets['pipeline']}```"
                f"{warning_note}"
            ),
            next_agent=None,
        )

    # ------------------------------------------------------------------

    async def _generate_model_card(self, context: AgentContext, repo_id: str) -> str:
        """Ask Claude to write a professional model card. Falls back to template."""
        tr = context.training_result
        ev = context.eval_result
        spec = context.task_spec
        profile = context.data_profile

        per_class = tr.get("per_class_f1", {})
        label_names = tr.get("label_names", spec.get("label_names") or [])

        payload = json.dumps({
            "repo_id": repo_id,
            "user_intent": context.user_intent,
            "task_type": spec.get("task_type", "text_classification"),
            "model": {
                "base_model":        tr.get("base_model"),
                "training_approach": tr.get("training_approach"),
                "num_epochs":        tr.get("num_epochs_completed"),
                "learning_rate":     tr.get("learning_rate"),
                "device":            tr.get("device"),
            },
            "dataset": {
                "filename":    context.dataset_path and Path(context.dataset_path).name,
                "train_samples": tr.get("train_samples"),
                "eval_samples":  tr.get("eval_samples"),
                "num_classes":   tr.get("num_labels"),
                "label_names":   label_names,
                "label_distribution": profile.get("label_distribution", {}),
            },
            "metrics": {
                "accuracy":    tr.get("accuracy"),
                "f1":          tr.get("f1"),
                "precision":   tr.get("precision"),
                "recall":      tr.get("recall"),
                "per_class_f1": per_class,
            },
            "evaluation": {
                "grade":      ev.get("evaluation_grade"),
                "summary":    ev.get("summary"),
                "concerns":   ev.get("concerns", []),
                "next_steps": ev.get("next_steps", []),
            },
        }, indent=2)

        try:
            card = await self._chat(
                system=_CARD_SYSTEM,
                messages=[{"role": "user", "content": payload}],
            )
            # Verify it looks like a model card
            if "---" in card and "#" in card:
                return card
            logger.warning("DeployAgent: model card output looked wrong, using fallback")
        except Exception as exc:
            logger.warning("DeployAgent: model card generation failed (%s), using fallback", exc)

        return _template_card(context, repo_id)
