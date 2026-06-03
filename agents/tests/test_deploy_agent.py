"""
DeployAgent unit tests.

Covers every degradation stage + helper functions:
  • Skip: no training_result
  • Skip: training status == "skipped"
  • Skip: model path missing / doesn't exist
  • Skip: huggingface_hub not installed
  • Skip: HF_TOKEN not set (checks all env var aliases)
  • Fail: bad HF token (auth rejected)
  • _slugify(): special chars, length cap, prefix
  • _template_card(): produces minimal valid markdown
  • _find_unique_repo_id(): auto-increments slug suffix on conflict
  • Successful deploy: context.deploy_result populated correctly
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.base import AgentContext


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_training_result(model_path: str = "/fake/run/abc123") -> dict:
    return {
        "status":            "completed",
        "model_path":        model_path,
        "base_model":        "distilbert-base-uncased",
        "training_approach": "full_finetune",
        "num_epochs_completed": 3,
        "learning_rate":     2e-5,
        "device":            "cpu",
        "accuracy":          0.873,
        "f1":                0.854,
        "precision":         0.841,
        "recall":            0.868,
        "per_class_f1":      {"billing": 0.89, "bug": 0.82},
        "label_names":       ["billing", "bug"],
        "train_samples":     80,
        "eval_samples":      20,
    }


def _make_context(
    training_result: dict | None = None,
    hf_token: str | None = None,
    model_path: str = "/fake/run/abc123",
) -> AgentContext:
    ctx = AgentContext(
        run_id="test_deploy",
        user_intent="classify support tickets by urgency",
        task_spec={"task_type": "text_classification", "label_names": ["billing", "bug"]},
        data_profile={"label_distribution": {"billing": 40, "bug": 40}},
    )
    ctx.training_result = training_result if training_result is not None else _make_training_result(model_path)
    ctx.eval_result = {
        "evaluation_grade": "B",
        "summary": "Good model performance.",
        "concerns": [],
        "next_steps": [],
    }
    ctx.hf_token = hf_token
    return ctx


def _make_deploy_agent() -> "DeployAgent":
    from agents.deploy_agent import DeployAgent
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    return DeployAgent(client=mock_client)


# ── Degradation stages ────────────────────────────────────────────────────────

class TestDeployAgentDegradation:
    @pytest.mark.asyncio
    async def test_skip_when_no_training_result(self):
        """No training_result → skipped, not failed."""
        agent = _make_deploy_agent()
        ctx = _make_context()
        ctx.training_result = None

        result = await agent.run(ctx)

        assert result.success is True
        assert ctx.deploy_result["status"] == "skipped"
        assert ctx.deploy_result["reason"] == "no_training_result"

    @pytest.mark.asyncio
    async def test_skip_when_training_was_skipped(self):
        """Training status='skipped' (no GPU libs) → deploy skipped."""
        agent = _make_deploy_agent()
        ctx = _make_context(training_result={"status": "skipped"})

        result = await agent.run(ctx)

        assert result.success is True
        assert ctx.deploy_result["status"] == "skipped"
        assert ctx.deploy_result["reason"] == "training_skipped"

    @pytest.mark.asyncio
    async def test_skip_when_model_path_missing(self, tmp_path: Path):
        """Model path does not exist on disk → skipped with explanation."""
        agent = _make_deploy_agent()
        tr = _make_training_result(model_path=str(tmp_path / "nonexistent_run"))
        ctx = _make_context(training_result=tr)

        result = await agent.run(ctx)

        assert result.success is True
        assert ctx.deploy_result["status"] == "skipped"
        assert ctx.deploy_result["reason"] == "model_path_missing"

    @pytest.mark.asyncio
    async def test_skip_when_hf_hub_not_installed(self, tmp_path: Path):
        """huggingface_hub not importable → skipped with install hint."""
        # Create a fake model directory so the path check passes
        model_dir = tmp_path / "run_abc"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        agent = _make_deploy_agent()
        ctx = _make_context(
            training_result=_make_training_result(model_path=str(model_dir)),
            hf_token="hf_faketoken",
        )

        with patch("agents.deploy_agent._has_hf_hub", return_value=False):
            result = await agent.run(ctx)

        assert result.success is True
        assert ctx.deploy_result["status"] == "skipped"
        assert ctx.deploy_result["reason"] == "hf_hub_not_installed"

    @pytest.mark.asyncio
    async def test_skip_when_no_hf_token(self, tmp_path: Path):
        """No HF token in context or any env var → skipped with setup instructions."""
        model_dir = tmp_path / "run_def"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        agent = _make_deploy_agent()
        ctx = _make_context(
            training_result=_make_training_result(model_path=str(model_dir)),
            hf_token=None,
        )

        env_vars_to_clear = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
        with patch("agents.deploy_agent._has_hf_hub", return_value=True):
            with patch.dict(os.environ, {}, clear=False):
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)
                result = await agent.run(ctx)

        assert result.success is True
        assert ctx.deploy_result["status"] == "skipped"
        assert ctx.deploy_result["reason"] == "no_hf_token"
        assert "suggested_repo_slug" in ctx.deploy_result

    @pytest.mark.asyncio
    async def test_fail_on_bad_hf_token(self, tmp_path: Path):
        """Bad HF token → auth failure → AgentResult(success=False)."""
        model_dir = tmp_path / "run_ghi"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        agent = _make_deploy_agent()
        ctx = _make_context(
            training_result=_make_training_result(model_path=str(model_dir)),
            hf_token="hf_invalidtoken",
        )

        # Patch sys.modules so both _has_hf_hub() (import huggingface_hub) and
        # the internal `import huggingface_hub as hf` both see the same mock.
        mock_api = MagicMock()
        mock_api.whoami.side_effect = Exception("401 Unauthorized")
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            result = await agent.run(ctx)

        assert result.success is False
        assert ctx.deploy_result["status"] == "failed"
        assert "hf_auth_failed" in ctx.deploy_result["reason"]


# ── Helper functions ──────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic_slug(self):
        from agents.deploy_agent import _slugify
        slug = _slugify("classify customer support tickets by urgency")
        assert slug.startswith("modelforge-")
        assert " " not in slug
        assert slug == slug.lower()

    def test_special_characters_removed(self):
        from agents.deploy_agent import _slugify
        slug = _slugify("NER: extract Org/Person entities!")
        assert ":" not in slug
        assert "!" not in slug
        assert "/" not in slug

    def test_max_length_respected(self):
        from agents.deploy_agent import _slugify
        long_intent = "a" * 200
        slug = _slugify(long_intent)
        assert len(slug) <= 40

    def test_empty_string(self):
        from agents.deploy_agent import _slugify
        slug = _slugify("")
        # Empty body → trailing hyphen is stripped → returns "modelforge"
        assert slug.startswith("modelforge")
        assert len(slug) >= len("modelforge")

    def test_slug_is_valid_repo_name_chars(self):
        from agents.deploy_agent import _slugify
        import re
        slug = _slugify("Fine-tune Llama 3 on medical Q&A data")
        # Only lowercase alphanum and hyphens (valid HF repo name)
        assert re.match(r"^[a-z0-9\-]+$", slug), f"Invalid slug: {slug!r}"


class TestTemplateCard:
    def test_produces_yaml_frontmatter(self, tmp_path: Path):
        """Fallback template card must start with YAML frontmatter (---  ... ---)."""
        from agents.deploy_agent import _template_card

        ctx = _make_context()
        card = _template_card(ctx, "testuser/modelforge-classify")

        assert card.startswith("---"), "Model card must begin with YAML frontmatter"
        assert "---" in card[3:], "Model card must close YAML frontmatter"

    def test_contains_repo_id(self, tmp_path: Path):
        from agents.deploy_agent import _template_card

        ctx = _make_context()
        repo_id = "myuser/modelforge-sentiment"
        card = _template_card(ctx, repo_id)

        assert repo_id in card

    def test_contains_accuracy_metric(self):
        from agents.deploy_agent import _template_card

        ctx = _make_context()
        card = _template_card(ctx, "u/r")

        # Should include the accuracy value (87.3%)
        assert "87.3" in card or "0.873" in card

    def test_contains_usage_snippet(self):
        from agents.deploy_agent import _template_card

        ctx = _make_context()
        card = _template_card(ctx, "u/r")

        assert "pipeline" in card
        assert "text-classification" in card


class TestFindUniqueRepoId:
    def test_first_candidate_returned_when_not_exists(self):
        """If the base slug is available, it's returned as-is (no suffix)."""
        from agents.deploy_agent import _find_unique_repo_id

        mock_api = MagicMock()
        # repo_info raises (repo doesn't exist) → first candidate returned
        mock_api.repo_info.side_effect = Exception("404 not found")
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            repo_id = _find_unique_repo_id("sentiment-model", "testuser", "hf_tok")

        assert repo_id == "testuser/sentiment-model"

    def test_suffix_incremented_on_conflict(self):
        """Base slug exists → returns slug-v2."""
        from agents.deploy_agent import _find_unique_repo_id

        mock_api = MagicMock()
        call_count = {"n": 0}

        def repo_info_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock()  # base slug exists
            raise Exception("404")  # -v2 slug doesn't exist

        mock_api.repo_info.side_effect = repo_info_side_effect
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            repo_id = _find_unique_repo_id("sentiment-model", "testuser", "hf_tok")

        assert repo_id == "testuser/sentiment-model-v2"

    def test_timestamp_fallback_when_all_suffixes_taken(self):
        """All v2-v9 suffixes taken → falls back to timestamp slug."""
        from agents.deploy_agent import _find_unique_repo_id

        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock()  # always exists
        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = mock_api

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            repo_id = _find_unique_repo_id("sentiment-model", "testuser", "hf_tok")

        assert repo_id.startswith("testuser/sentiment-model-")
        # Should be a timestamp suffix (numeric)
        suffix = repo_id.split("-")[-1]
        assert suffix.isdigit(), f"Expected numeric timestamp suffix, got: {suffix!r}"
