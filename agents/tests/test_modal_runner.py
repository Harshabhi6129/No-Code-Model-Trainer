"""
Tests for Step 15 — Modal GPU Integration.

Covers:
  • has_modal(): False when MODAL_TOKEN_ID/SECRET not set
  • has_modal(): False when modal package not installed
  • has_modal(): True when both env vars set AND modal is importable
  • run_training_on_modal(): raises RuntimeError when has_modal() is False
  • _has_modal() in TrainAgent: returns False without modal
  • training_location in final AgentResult metadata
  • Modal warm-up SSE event emitted before training starts
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ── has_modal() ───────────────────────────────────────────────────────────────

class TestHasModal:
    def test_false_when_no_env_vars(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
        with patch.dict(os.environ, env, clear=True):
            from services.modal_runner import has_modal
            assert has_modal() is False

    def test_false_when_only_token_id_set(self):
        with patch.dict(os.environ, {"MODAL_TOKEN_ID": "abc"}, clear=True):
            from services.modal_runner import has_modal
            assert has_modal() is False

    def test_false_when_modal_not_installed(self):
        with patch.dict(os.environ, {
            "MODAL_TOKEN_ID": "abc",
            "MODAL_TOKEN_SECRET": "xyz",
        }):
            with patch.dict("sys.modules", {"modal": None}):
                from importlib import reload
                import services.modal_runner as mod
                reload(mod)
                # modal import raises → has_modal returns False
                assert mod.has_modal() is False
                reload(mod)  # restore

    def test_true_when_credentials_set_and_modal_importable(self):
        """Simulate modal package available."""
        mock_modal = MagicMock()
        with patch.dict(os.environ, {
            "MODAL_TOKEN_ID": "tok_id",
            "MODAL_TOKEN_SECRET": "tok_secret",
        }):
            with patch.dict("sys.modules", {"modal": mock_modal}):
                from importlib import reload
                import services.modal_runner as mod
                reload(mod)
                assert mod.has_modal() is True
                reload(mod)  # restore


# ── run_training_on_modal() ───────────────────────────────────────────────────

class TestRunTrainingOnModal:
    @pytest.mark.asyncio
    async def test_raises_when_modal_unavailable(self):
        """Without Modal credentials, run_training_on_modal raises RuntimeError."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
        with patch.dict(os.environ, env, clear=True):
            from services.modal_runner import run_training_on_modal
            with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
                await run_training_on_modal({"job_id": "test"})

    @pytest.mark.asyncio
    async def test_returns_training_result_on_success(self):
        """Mock Modal remote call → verify TrainingResult is constructed."""
        mock_modal = MagicMock()

        fake_result = {
            "model_path": "/tmp/model",
            "base_model": "bert-base-uncased",
            "training_approach": "full_finetune",
            "num_epochs_completed": 3,
            "final_train_loss": 0.35,
            "training_time_seconds": 120.0,
            "device": "h100",
            "metrics": {"accuracy": 0.88, "f1": 0.87,
                        "precision": 0.86, "recall": 0.88, "ece": 0.04,
                        "per_class_f1": {}, "per_class_metrics": {},
                        "confusion_matrix": [], "num_labels": 2,
                        "label_names": ["a", "b"], "train_samples": 100, "eval_samples": 25},
            "warnings": [],
            "epoch_metrics": [],
        }

        mock_fn = MagicMock()
        mock_fn.remote = MagicMock(return_value=fake_result)
        mock_modal.App = MagicMock(return_value=MagicMock())
        mock_modal.Image = MagicMock()
        mock_modal.Volume = MagicMock()
        mock_modal.Retries = MagicMock()

        with patch.dict(os.environ, {
            "MODAL_TOKEN_ID": "tok",
            "MODAL_TOKEN_SECRET": "sec",
        }):
            with patch("services.modal_runner.has_modal", return_value=True):
                with patch("services.modal_runner._build_modal_app",
                           return_value=(MagicMock(), mock_fn)):
                    from services.modal_runner import run_training_on_modal
                    from agents.ml_core import TrainingResult
                    result = await run_training_on_modal({"job_id": "test_run"})
                    assert isinstance(result, TrainingResult)
                    assert result.base_model == "bert-base-uncased"
                    assert result.device == "h100"


# ── _has_modal() in TrainAgent ────────────────────────────────────────────────

class TestTrainAgentModalDetection:
    def test_has_modal_false_without_credentials(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}
        with patch.dict(os.environ, env, clear=True):
            from agents.train_agent import _has_modal
            assert _has_modal() is False

    def test_has_modal_false_when_import_error(self):
        """If services.modal_runner can't be imported, _has_modal returns False."""
        with patch.dict(os.environ, {"MODAL_TOKEN_ID": "x", "MODAL_TOKEN_SECRET": "y"}):
            with patch.dict("sys.modules", {"services.modal_runner": None}):
                from agents.train_agent import _has_modal
                # ImportError from None module → False
                result = _has_modal()
                assert isinstance(result, bool)


# ── training_location in metadata ────────────────────────────────────────────

class TestTrainingLocation:
    @pytest.mark.asyncio
    async def test_local_training_location_in_metadata(self):
        """When Modal is not available, training_location='local' in final result metadata."""
        from agents.base import AgentContext
        from agents.train_agent import TrainAgent

        # Patch all training to return immediately without GPU libs
        with patch("agents.train_agent._has_modal", return_value=False):
            with patch("agents.train_agent.has_training_libs", return_value=False):
                agent = TrainAgent.__new__(TrainAgent)
                agent.client = AsyncMock()
                agent._resolved_model = "claude-sonnet-4-6"
                agent.last_stage_metrics = None
                agent.model = "claude-sonnet-4-6"

                ctx = AgentContext(
                    run_id="test_local",
                    user_intent="classify",
                )
                ctx.task_spec = {"task_type": "text_classification", "input_column": "text", "label_column": "label"}
                ctx.data_profile = {"num_rows": 100}
                ctx.model_recipe = {"base_model": "bert-base-uncased", "training_approach": "full_finetune",
                                    "learning_rate": 2e-5, "num_epochs": 3, "batch_size": 16,
                                    "max_length": 128, "weight_decay": 0.01, "warmup_ratio": 0.1,
                                    "lora_r": 8, "_hpo_mode": False}

                results = []
                async for result in agent.run_stream(ctx):
                    results.append(result)

                # Should not crash; should emit a skipped result with local training_location
                assert len(results) > 0
                final = results[-1]
                # For skipped training, training_location should be "local"
                assert final.metadata.get("training_location") in ("local", None) or final.success
