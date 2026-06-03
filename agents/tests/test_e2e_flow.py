"""
End-to-end flow tests.

Covers:
  • RUNS_DIR path consistency — ml_core and the backend must agree on where
    artifacts are written so inference/export work after training.
  • Full training → artifact present → inference loadable (skipped when
    torch/transformers are not installed).
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the monorepo root (parent of the 'agents' package directory)."""
    return Path(__file__).parent.parent.parent


def _ml_core_runs_dir() -> Path:
    """Resolve the RUNS_DIR default used by ml_core (no env override)."""
    old = os.environ.pop("RUNS_DIR", None)
    try:
        # Force a fresh import so we pick up the real default, not a cached value.
        if "agents.ml_core" in sys.modules:
            del sys.modules["agents.ml_core"]
        from agents.ml_core import RUNS_DIR
        return RUNS_DIR.resolve()
    finally:
        if old is not None:
            os.environ["RUNS_DIR"] = old
        # Restore module so subsequent imports work normally.
        if "agents.ml_core" in sys.modules:
            del sys.modules["agents.ml_core"]


def _backend_runs_dir() -> Path:
    """
    Resolve the RUNS_DIR default used by backend/main.py.
    We compute it here the same way main.py does rather than importing the
    whole FastAPI app (which has side-effects).
    """
    backend_main = _project_root() / "backend" / "main.py"
    # main.py line 41: RUNS_DIR = Path(os.getenv("RUNS_DIR", str(
    #     Path(__file__).parent.parent / "agents" / "runs")))
    return (backend_main.parent.parent / "agents" / "runs").resolve()


def _tiny_csv(tmp_path: Path) -> Path:
    """Write a 30-row balanced 3-class CSV for a quick training smoke-test."""
    path = tmp_path / "tiny.csv"
    labels = ["cat", "dog", "bird"]
    rows = [
        {"text": f"This is a sample sentence number {i} about various topics.", "label": labels[i % 3]}
        for i in range(30)
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── RUNS_DIR consistency ──────────────────────────────────────────────────────

class TestRunsDirConsistency:
    def test_ml_core_and_backend_share_runs_dir(self):
        """
        The single most critical structural invariant: ml_core.RUNS_DIR and
        backend/main.py RUNS_DIR must resolve to the same directory so that
        artifact_path values produced during training are valid for inference
        and export.
        """
        ml_core_dir  = _ml_core_runs_dir()
        backend_dir  = _backend_runs_dir()
        assert ml_core_dir == backend_dir, (
            f"RUNS_DIR mismatch — ml_core defaults to {ml_core_dir!r} "
            f"but backend defaults to {backend_dir!r}. "
            "Inference and export will always fail after training until this is fixed."
        )

    def test_runs_dir_env_override_respected(self, tmp_path: Path):
        """Setting RUNS_DIR env var overrides both defaults."""
        custom = str(tmp_path / "custom_runs")
        os.environ["RUNS_DIR"] = custom

        # Clear cached module to pick up new env var.
        sys.modules.pop("agents.ml_core", None)
        try:
            from agents.ml_core import RUNS_DIR
            assert RUNS_DIR.resolve() == Path(custom).resolve()
        finally:
            del os.environ["RUNS_DIR"]
            sys.modules.pop("agents.ml_core", None)

    def test_runs_dir_is_inside_project(self):
        """RUNS_DIR should be a subdirectory of the project root, not some random path."""
        root = _project_root().resolve()
        runs = _ml_core_runs_dir()
        assert str(runs).startswith(str(root)), (
            f"RUNS_DIR {runs!r} is outside the project root {root!r}"
        )


# ── Training smoke test ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("sklearn") is not None
        and importlib.util.find_spec("datasets") is not None
    ),
    reason="ML libraries (torch/transformers/sklearn/datasets) not installed",
)
class TestTrainingArtifactPath:
    @pytest.mark.asyncio
    async def test_artifact_saved_in_runs_dir(self, tmp_path: Path):
        """
        Training a tiny model saves artifacts inside RUNS_DIR.
        The artifact path returned by train_model_async must:
          1. Exist on disk after training completes.
          2. Be a subdirectory of RUNS_DIR (so the backend's security validator
             will accept it for inference and export).
        """
        from agents.ml_core import train_model_async, RUNS_DIR, has_training_libs
        assert has_training_libs(), "Guard: libs were found by find_spec but not importable"

        csv_path = _tiny_csv(tmp_path)
        job_id = "test_e2e_artifact"

        result = await train_model_async(
            job_id=job_id,
            model_id="distilbert-base-uncased",
            dataset_path=str(csv_path),
            text_col="text",
            label_col="label",
            task_type="text_classification",
            training_approach="full_finetune",
            learning_rate=5e-5,
            num_epochs=1,
            batch_size=8,
            max_length=32,
        )

        artifact = Path(result.model_path)

        # Artifact must exist
        assert artifact.exists(), f"Artifact not found at {artifact}"

        # Artifact must be inside RUNS_DIR (enforced by backend security validator)
        assert str(artifact.resolve()).startswith(str(RUNS_DIR.resolve())), (
            f"Artifact {artifact!r} is outside RUNS_DIR {RUNS_DIR!r} — "
            "inference/export endpoints will reject it."
        )

        # Tokenizer config must be present (minimum viable saved model)
        assert (artifact / "tokenizer_config.json").exists() or \
               (artifact / "config.json").exists(), \
               "No tokenizer_config.json or config.json found in artifact directory"

    @pytest.mark.asyncio
    async def test_validation_rejects_missing_file(self):
        """validate_training_inputs returns an error for a non-existent dataset."""
        from agents.ml_core import validate_training_inputs

        result = validate_training_inputs(
            dataset_path="/nonexistent/path/data.csv",
            task_type="text_classification",
            text_col="text",
            label_col="label",
            model_id="distilbert-base-uncased",
        )
        assert not result.ok
        assert "not found" in result.error.lower() or "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validation_rejects_too_few_rows(self, tmp_path: Path):
        """Datasets with fewer than the minimum row count fail validation."""
        path = tmp_path / "tiny.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "hello", "label": "a"})

        from agents.ml_core import validate_training_inputs

        result = validate_training_inputs(
            dataset_path=str(path),
            task_type="text_classification",
            text_col="text",
            label_col="label",
            model_id="distilbert-base-uncased",
        )
        # Fewer than min rows → either error or warning; at minimum should flag it
        flagged = (not result.ok) or bool(result.warnings)
        assert flagged, "Expected validation to flag a 1-row dataset"
