# backend/exporter.py
import os
import zipfile
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"


def create_zip(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "model").exists():
        raise FileNotFoundError("Run not found or training failed.")

    zip_path = RUNS_DIR / f"{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(run_dir))

    return zip_path


def export_run(run_id: str) -> Path:
    """Alias for create_zip."""
    return create_zip(run_id)
