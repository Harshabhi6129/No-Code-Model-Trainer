import os
from pathlib import Path

# Robust way to find the backend root
# This file is in backend/core/config.py
# So parent is core, parent.parent is backend
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"

# Verify/Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Database Path
DB_PATH = DATA_DIR / "model_registry.db"
CANDIDATES_PATH = DATA_DIR / "candidates_list.json"
