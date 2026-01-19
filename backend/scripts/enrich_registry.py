#!/usr/bin/env python3
"""
Enrich the Model Registry by fetching config.json for each candidate.
This extracts technical specs needed for parameter mapping.
"""
import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
# Paths
from core.config import DATA_DIR, DB_PATH
CANDIDATES_FILE = DATA_DIR / "candidates_list.json"

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            task TEXT,
            downloads INTEGER,
            likes INTEGER,
            architectures TEXT,  -- JSON array
            max_position_embeddings INTEGER,
            vocab_size INTEGER,
            hidden_size INTEGER,
            num_layers INTEGER,
            model_type TEXT
        )
    ''')
    conn.commit()
    return conn

def fetch_config(model_id: str) -> dict:
    """
    Fetch config.json for a model from HuggingFace.
    Returns parsed config dict or empty dict on failure.
    """
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=DATA_DIR / "hf_cache"
        )
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to fetch config for {model_id}: {e}")
        return {}

def enrich_candidates():
    """
    Loop through candidates and enrich with technical specs.
    """
    # Load candidates
    if not CANDIDATES_FILE.exists():
        logger.error(f"Candidates file not found: {CANDIDATES_FILE}")
        logger.info("Run scripts/bootstrap_registry.py first")
        return
    
    with open(CANDIDATES_FILE, 'r') as f:
        candidates = json.load(f)
    
    logger.info(f"Loaded {len(candidates)} candidates")
    
    # Initialize DB
    conn = init_db()
    c = conn.cursor()
    
    enriched_count = 0
    failed_count = 0
    
    for candidate in tqdm(candidates, desc="Enriching models"):
        model_id = candidate['model_id']
        
        # Check if already exists
        c.execute("SELECT 1 FROM models WHERE id = ?", (model_id,))
        if c.fetchone():
            logger.debug(f"Skipping {model_id} (already in DB)")
            continue
        
        # Fetch config
        config = fetch_config(model_id)
        
        if not config:
            failed_count += 1
            continue
        
        # Extract fields
        architectures = config.get('architectures', [])
        max_pos = config.get('max_position_embeddings', config.get('n_positions', None))
        vocab_size = config.get('vocab_size', None)
        hidden_size = config.get('hidden_size', None)
        num_layers = config.get('num_hidden_layers', config.get('num_layers', None))
        model_type = config.get('model_type', 'unknown')
        
        # Insert into DB
        c.execute('''
            INSERT INTO models (
                id, task, downloads, likes, architectures,
                max_position_embeddings, vocab_size, hidden_size, num_layers, model_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            candidate['task'],
            candidate['downloads'],
            candidate['likes'],
            json.dumps(architectures),
            max_pos,
            vocab_size,
            hidden_size,
            num_layers,
            model_type
        ))
        
        conn.commit()
        enriched_count += 1
    
    conn.close()
    
    logger.info(f"✅ Enriched {enriched_count} models")
    logger.info(f"⚠️  Failed to fetch {failed_count} configs")
    logger.info(f"Database saved to {DB_PATH}")

if __name__ == "__main__":
    enrich_candidates()
