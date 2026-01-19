#!/usr/bin/env python3
"""
Build vector embeddings index for all models in the registry.
This enables fast semantic search for model recommendations.
"""
import os
import sys
import sqlite3
import pickle
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "model_registry.db"
EMBEDDINGS_FILE = DATA_DIR / "model_embeddings.pkl"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_searchable_string(model_data: dict) -> str:
    """
    Construct a searchable string from model metadata.
    """
    parts = [
        model_data.get('id', ''),
        model_data.get('model_type', ''),
        model_data.get('task', ''),
        f"architecture: {model_data.get('architectures', '')}",
    ]
    return " ".join([p for p in parts if p])

def build_index():
    """
    Build vector embeddings for all models in the registry.
    """
    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Connect to database
    if not DB_PATH.exists():
        logger.error(f"Model registry database not found at {DB_PATH}")
        logger.info("Please run scripts/bootstrap_registry.py and scripts/enrich_registry.py first")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Fetch all models
    c.execute("SELECT id, task, model_type, architectures FROM models")
    rows = c.fetchall()
    
    if not rows:
        logger.error("No models found in database")
        return
    
    logger.info(f"Found {len(rows)} models in database")
    
    # Build searchable strings and embeddings
    model_ids = []
    searchable_texts = []
    
    for row in rows:
        model_data = {
            'id': row[0],
            'task': row[1],
            'model_type': row[2],
            'architectures': row[3]
        }
        
        model_ids.append(row[0])
        searchable_texts.append(build_searchable_string(model_data))
    
    logger.info("Generating embeddings...")
    embeddings = embedder.encode(searchable_texts, show_progress_bar=True)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Save to pickle file
    index_data = {
        'model_ids': model_ids,
        'embeddings': embeddings_array,
        'metadata': {
            'model': EMBEDDING_MODEL,
            'total_models': len(model_ids),
            'embedding_dim': embeddings_array.shape[1]
        }
    }
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    
    logger.info(f"✅ Saved embeddings to {EMBEDDINGS_FILE}")
    logger.info(f"Index stats: {len(model_ids)} models, {embeddings_array.shape[1]} dimensions")
    
    conn.close()

if __name__ == "__main__":
    build_index()
