import os
import sys
import sqlite3
import logging
from huggingface_hub import list_models
from tqdm import tqdm

# Add backend to path to import llm_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.llm_service import enrich_model_metadata
from core.config import DB_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TASKS = [
    "text-classification",
    "token-classification",
    "text-generation",
    "summarization",
    "translation"
]

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT,
            task TEXT,
            downloads INTEGER,
            likes INTEGER,
            description TEXT,
            pros TEXT, -- JSON string
            cons TEXT, -- JSON string
            best_for TEXT,
            architecture TEXT,
            max_seq_length INTEGER
        )
    ''')
    conn.commit()
    return conn

def fetch_and_process_models():
    conn = init_db()
    c = conn.cursor()
    
    total_models = 0
    
    for task in TASKS:
        logger.info(f"Fetching models for task: {task}")
        
        # Fetch top 20 models per task to keep it manageable for now (User said 500 total, but let's start small for testing)
        # We can increase this later.
        models = list_models(
            filter=task,
            sort="downloads",
            direction=-1,
            limit=2,
            cardData=True
        )
        
        for model in tqdm(models, desc=f"Processing {task}"):
            model_id = model.modelId
            
            # Skip if already exists
            c.execute("SELECT 1 FROM models WHERE id = ?", (model_id,))
            if c.fetchone():
                continue
                
            # Extract metadata
            downloads = model.downloads
            likes = model.likes
            card_data = model.cardData or {}
            description = model.description or "" # This might be empty if not fetched fully, but list_models with cardData helps
            
            # Get config if possible (simulated for now or extracted from tags)
            # In a real full implementation, we might fetch config.json
            # For now, we rely on tags and basic info
            
            # Enrich with Gemini
            # We use a truncated description to save tokens
            enrichment = enrich_model_metadata(model_id, description)
            
            # Insert into DB
            import json
            c.execute('''
                INSERT INTO models (id, name, task, downloads, likes, description, pros, cons, best_for, architecture, max_seq_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id,
                model_id.split('/')[-1],
                task,
                downloads,
                likes,
                description,
                json.dumps(enrichment.get('pros', [])),
                json.dumps(enrichment.get('cons', [])),
                enrichment.get('best_for', 'General purpose'),
                "Transformer", # Placeholder, would need config.json fetch to be accurate
                512 # Placeholder
            ))
            
            conn.commit()
            total_models += 1
            
    logger.info(f"Registry build complete. Added {total_models} new models.")
    conn.close()

if __name__ == "__main__":
    fetch_and_process_models()
