#!/usr/bin/env python3
"""
Bootstrap the Model Registry using HuggingFace API (no CSV required).
Fetches top models directly from the Hub for guaranteed quality.
"""
import json
import logging
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
CANDIDATES_FILE = DATA_DIR / "candidates_list.json"

# Configuration
TARGET_TASKS = ["text-classification", "text-generation", "image-classification"]
MODELS_PER_TASK = 50

def bootstrap_from_hub():
    """
    Fetch top models from HuggingFace Hub using the API.
    This ensures we only get valid, popular models.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    api = HfApi()
    all_candidates = []
    
    logger.info("Fetching top models from HuggingFace Hub...")
    
    for task in TARGET_TASKS:
        logger.info(f"Fetching top {MODELS_PER_TASK} models for task: {task}")
        
        try:
            # Fetch models with task filter (use task string directly)
            models = list(api.list_models(
                filter=task,
                sort="downloads",
                direction=-1,
                limit=MODELS_PER_TASK
            ))
            
            task_candidates = []
            for model in models:
                # Skip private models
                if hasattr(model, 'private') and model.private:
                    continue
                
                # Extract metadata
                model_id = model.id if hasattr(model, 'id') else model.modelId
                downloads = model.downloads if hasattr(model, 'downloads') else 0
                likes = model.likes if hasattr(model, 'likes') else 0
                
                task_candidates.append({
                    "model_id": model_id,
                    "downloads": downloads,
                    "likes": likes,
                    "task": task
                })
            
            logger.info(f"  Found {len(task_candidates)} models for {task}")
            all_candidates.extend(task_candidates)
            
        except Exception as e:
            logger.error(f"Failed to fetch models for {task}: {e}")
            continue
    
    # Save to JSON
    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(all_candidates, f, indent=2)
    
    logger.info(f"✅ Saved {len(all_candidates)} candidates to {CANDIDATES_FILE}")
    logger.info(f"Breakdown by task:")
    
    # Count models per task
    task_counts = {}
    for candidate in all_candidates:
        task = candidate['task']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in task_counts.items():
        logger.info(f"  {task}: {count} models")
    
    logger.info(f"\nNext step: Run scripts/enrich_registry.py to fetch configs")

if __name__ == "__main__":
    bootstrap_from_hub()
