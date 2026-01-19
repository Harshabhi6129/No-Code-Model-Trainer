#!/usr/bin/env python3
"""Simple training test to debug issues"""
import sys
from pathlib import Path

# Test basic imports
try:
    from training_runner import start_training
    print("✓ Training runner imported")
except Exception as e:
    print(f"✗ Failed to import training_runner: {e}")
    sys.exit(1)

# Test starting training
try:
    config = {
        "model": "distilbert-base-uncased",
        "dataset_path": "uploads/test_sentiment.csv",
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 5e-5
    }
    
    run_id = "test-run-123"
    print(f"Starting training with run_id: {run_id}")
    start_training(run_id, config)
    
    import time
    time.sleep(5)
    
    # Check if run directory was created
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        print(f"✓ Run directory created: {run_dir}")
        print(f"  Contents: {list(run_dir.iterdir())}")
    else:
        print(f"✗ Run directory NOT created: {run_dir}")
        
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
