#!/usr/bin/env python3
"""
Test script for training control functionality
"""
import asyncio
import json
import time
from pathlib import Path
import requests

BASE_URL = "http://localhost:8000"

def test_training_control():
    """Test the training control endpoints"""
    
    # 1. Start a dummy training job
    print("🚀 Starting training job...")
    
    # Create a simple test payload
    payload = {
        "model": "distilbert-base-uncased",
        "dataset_path": "uploads/sample.csv",  # Assuming you have a sample CSV
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "learning_rate": 5e-5,
        "batch_size": 16,
        "epochs": 5,
        "weight_decay": 0.01
    }
    
    try:
        response = requests.post(f"{BASE_URL}/train", json=payload)
        if response.status_code == 200:
            run_id = response.json()["run_id"]
            print(f"✅ Training started with run_id: {run_id}")
        else:
            print(f"❌ Failed to start training: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        return
    
    # 2. Wait a bit for training to start
    print("⏳ Waiting for training to start...")
    time.sleep(10)
    
    # 3. Check training status
    print("📊 Checking training status...")
    try:
        response = requests.get(f"{BASE_URL}/api/training/{run_id}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Training status: {status}")
        else:
            print(f"❌ Failed to get status: {response.text}")
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    # 4. Test pause functionality
    print("⏸️ Testing pause functionality...")
    try:
        response = requests.post(f"{BASE_URL}/api/training/{run_id}/pause")
        if response.status_code == 200:
            print(f"✅ Training paused: {response.json()}")
        else:
            print(f"❌ Failed to pause: {response.text}")
    except Exception as e:
        print(f"❌ Error pausing: {e}")
    
    # 5. Wait and check status
    time.sleep(5)
    try:
        response = requests.get(f"{BASE_URL}/api/training/{run_id}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Status after pause: {status}")
        else:
            print(f"❌ Failed to get status: {response.text}")
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    # 6. Test parameter update
    print("🔧 Testing parameter update...")
    try:
        new_params = {
            "learning_rate": 1e-5,
            "weight_decay": 0.02
        }
        response = requests.post(f"{BASE_URL}/api/training/{run_id}/update-params", json=new_params)
        if response.status_code == 200:
            print(f"✅ Parameters updated: {response.json()}")
        else:
            print(f"❌ Failed to update params: {response.text}")
    except Exception as e:
        print(f"❌ Error updating params: {e}")
    
    # 7. Test resume functionality
    print("▶️ Testing resume functionality...")
    try:
        response = requests.post(f"{BASE_URL}/api/training/{run_id}/resume")
        if response.status_code == 200:
            print(f"✅ Training resumed: {response.json()}")
        else:
            print(f"❌ Failed to resume: {response.text}")
    except Exception as e:
        print(f"❌ Error resuming: {e}")
    
    # 8. Wait a bit then stop
    print("⏳ Waiting before stopping...")
    time.sleep(10)
    
    # 9. Test stop functionality
    print("🛑 Testing stop functionality...")
    try:
        response = requests.post(f"{BASE_URL}/api/training/{run_id}/stop")
        if response.status_code == 200:
            print(f"✅ Training stopped: {response.json()}")
        else:
            print(f"❌ Failed to stop: {response.text}")
    except Exception as e:
        print(f"❌ Error stopping: {e}")
    
    # 10. Final status check
    time.sleep(2)
    try:
        response = requests.get(f"{BASE_URL}/api/training/{run_id}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Final status: {status}")
        else:
            print(f"❌ Failed to get final status: {response.text}")
    except Exception as e:
        print(f"❌ Error getting final status: {e}")

def create_sample_dataset():
    """Create a sample dataset for testing"""
    sample_data = """text,label
"This is a positive example",1
"This is a negative example",0
"Another positive text",1
"Another negative text",0
"Great product, highly recommend",1
"Terrible service, avoid at all costs",0
"Amazing experience, will come back",1
"Poor quality, waste of money",0
"""
    
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    sample_file = uploads_dir / "sample.csv"
    sample_file.write_text(sample_data)
    print(f"✅ Created sample dataset: {sample_file}")

if __name__ == "__main__":
    print("🧪 Training Control Test Suite")
    print("=" * 50)
    
    # Create sample dataset
    create_sample_dataset()
    
    # Run tests
    test_training_control()
    
    print("=" * 50)
    print("🏁 Test completed!")