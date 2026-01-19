# 🧪 Testing Guide - No-Code ML Training Platform

## Test Strategy Overview

### Testing Phases
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - API endpoint testing
3. **End-to-End Tests** - Full workflow testing
4. **Performance Tests** - Load and stress testing
5. **User Acceptance Tests** - Real-world scenarios

---

## 🔬 Phase 1: Unit Tests

### Backend Unit Tests

Create `backend/tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Test basic health endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_candidates():
    """Test model recommendation endpoint"""
    response = client.get("/model-candidates?task=classification")
    assert response.status_code == 200
    data = response.json()
    assert "transformers" in data or "finetuned" in data

def test_training_list():
    """Test listing training runs"""
    response = client.get("/api/training/list")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**Run Backend Tests:**
```bash
cd backend
pytest tests/ -v
```

### Frontend Unit Tests

Create `nlp-finetune-ui/src/components/__tests__/GlassCard.test.tsx`:

```typescript
import { render, screen } from '@testing-library/react';
import { GlassCard } from '../GlassCard';

describe('GlassCard', () => {
  it('renders children correctly', () => {
    render(<GlassCard>Test Content</GlassCard>);
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('applies correct variant classes', () => {
    const { container } = render(
      <GlassCard variant="strong">Content</GlassCard>
    );
    expect(container.firstChild).toHaveClass('backdrop-blur-xl');
  });
});
```

**Run Frontend Tests:**
```bash
cd nlp-finetune-ui
npm test
```

---

## 🔗 Phase 2: Integration Tests

### API Integration Test Script

Create `backend/tests/test_integration.py`:

```python
import pytest
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

class TestTrainingWorkflow:
    """Test complete training workflow"""
    
    def test_01_upload_dataset(self):
        """Test dataset upload"""
        # Create test CSV
        test_csv = Path("test_data.csv")
        test_csv.write_text("text,label\nHello,1\nWorld,0\n")
        
        with open(test_csv, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/validate-dataset",
                files={"file": f}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("valid") == True
        
        test_csv.unlink()
    
    def test_02_get_model_recommendations(self):
        """Test model recommendations"""
        response = requests.get(
            f"{BASE_URL}/model-candidates",
            params={"task": "classification"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("transformers", [])) > 0
    
    def test_03_start_training(self):
        """Test starting training job"""
        payload = {
            "model": "distilbert-base-uncased",
            "dataset_path": "uploads/test_data.csv",
            "text_col": "text",
            "label_col": "label",
            "num_labels": 2,
            "epochs": 1,
            "batch_size": 2
        }
        
        response = requests.post(f"{BASE_URL}/train", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        
        # Store run_id for next tests
        self.run_id = data["run_id"]
    
    def test_04_check_training_status(self):
        """Test training status endpoint"""
        if not hasattr(self, 'run_id'):
            pytest.skip("No run_id from previous test")
        
        time.sleep(5)  # Wait for training to start
        
        response = requests.get(
            f"{BASE_URL}/api/training/{self.run_id}/status"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_05_pause_training(self):
        """Test pause functionality"""
        if not hasattr(self, 'run_id'):
            pytest.skip("No run_id from previous test")
        
        response = requests.post(
            f"{BASE_URL}/api/training/{self.run_id}/pause"
        )
        
        # May fail if training already completed
        assert response.status_code in [200, 400]
    
    def test_06_list_training_runs(self):
        """Test listing all runs"""
        response = requests.get(f"{BASE_URL}/api/training/list")
        
        assert response.status_code == 200
        runs = response.json()
        assert isinstance(runs, list)
        assert len(runs) > 0
```

**Run Integration Tests:**
```bash
# Start backend first
cd backend
uvicorn app:app --reload &

# Run tests
pytest tests/test_integration.py -v -s

# Stop backend
pkill -f uvicorn
```

---

## 🎭 Phase 3: End-to-End Tests

### Manual E2E Test Checklist

#### Test Case 1: Complete Training Workflow
```
✓ Step 1: Open application (http://localhost:5173)
✓ Step 2: Click "New Project"
✓ Step 3: Upload CSV dataset
✓ Step 4: Verify dataset validation passes
✓ Step 5: Select task type (Classification)
✓ Step 6: Choose recommended model (BERT)
✓ Step 7: Configure hyperparameters
✓ Step 8: Start training
✓ Step 9: Verify real-time metrics appear
✓ Step 10: Test pause button
✓ Step 11: Test resume button
✓ Step 12: Wait for completion
✓ Step 13: Download trained model
✓ Step 14: Generate training report
```

#### Test Case 2: AI Assistant Interaction
```
✓ Step 1: Start training
✓ Step 2: Wait for 5 epochs
✓ Step 3: Check AI assistant panel
✓ Step 4: Verify analysis appears
✓ Step 5: Apply suggested parameter change
✓ Step 6: Verify training continues with new params
```

#### Test Case 3: Hyperparameter Optimization
```
✓ Step 1: Open hyperparameter optimizer
✓ Step 2: Select predefined search space
✓ Step 3: Configure max trials
✓ Step 4: Start optimization
✓ Step 5: Monitor progress
✓ Step 6: Verify best config found
✓ Step 7: Apply best config to new training
```

### Automated E2E Tests (Playwright)

Create `nlp-finetune-ui/e2e/training.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test.describe('Training Workflow', () => {
  test('should complete full training cycle', async ({ page }) => {
    // Navigate to app
    await page.goto('http://localhost:5173');
    
    // Click new project
    await page.click('text=New Project');
    
    // Upload dataset
    const fileInput = await page.locator('input[type="file"]');
    await fileInput.setInputFiles('test_data.csv');
    
    // Wait for validation
    await expect(page.locator('text=Valid')).toBeVisible();
    
    // Select model
    await page.click('text=BERT Base');
    
    // Start training
    await page.click('text=Start Training');
    
    // Verify training started
    await expect(page.locator('text=Training')).toBeVisible();
    
    // Check metrics appear
    await expect(page.locator('text=Train Loss')).toBeVisible();
  });
});
```

**Run E2E Tests:**
```bash
cd nlp-finetune-ui
npx playwright test
```

---

## ⚡ Phase 4: Performance Tests

### Load Testing Script

Create `backend/tests/load_test.py`:

```python
import asyncio
import aiohttp
import time

async def make_request(session, url):
    """Make single API request"""
    async with session.get(url) as response:
        return await response.json()

async def load_test(num_requests=100):
    """Simulate concurrent requests"""
    url = "http://localhost:8000/model-candidates?task=classification"
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, url) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {num_requests} requests in {duration:.2f}s")
    print(f"Average: {duration/num_requests:.3f}s per request")
    print(f"Throughput: {num_requests/duration:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(load_test(100))
```

**Run Load Test:**
```bash
python backend/tests/load_test.py
```

**Expected Results:**
- Average response time: < 100ms
- Throughput: > 50 req/s
- No errors or timeouts

---

## 📊 Test Results Documentation

### Create Test Report Template

```markdown
# Test Report - [Date]

## Summary
- Total Tests: X
- Passed: Y
- Failed: Z
- Success Rate: Y/X %

## Unit Tests
- Backend: ✓ All passed
- Frontend: ✓ All passed

## Integration Tests
- API Endpoints: ✓ All passed
- WebSocket: ✓ Connected
- Database: ✓ Operational

## E2E Tests
- Training Workflow: ✓ Passed
- AI Assistant: ✓ Passed
- Export Features: ✓ Passed

## Performance Tests
- Load Test (100 concurrent): ✓ Passed
- Average Response Time: 85ms
- Throughput: 67 req/s

## Issues Found
1. [Issue description]
   - Severity: High/Medium/Low
   - Status: Fixed/In Progress/Open

## Recommendations
- [Recommendation 1]
- [Recommendation 2]
```

---

## 🐛 Bug Tracking

### Bug Report Template

```markdown
**Bug ID:** BUG-001
**Title:** [Short description]
**Severity:** Critical/High/Medium/Low
**Status:** Open/In Progress/Fixed/Closed

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Screenshots:**
[If applicable]

**Environment:**
- OS: macOS/Windows/Linux
- Browser: Chrome/Firefox/Safari
- Python: 3.x
- Node: 16.x

**Logs:**
```
[Paste relevant logs]
```

**Fix:**
[Description of fix if resolved]
```

---

## ✅ Pre-Deployment Checklist

Before deploying to Vercel:

### Code Quality
- [ ] All tests passing
- [ ] No console errors
- [ ] No TypeScript errors
- [ ] Code linted and formatted

### Functionality
- [ ] Dataset upload works
- [ ] Training starts successfully
- [ ] Real-time updates working
- [ ] Pause/resume functional
- [ ] Export works correctly
- [ ] Reports generate properly

### Performance
- [ ] Page load < 3 seconds
- [ ] API response < 200ms
- [ ] No memory leaks
- [ ] Efficient resource usage

### Security
- [ ] API keys not exposed
- [ ] CORS configured properly
- [ ] Input validation working
- [ ] Error messages don't leak info

### Documentation
- [ ] README updated
- [ ] API docs complete
- [ ] Setup guide tested
- [ ] Deployment guide ready

---

## 🚀 Next Steps

After all tests pass:
1. ✅ Fix any critical bugs
2. ✅ Optimize performance bottlenecks
3. ✅ Update documentation
4. 🚀 Proceed to deployment phase
