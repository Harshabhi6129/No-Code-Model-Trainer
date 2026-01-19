# ✅ Local Testing Checklist

## 🎯 Your Step-by-Step Testing Guide

Follow this checklist to test the platform locally. Check off each item as you complete it.

---

## Phase 1: Initial Setup (15 minutes)

### Step 1: Verify Prerequisites
- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Node.js 16+ installed (`node --version`)
- [ ] At least 8GB RAM available
- [ ] 10GB free disk space

### Step 2: Get API Keys
- [ ] Visit https://makersuite.google.com/app/apikey
- [ ] Create Google API key
- [ ] Copy key to clipboard
- [ ] (Optional) Get Weights & Biases key from https://wandb.ai

### Step 3: Quick Start
```bash
cd /Users/harshaabhinavkusampudi/Documents/Agent
./quick_start.sh
```

**Expected Output:**
- [ ] Backend starts on http://localhost:8000
- [ ] Frontend starts on http://localhost:5173
- [ ] No error messages in terminal
- [ ] Test dataset created

### Step 4: Add API Key
```bash
# Edit backend/.env file
nano backend/.env

# Add your Google API key:
GOOGLE_API_KEY=your_actual_key_here
```

- [ ] API key added to .env file
- [ ] Restart backend (Ctrl+C and run `./quick_start.sh` again)

---

## Phase 2: Basic Functionality Tests (20 minutes)

### Test 1: Backend Health Check
```bash
# In a new terminal
curl http://localhost:8000/
```

**Expected:** `{"status": "ok", "message": "Backend is running!"}`

- [ ] Backend responds correctly
- [ ] No errors in backend terminal

### Test 2: API Endpoints
```bash
./test_api.sh
```

**Expected:** All tests pass

- [ ] Health check passes
- [ ] Model candidates endpoint works
- [ ] Training list endpoint works
- [ ] Hyperparameter endpoint works

### Test 3: Frontend Loads
1. Open browser to http://localhost:5173
2. Check browser console (F12)

- [ ] Page loads without errors
- [ ] No console errors
- [ ] UI renders correctly
- [ ] Modern glassmorphic design visible

---

## Phase 3: Core Workflow Test (30 minutes)

### Test 4: Dataset Upload

1. **Navigate to application**
   - [ ] Open http://localhost:5173
   - [ ] See dashboard with "New Project" button

2. **Upload test dataset**
   - [ ] Click "New Project" or upload button
   - [ ] Select `backend/uploads/test_sentiment.csv`
   - [ ] Wait for validation

**Expected Results:**
- [ ] File uploads successfully
- [ ] Validation passes
- [ ] Dataset preview shows
- [ ] No errors displayed

### Test 5: Model Selection

1. **Select task type**
   - [ ] Choose "Text Classification"
   - [ ] Task card highlights when selected

2. **Choose model**
   - [ ] See recommended models appear
   - [ ] DistilBERT shows as option
   - [ ] Click to select DistilBERT

**Expected Results:**
- [ ] Models load and display
- [ ] Can select a model
- [ ] Model details visible
- [ ] "Continue" button appears

### Test 6: Training Configuration

1. **Configure training**
   - [ ] Set epochs to 2 (for quick test)
   - [ ] Set batch size to 4
   - [ ] Keep other defaults

2. **Start training**
   - [ ] Click "Start Training"
   - [ ] Training dashboard appears

**Expected Results:**
- [ ] Training starts without errors
- [ ] Redirected to training dashboard
- [ ] Run ID generated

---

## Phase 4: Real-time Features Test (15 minutes)

### Test 7: Real-time Metrics

**Watch for:**
- [ ] Loss curves update in real-time
- [ ] Accuracy metrics appear
- [ ] Progress bar moves
- [ ] Epoch counter increments
- [ ] Time elapsed updates

**Check WebSocket:**
- [ ] Open browser DevTools → Network tab
- [ ] Filter by "WS" (WebSocket)
- [ ] See active WebSocket connection
- [ ] Messages flowing

### Test 8: Training Controls

1. **Test Pause**
   - [ ] Click "Pause" button
   - [ ] Training pauses
   - [ ] Status changes to "Paused"
   - [ ] Metrics stop updating

2. **Test Resume**
   - [ ] Click "Resume" button
   - [ ] Training resumes
   - [ ] Status changes to "Running"
   - [ ] Metrics continue updating

3. **Test Parameter Adjustment**
   - [ ] Click settings/controls panel
   - [ ] Adjust learning rate slider
   - [ ] Click "Apply Changes"
   - [ ] See confirmation

**Expected Results:**
- [ ] All controls work smoothly
- [ ] No errors or crashes
- [ ] State persists correctly

### Test 9: AI Assistant

**Wait for 2-3 epochs, then check:**
- [ ] AI Assistant panel shows analysis
- [ ] Status indicator appears (Healthy/Warning)
- [ ] Suggestions provided
- [ ] Can apply suggestions

---

## Phase 5: Advanced Features Test (20 minutes)

### Test 10: Resource Monitoring

**Check resource panel:**
- [ ] CPU usage displays
- [ ] Memory usage displays
- [ ] GPU usage displays (if available)
- [ ] Values update in real-time
- [ ] Alerts show if usage > 85%

### Test 11: Training Completion

**Wait for training to complete:**
- [ ] Training finishes all epochs
- [ ] Status changes to "Completed"
- [ ] Final metrics displayed
- [ ] No errors occurred

### Test 12: Export & Reports

1. **Generate Report**
   - [ ] Click "Generate Report"
   - [ ] Report loads with charts
   - [ ] All sections populated
   - [ ] Can download report

2. **Export Model**
   - [ ] Click "Export Model"
   - [ ] Download starts
   - [ ] ZIP file downloads
   - [ ] File size reasonable (>1MB)

3. **Verify Export Contents**
   ```bash
   cd backend/exports
   unzip <run_id>_complete_package.zip
   ls -la
   ```
   
   - [ ] Model weights present
   - [ ] inference.py included
   - [ ] README.md included
   - [ ] requirements.txt included

---

## Phase 6: Edge Cases & Error Handling (15 minutes)

### Test 13: Invalid Dataset

1. **Create invalid CSV**
   ```bash
   echo "invalid,data" > backend/uploads/invalid.csv
   ```

2. **Try to upload**
   - [ ] Upload invalid.csv
   - [ ] Validation fails gracefully
   - [ ] Error message clear
   - [ ] Can try again

### Test 14: Stop Training

1. **Start new training**
2. **Click "Stop" button**
   - [ ] Confirmation dialog appears
   - [ ] Training stops permanently
   - [ ] Checkpoint saved
   - [ ] Can view partial results

### Test 15: Multiple Training Runs

1. **Start 2-3 training runs**
2. **Check training list**
   - [ ] All runs listed
   - [ ] Correct status for each
   - [ ] Can view each run
   - [ ] Can delete runs

---

## Phase 7: Performance & Stability (10 minutes)

### Test 16: Memory Leaks

1. **Monitor memory usage**
   ```bash
   # In new terminal
   watch -n 1 'ps aux | grep python'
   ```

2. **Run training for 5+ minutes**
   - [ ] Memory usage stable
   - [ ] No continuous growth
   - [ ] CPU usage reasonable

### Test 17: Browser Performance

1. **Open DevTools → Performance**
2. **Record during training**
   - [ ] FPS stays above 30
   - [ ] No long tasks (>50ms)
   - [ ] Smooth animations

### Test 18: Concurrent Operations

1. **Open 2 browser tabs**
2. **Start training in both**
   - [ ] Both work independently
   - [ ] No conflicts
   - [ ] WebSockets separate

---

## Phase 8: Final Validation (10 minutes)

### Test 19: Complete Workflow (End-to-End)

**Run through entire workflow one more time:**
1. [ ] Upload dataset
2. [ ] Select task
3. [ ] Choose model
4. [ ] Configure parameters
5. [ ] Start training
6. [ ] Monitor real-time
7. [ ] Test pause/resume
8. [ ] Wait for completion
9. [ ] Generate report
10. [ ] Export model
11. [ ] Verify export works

**Time the workflow:**
- [ ] Complete workflow takes < 10 minutes (for 2 epochs)
- [ ] No errors encountered
- [ ] All features work

### Test 20: Documentation Check

- [ ] README.md is clear
- [ ] SETUP_GUIDE.md is accurate
- [ ] All commands work as documented
- [ ] Screenshots match actual UI (if any)

---

## 📊 Test Results Summary

### Completion Status
- Total Tests: 20
- Passed: ___
- Failed: ___
- Skipped: ___

### Critical Issues Found
1. _______________________________
2. _______________________________
3. _______________________________

### Minor Issues Found
1. _______________________________
2. _______________________________
3. _______________________________

### Performance Notes
- Average training time (2 epochs): ___ minutes
- Memory usage: ___ GB
- CPU usage: ___ %
- Page load time: ___ seconds

---

## ✅ Ready for Deployment?

**All critical tests must pass before deployment:**

- [ ] All Phase 1-3 tests passed (Core functionality)
- [ ] No critical bugs found
- [ ] Performance acceptable
- [ ] Documentation accurate
- [ ] API keys configured
- [ ] Export functionality works

**If all checked, you're ready to proceed to deployment!**

---

## 🐛 Bug Report Template

If you find issues, document them here:

```markdown
**Bug #1**
- Severity: Critical/High/Medium/Low
- Component: Backend/Frontend/Both
- Steps to Reproduce:
  1. 
  2. 
  3. 
- Expected: 
- Actual: 
- Error Message: 
- Screenshot: 
```

---

## 📝 Notes & Observations

Use this space for any additional notes:

```
- 
- 
- 
```

---

## 🚀 Next Steps

After completing this checklist:

1. **If all tests pass:**
   - Review DEPLOYMENT_GUIDE.md
   - Prepare for production deployment
   - Set up monitoring

2. **If tests fail:**
   - Document all issues
   - Prioritize critical bugs
   - Fix and retest
   - Repeat until all pass

3. **Performance optimization:**
   - Identify bottlenecks
   - Optimize slow operations
   - Reduce memory usage if needed

---

**Good luck with testing! 🎉**