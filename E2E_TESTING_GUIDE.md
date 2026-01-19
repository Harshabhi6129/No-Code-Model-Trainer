# 🎬 Complete End-to-End Testing Guide
## Movie Reviews Sentiment Analysis Platform

---

## 📋 Test Overview

**Dataset**: Movie Reviews Sentiment Analysis  
**File**: `movie_reviews_sentiment.csv` (300 samples, evenly balanced)  
**Task**: Binary sentiment classification (positive/negative)  
**Location**: `/Users/harshaabhinavkusampudi/Documents/Agent/movie_reviews_sentiment.csv`

---

## 🚀 Step-by-Step Testing Instructions

### Step 1: Verify Servers Are Running ✅

**Backend** (Terminal 1):
```bash
cd /Users/harshaabhinavkusampudi/Documents/Agent/backend
uvicorn app:app --reload --port 8000
```

**Expected Output**:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Frontend** (Terminal 2):
```bash
cd /Users/harshaabhinavkusampudi/Documents/Agent/nlp-finetune-ui
npm run dev
```

**Expected Output**:
```
VITE v6.3.5  ready in XXXms
➜  Local:   http://localhost:5174/
```

---

### Step 2: Open the Application 🌐

1. **Navigate to**: `http://localhost:5174`
2. **You should see**: The main app interface with dataset upload section

---

### Step 3: Upload Dataset 📁

1. **Click**: "Choose File" or file upload button
2. **Select**: `/Users/harshaabhinavkusampudi/Documents/Agent/movie_reviews_sentiment.csv`
3. **Click**: "Upload" or "Analyze Dataset"

**Expected Result**:
```json
{
  "detected_task": "text-classification",
  "summary": "Movie review sentiment analysis",
  "columns": ["review", "sentiment"],
  "num_samples": 300,
  "label_distribution": {
    "positive": 150,
    "negative": 150
  }
}
```

**What to Look For**:
- ✅ Success message appears
- ✅ Dataset analysis shown (columns, sample count, balance)
- ✅ Gemini detected task: "Text Classification" or "Sentiment Analysis"
- ✅ Intent summary appears

---

### Step 4: Model Recommendation 🤖

#### 4a. Enter User Intent

**In the Intent Box**, enter:
```
Classify movie reviews as positive or negative based on sentiment
```

Or simply:
```
movie review sentiment analysis
```

#### 4b. Click "Find Best Model"

**Expected Behavior**:
- Loading spinner appears
- AI analyzes your intent
- **Backend logs** show:
  ```
  INFO: POST /api/recommend
  Semantic search: top 5 candidates
  Gemini ranking...
  ```

#### 4c. View Recommendations

**You should see**:
- **5 candidate models** in a leaderboard
- **Gold "⭐ AI TOP PICK" badge** on #1 model
- Each card shows:
  - Model ID (e.g., `distilbert-base-uncased-finetuned-sst-2-english`)
  - AI reasoning (1-2 sentences)
  - 📊 Downloads, 💖 Likes, 🎯 Match %
  
**Example Card**:
```
⭐ AI TOP PICK  #1

distilbert-base-uncased-finetuned-sst-2-english

Perfect for binary sentiment classification. Pre-trained on SST-2, 
which is a movie review dataset, making it ideal for this task.

📊 15,234,567 downloads  💖 1,234 likes  🎯 95% match
```

---

### Step 5: Select Model & Configure Parameters ⚙️

#### 5a. Click Any Model Card

- **Selected model** gets purple gradient border + ✓ checkmark
- **Parameter Playground** (right column) updates instantly

#### 5b. Adjust Hyperparameters

**You should see sliders/dropdowns for**:
- **Learning Rate**: `2e-5` (slider: 1e-6 to 1e-3)
- **Num Epochs**: `3` (dropdown: 1, 2, 3, 5, 10)
- **Batch Size**: `8` (dropdown: 4, 8, 16, 32)
- **Weight Decay**: `0.01` (slider: 0.0 to 0.1)

**Try adjusting**:
- Learning Rate → `5e-5`
- Num Epochs → `2`
- Batch Size → `16`

---

### Step 6: Start Training 🚂

#### 6a. Click "Start Training" Button

**Expected**:
- View transitions to **Training Dashboard**
- WebSocket connection establishes

**Backend Logs**:
```
✅ WebSocket connected for client train_167...
Loading model...
Loading dataset...
Auto-detected columns - Text: review, Label: sentiment
Dataset loaded: 300 samples, 2 labels
Starting trainer.train() for job train_167...
```

---

### Step 7: Monitor Live Training 📊

**Training Dashboard Components**:

#### A. Header
- `← Back to Model Selection` button (top left)
- `[●] Connected` status (green)
- Model ID displayed

#### B. Metric Cards (4 boxes)
1. **Current Epoch**: `0/2` → `1/2` → `2/2`
2. **Learning Rate**: `5.0e-5`
3. **Elapsed Time**: `0:15` → `0:45` → `1:30`
4. **Current Loss**: `0.6932` → `0.3421` → `0.1245`

#### C. Charts (2 graphs)
1. **Training Loss**: Line chart showing loss decreasing
2. **Learning Rate**: Area chart (flat or with warmup/decay)

#### D. Status Panel
- Status: `Initializing...` → `Loading model...` → `Training...` → `Complete!`

#### E. Training Logs
```
[12:34:01] Loading model...
[12:34:05] Loading dataset...
[12:34:07] Auto-detected columns - Text: review, Label: sentiment
[12:34:08] Dataset loaded: 300 samples, 2 labels
[12:34:10] Initializing model with 2 classes...
[12:34:15] Training started...
[12:34:20] Loss: 0.6932
[12:34:35] Loss: 0.4521
[12:34:50] Completed epoch 1
[12:35:05] Loss: 0.3214
[12:35:20] Loss: 0.2156
[12:35:35] Completed epoch 2
[12:35:40] ✅ Model saved to: /path/to/outputs/train_167.../final_model
```

---

### Step 8: Check Training Results ✅

**When Training Completes**:

1. **Status Panel** shows:
   - `Training Complete!` message
   - Green checkmark ✓
   - "Download Trained Model" button

2. **Final Metrics**:
   - Final loss: ~0.1-0.3 (should decrease from ~0.7)
   - Total time: 1-3 minutes (with 300 samples, 2 epochs)

3. **Backend Logs**:
   ```
   Training job train_167... completed successfully
   Model saved to: /path/to/outputs/train.../final_model
   ```

---

### Step 9: Navigation Testing 🔙

#### Test the Back Button

1. **Click** `← Back to Model Selection`
2. **If training active**: See confirmation dialog
   ```
   Training is still in progress. Are you sure you want to go back?
   This will not stop the training job.
   ```
3. **After completion**: Instant return to model selection
4. **You should see**: Parameter playground again with same selections

---

## 🐛 Troubleshooting

### Issue 1: "No models found for task type: Text Classification"

**Cause**: Task type mismatch  
**Fix**: Already fixed! Backend now normalizes task types

### Issue 2: "Stuck on Initializing..."

**Possible Causes**:
1. Dataset path not found
2. Column auto-detection failed
3. Model download error

**Check Backend Logs**:
```bash
# Look for error messages
tail -f backend/backend.log
```

**Solutions**:
- Verify CSV path is correct
- Check CSV has headers: `review,sentiment`
- Ensure internet connection (for model download)

### Issue 3: WebSocket Not Connecting

**Symptoms**: Dashboard shows `Connecting...` forever

**Check**:
1. Backend is running on port 8000
2. No firewall blocking WebSocket
3. Browser console for errors (F12)

**Restart**:
```bash
# Kill and restart backend
# Terminal 1
Ctrl+C
uvicorn app:app --reload --port 8000
```

### Issue 4: Training Error

**Check Logs** for:
```
❌ Error: Training failed: [error message]
```

**Common Errors**:
- `Out of memory` → Reduce batch_size to 4
- `Column not found` → CSV format issue
- `Shape mismatch` → Model/data incompatibility

---

## 📊 Expected Performance

**With movie_reviews_sentiment.csv** (300 samples):

| Metric | Expected Value |
|--------|---------------|
| Training Time | 1-3 minutes |
| Initial Loss | ~0.69 (random) |
| Final Loss (Epoch 2) | ~0.1-0.3 |
| Memory Usage | <2GB RAM |
| GPU | Optional (CPU works fine) |

---

## ✅ Success Criteria

**The test is successful if**:

1. ✅ Dataset uploads and analyzes correctly
2. ✅ Gemini recommends 5 models with reasoning
3. ✅ You can select and switch between models
4. ✅ Parameters update when switching models
5. ✅ Training starts without errors
6. ✅ WebSocket shows `Connected`
7. ✅ Metrics update in real-time
8. ✅ Loss decreases over epochs
9. ✅ Training completes successfully
10. ✅ Back button works correctly

---

## 🎯 What You're Testing

### Full Stack Integration:
- **Frontend** → **API** → **Backend** → **ML Pipeline**

### Key Features:
1. **Data Upload & Analysis** (Gemini integration)
2. **RAG Model Recommendation** (Vector search + LLM ranking)
3. **Multi-Model Selection** (Interactive UI)
4. **Dynamic Parameters** (Architecture-specific schemas)
5. **Non-Blocking Training** (Threading)
6. **Real-Time Streaming** (WebSocket)
7. **Error Handling** (Frontend + Backend)
8. **Navigation** (Back button with confirmation)

---

## 📸 Screenshots to Capture

1. **Dataset Upload Success**
2. **Model Leaderboard** (5 candidates with AI top pick)
3. **Parameter Playground** (sliders and dropdowns)
4. **Training Dashboard** (connecting)
5. **Training Dashboard** (active with graphs)
6. **Training Complete** (with download button)

---

## 🎉 Next Steps After Success

1. Try different datasets
2. Test with larger datasets (1000+ samples)
3. Test error scenarios (invalid CSV, connection drop)
4. Add more model architectures to registry
5. Implement model download functionality
6. Deploy to production!

---

**Good Luck! 🚀**

_If you encounter any issues, check the backend logs first. Most problems are clearly logged with actionable error messages._
