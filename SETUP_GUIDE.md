# 🚀 No-Code ML Training Platform - Setup Guide

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional (CUDA-compatible for faster training)

### Required Accounts
- **Google API Key** (for Gemini LLM integration)
- **Weights & Biases Account** (optional, for experiment tracking)

---

## 🔧 Backend Setup

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `backend` directory:

```env
# Google Gemini API Key (Required)
GOOGLE_API_KEY=your_google_api_key_here

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

**Get Google API Key:**
1. Visit https://makersuite.google.com/app/apikey
2. Create new API key
3. Copy and paste into `.env` file

### 5. Create Required Directories
```bash
mkdir -p uploads runs checkpoints exports reports
```

### 6. Start Backend Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Test Backend:**
Open browser to http://localhost:8000
You should see: `{"status": "ok", "message": "Backend is running!"}`

---

## 🎨 Frontend Setup

### 1. Navigate to Frontend Directory
```bash
cd nlp-finetune-ui
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Configure Environment Variables
Create a `.env` file in the `nlp-finetune-ui` directory:

```env
VITE_API_URL=http://localhost:8000
```

### 4. Start Development Server
```bash
npm run dev
```

**Expected Output:**
```
VITE v6.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### 5. Open Application
Navigate to http://localhost:5173 in your browser

---

## ✅ Verification Checklist

### Backend Health Check
- [ ] Backend server running on port 8000
- [ ] Health endpoint returns OK: `curl http://localhost:8000`
- [ ] WebSocket connection available
- [ ] Required directories created

### Frontend Health Check
- [ ] Frontend running on port 5173
- [ ] Can access dashboard page
- [ ] No console errors in browser DevTools
- [ ] API connection successful

### Integration Test
- [ ] Upload a test CSV file
- [ ] Select a model
- [ ] Start training (even for 1 epoch)
- [ ] See real-time metrics updating
- [ ] Pause/resume works
- [ ] Export model works

---

## 🧪 Quick Test Workflow

### 1. Prepare Test Dataset
Create `test_data.csv` in `backend/uploads/`:

```csv
text,label
"This is great!",1
"I love this product",1
"Terrible experience",0
"Not recommended",0
"Amazing quality",1
"Very disappointed",0
```

### 2. Test API Endpoints

**Validate Dataset:**
```bash
curl -X POST http://localhost:8000/validate-dataset \
  -F "file=@backend/uploads/test_data.csv"
```

**Get Model Candidates:**
```bash
curl "http://localhost:8000/model-candidates?task=classification"
```

**Start Training:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "distilbert-base-uncased",
    "dataset_path": "uploads/test_data.csv",
    "text_col": "text",
    "label_col": "label",
    "num_labels": 2,
    "epochs": 2,
    "batch_size": 2
  }'
```

### 3. Monitor Training
- Open http://localhost:5173
- Navigate to training dashboard
- Watch real-time metrics
- Test pause/resume controls

---

## 🐛 Troubleshooting

### Backend Issues

**Port Already in Use:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

**Missing Dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**CUDA/GPU Issues:**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, training will use CPU (slower but works)
```

### Frontend Issues

**Port 5173 in Use:**
```bash
# Change port in vite.config.ts
export default defineConfig({
  server: { port: 3000 }
})
```

**Module Not Found:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**API Connection Failed:**
- Check backend is running
- Verify VITE_API_URL in `.env`
- Check CORS settings in backend

### Common Errors

**"Training job not found":**
- Backend restarted (training state lost)
- Use checkpoint system to resume

**"Out of memory":**
- Reduce batch_size in training config
- Use smaller model (DistilBERT)
- Close other applications

**WebSocket disconnected:**
- Check firewall settings
- Verify WebSocket endpoint accessible
- Backend may have crashed (check logs)

---

## 📊 Performance Optimization

### For CPU Training
```python
# In training config
{
  "batch_size": 4,  # Smaller batches
  "epochs": 3,      # Fewer epochs
  "model": "distilbert-base-uncased"  # Smaller model
}
```

### For GPU Training
```python
# In training config
{
  "batch_size": 32,  # Larger batches
  "epochs": 10,
  "model": "bert-base-uncased"
}
```

---

## 🔐 Security Notes

### Development Mode
- CORS is set to allow all origins (`*`)
- No authentication required
- Suitable for local testing only

### Before Production
- [ ] Enable authentication
- [ ] Restrict CORS origins
- [ ] Add rate limiting
- [ ] Enable HTTPS
- [ ] Secure API keys

---

## 📝 Next Steps

After successful local testing:
1. ✅ Run comprehensive test suite
2. ✅ Fix any bugs discovered
3. ✅ Optimize performance
4. 🚀 Deploy to Vercel (Frontend) + Railway/Render (Backend)

---

## 🆘 Getting Help

**Check Logs:**
- Backend: Terminal where `uvicorn` is running
- Frontend: Browser DevTools Console
- Training: `backend/runs/<run_id>/logs/`

**Common Log Locations:**
- Backend errors: Terminal output
- Training metrics: `backend/runs/<run_id>/metrics.json`
- WebSocket messages: Browser Network tab

**Need Support?**
- Check GitHub Issues
- Review API documentation
- Test with minimal example first
