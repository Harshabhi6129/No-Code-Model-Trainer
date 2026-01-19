# Testing Guide for No-Code Model Trainer

## Dataset Created: `test_sentiment_dataset.csv`

### Dataset Details:
- **Task Type:** Text Classification (Sentiment Analysis)
- **Total Records:** 50 samples
- **Labels:** 3 classes (positive, negative, neutral)
- **Use Case:** Product/Movie review sentiment classification

---

## Step-by-Step Testing Flow

### 1. **Upload Dataset** 📤
1. Go to http://localhost:5173
2. Look for the "Upload Dataset" or "Get Started" section
3. Click "Choose File" or drag-and-drop area
4. Select: `/Users/harshaabhinavkusampudi/Documents/Agent/backend/uploads/test_sentiment_dataset.csv`
5. Click "Upload" or "Validate"

**Expected Result:**
- ✅ Dataset validation passes
- See preview of first 5 rows
- Statistics: 50 rows, 2 columns (text, label)
- Label distribution chart showing positive/negative/neutral counts

---

### 2. **Review Dataset Insights** 🔍
After upload, you should see:
- **Column Analysis:** `text` (string), `label` (categorical)
- **Missing Values:** None
- **Label Balance:** 
  - Positive: ~40% (20 samples)
  - Negative: ~40% (20 samples)  
  - Neutral: ~20% (10 samples)
- **AI Insights:** LLM-generated suggestions about your dataset

---

### 3. **Choose Task Type** 🎯
- Select: **"Text Classification"** or **"Sentiment Analysis"**
- The system should auto-detect this from your labels

---

### 4. **Model Selection** 🤖
You'll see recommended models:
- **distilbert-base-uncased** (Recommended for beginners - fast, efficient)
- **bert-base-uncased** (More accurate, slower)
- **roberta-base** (Best accuracy, needs more resources)

**For Testing:** Choose `distilbert-base-uncased`

---

### 5. **Configure Hyperparameters** ⚙️

The system will suggest optimal parameters. For quick testing:

```
Learning Rate: 2e-5 (default)
Batch Size: 8 (reduce to 4 if memory issues)
Epochs: 2 (just for testing - normally use 3-5)
Weight Decay: 0.01
```

**Why these values?**
- Small dataset (50 samples) = small batch size
- 2 epochs = quick test run (~2-3 minutes)
- Learning rate 2e-5 is standard for BERT models

---

### 6. **Start Training** 🚀
1. Click "Start Training" or "Begin Fine-tuning"
2. You'll see:
   - Real-time progress bar
   - Loss metrics decreasing
   - Accuracy increasing each epoch
   - Training/validation splits
   - System resource usage (CPU/RAM)

**Expected Timeline:**
- Epoch 1: ~1-2 minutes
- Epoch 2: ~1-2 minutes
- **Total:** ~3-4 minutes

---

### 7. **Monitor Training** 📊
Watch the live metrics:
- **Training Loss:** Should decrease (e.g., 1.2 → 0.8 → 0.4)
- **Validation Accuracy:** Should increase (e.g., 60% → 75% → 85%)
- **Learning Rate Schedule:** Shows adjustments
- **Gradient Norms:** Check for stability

---

### 8. **Review Results** 📈
After training completes:
- **Final Metrics:**
  - Accuracy: ~85-90% (on this small dataset)
  - F1 Score: ~0.85
  - Precision/Recall per class
- **Confusion Matrix:** Visual breakdown of predictions
- **Sample Predictions:** See 5 example predictions with confidence scores

---

### 9. **Export Model** 💾
Options to export:
- **PyTorch Model** (.pt format)
- **ONNX Format** (for deployment)
- **HuggingFace Format** (for sharing)
- **Training Report** (PDF with all metrics)

---

## Alternative Test Scenarios

### Quick 30-Second Test:
```
Dataset: test_sentiment_dataset.csv
Model: distilbert-base-uncased
Epochs: 1
Batch Size: 8
```

### Full Production Test:
```
Dataset: test_sentiment_dataset.csv
Model: roberta-base
Epochs: 5
Batch Size: 16
Enable early stopping
Enable checkpointing
```

---

## API Testing (Optional)

You can also test via API:

```bash
# 1. Validate dataset
curl -X POST http://localhost:8000/validate-dataset \
  -F "file=@backend/uploads/test_sentiment_dataset.csv"

# 2. Get model suggestions
curl "http://localhost:8000/model-candidates?task=classification"

# 3. Start training
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model": "distilbert-base-uncased",
    "dataset_path": "uploads/test_sentiment_dataset.csv",
    "text_col": "text",
    "label_col": "label",
    "num_labels": 3,
    "epochs": 2,
    "batch_size": 8,
    "learning_rate": 2e-5
  }'
```

---

## Expected Outcomes

### ✅ Success Indicators:
- Dataset validates without errors
- Training starts within 30 seconds
- Loss decreases steadily
- Accuracy reaches >80% by epoch 2
- Model exports successfully

### ⚠️ Potential Issues:
- **Out of Memory:** Reduce batch_size to 4
- **Training too slow:** Already using smallest model
- **Low accuracy:** Expected with only 50 samples (use for demo only)

---

## Tips for Real Use

This test dataset is small (50 samples) for demonstration. For production:
- **Minimum:** 1,000+ samples per class
- **Recommended:** 5,000+ samples per class
- **Optimal:** 10,000+ samples per class

---

**Ready to test!** The dataset is already uploaded at:
`/Users/harshaabhinavkusampudi/Documents/Agent/backend/uploads/test_sentiment_dataset.csv`

Start at http://localhost:5173 and follow the steps above! 🚀
