# Model Registry Setup Guide

## Step 1: Obtain HuggingFace Models Metadata CSV

You need a CSV file containing HuggingFace model metadata with the following columns:
- `modelId` (or similar): The model identifier (e.g., "bert-base-uncased")
- `downloads`: Number of downloads
- `likes`: Number of likes
- `task` or `pipeline_tag`: The task type (text-classification, etc.)

### Option 1: Download from Kaggle (Recommended)
1. Visit Kaggle and search for "HuggingFace Models Dataset" or "HF Hub Models Metadata"
2. Download the CSV file
3. Place it at: `backend/data/hf_models_metadata.csv`

### Option 2: Generate from HuggingFace API (Slower, one-time)
If no pre-made dataset is available, you can generate one:

```bash
cd backend/scripts
python generate_metadata_csv.py
```

(This script would call `list_models()` and save to CSV - can be created if needed)

### Option 3: Use a Subset for Testing
For quick testing, you can use a smaller manually-created CSV:

```csv
modelId,downloads,likes,task
distilbert-base-uncased,50000000,250,text-classification
bert-base-uncased,100000000,500,text-classification
roberta-base,80000000,400,text-classification
```

Save this as `backend/data/hf_models_metadata.csv`

## Step 2: Run Bootstrap Script

```bash
cd backend
python scripts/bootstrap_registry.py
```

This will:
- Load the metadata CSV
- Filter models (downloads > 10,000, specific tasks)
- Save top 500 to `data/candidates_list.json`

## Step 3: Run Enrichment Script

```bash
python scripts/enrich_registry.py
```

This will:
- Fetch `config.json` for each candidate
- Extract technical specs (architectures, max_seq_length, etc.)
- Save to `data/model_registry.db`

## Step 4: Verify Database

```bash
sqlite3 data/model_registry.db
```

```sql
SELECT COUNT(*) FROM models;
SELECT * FROM models LIMIT 5;
```

---

Ready to proceed! 🚀
