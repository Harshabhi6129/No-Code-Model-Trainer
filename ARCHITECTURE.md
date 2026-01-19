# Complete Application Architecture & Flow Documentation

## 🎯 CURRENT ISSUE: "No models found for task type: Text Classification"

**Root Cause**: Task type mismatch between frontend and backend
- **Frontend sends**: `"Text Classification"` (with capital letters and space)
- **Database has**: `"text-classification"` (lowercase with hyphen)
- **Fix needed**: Convert task type to lowercase with hyphens before filtering

---

## 📁 PROJECT STRUCTURE

```
/Users/harshaabhinavkusampudi/Documents/Agent/
│
├── backend/                          # FastAPI Backend
│   ├── app.py                        # Main API server (506 lines)
│   ├── dataset_validator.py         # CSV validation & analysis
│   ├── llm_service.py               # Gemini AI integration
│   ├── model_params.py              # Parameter schemas (simplified)
│   │
│   ├── services/                    # New services directory
│   │   ├── recommender.py           # ⭐ RAG-based model recommender
│   │   ├── socket_manager.py        # WebSocket connection manager
│   │   ├── callbacks.py             # Custom HF training callbacks
│   │   └── trainer.py               # Training orchestrator
│   │
│   ├── scripts/                     # Data processing scripts
│   │   ├── bootstrap_registry.py    # Fetch top models from HF API
│   │   ├── enrich_registry.py       # Download config.json files
│   │   └── build_index.py           # Create vector embeddings
│   │
│   ├── config/
│   │   └── param_mappings.py        # Architecture → Hyperparam mappings
│   │
│   └── data/                        # Generated data files
│       ├── model_registry.db        # SQLite: 143 models
│       ├── model_embeddings.pkl     # Vector index (220KB)
│       └── candidates_list.json     # Bootstrap output
│
└── nlp-finetune-ui/                 # React + Vite Frontend
    └── src/
        ├── components/
        │   ├── ModelSelection.tsx   # ⭐ Main model selection UI
        │   └── TrainingDashboard.tsx # Live training monitor
        ├── hooks/
        │   ├── useModelRecommendation.ts  # API hook for /api/recommend
        │   └── useTrainingSocket.ts       # WebSocket hook
        └── config/
            └── frontend_params.ts    # UI parameter schemas
```

---

## 🔄 APPLICATION FLOW

### Phase 1: Data Ingestion (Bootstrap & Enrich)

**What Happened:**
```
1. bootstrap_registry.py
   ├── Called HuggingFace API: HfApi.list_models()
   ├── Fetched TOP 50 models for each task:
   │   - text-classification (50 models)
   │   - text-generation (50 models)
   │   - image-classification (50 models)
   ├── Saved to: candidates_list.json (150 models)
   └── Task format: "text-classification" ✅

2. enrich_registry.py
   ├── Read candidates_list.json
   ├── For each model:
   │   ├── Download config.json from HuggingFace
   │   └── Extract: architectures, vocab_size, model_type, etc.
   ├── Store in SQLite: model_registry.db
   └── Success rate: 143/150 (95.3%)

3. build_index.py
   ├── Load all 143 models from DB
   ├── Create searchable strings: "model_id + task + architecture"
   ├── Generate embeddings: sentence-transformers (all-MiniLM-L6-v2)
   └── Save: model_embeddings.pkl (384 dimensions)
```

**Database Schema:**
```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,           -- e.g., "distilbert/distilbert-base-uncased..."
    task TEXT,                     -- e.g., "text-classification"
    downloads INTEGER,
    likes INTEGER,
    architectures TEXT,            -- JSON array
    max_position_embeddings INTEGER,
    vocab_size INTEGER,
    model_type TEXT
);
```

---

### Phase 2: Model Recommendation (RAG System)

**The 3-Stage Pipeline:**

#### Stage 1: Hard Filter (Task Type)
```python
# In services/recommender.py
filtered_indices = self._filter_by_task(task_type)
```

**❌ CURRENT BUG:**
```python
# Frontend sends:
task_type = "Text Classification"

# Recommender filters:
if model['task'] == task_type:  # FAILS!
    # Because DB has "text-classification" not "Text Classification"
```

**✅ FIX NEEDED:**
```python
# Normalize task type before filtering
task_type = task_type.lower().replace(" ", "-")
# "Text Classification" → "text-classification"
```

#### Stage 2: Semantic Search
```python
# Convert user intent to vector
query_embedding = embedder.encode(["detect toxic comments"])

# Calculate cosine similarity with all filtered models
similarities = cosine_similarity(query_embedding, model_embeddings)

# Get top 5 candidates
top_5 = models.sort_by_similarity().take(5)
```

#### Stage 3: LLM Ranking (Gemini)
```python
prompt = f"""
User Intent: {user_intent}
Candidates: {top_5_models}
Return JSON:
{{
  "selected_model_id": "...",
  "reasoning": "...",
  "suggested_hyperparams": {{...}}
}}
"""

response = query_llm_gemini(prompt)
```

---

### Phase 3: Training Flow

**User Journey:**
```
1. Upload Dataset
   ├── POST /validate-dataset
   ├── Gemini analyzes intent
   └── Returns: detected_task, summary

2. Model Selection
   ├── User enters intent: "classify toxic comments"
   ├── POST /api/recommend
   │   └── {intent: "...", task_type: "text-classification"}
   ├── RAG pipeline runs
   └── Returns: {selected_model_id, parameters}

3. Start Training
   ├── User clicks "Start Training"
   ├── POST /api/train/start
   │   └── {client_id, model_id, parameters, dataset_path}
   ├── Backend:
   │   ├── Load model & tokenizer
   │   ├── Load dataset
   │   ├── Create Trainer with WebSocketCallback
   │   └── trainer.train() in background
   └── Returns: {job_id, ws_url}

4. Real-Time Monitor
   ├── Frontend connects to WS: ws://localhost:8000/ws/train/{job_id}
   ├── Backend streams:
   │   ├── {type: "metrics", loss: 0.52, lr: 2e-5}
   │   ├── {type: "epoch_end", epoch: 1}
   │   └── {type: "completion", model_path: "..."}
   └── Dashboard updates charts live
```

---

## ⭐ KEY FILES EXPLANATION

### Backend

#### `services/recommender.py` (237 lines)
**Purpose**: RAG-based model recommendation engine

**Key Methods:**
```python
class ModelRecommender:
    def __init__(self):
        # Load vector index and DB
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_data = pickle.load(embeddings_file)
        self.db_conn = sqlite3.connect(model_registry.db)
    
    def recommend(user_intent, task_type):
        # Stage 1: Filter by task
        candidates = filter_by_task(task_type)
        
        # Stage 2: Semantic search (top 5)
        candidates = semantic_search(user_intent, candidates, top_k=5)
        
        # Stage 3: LLM ranking
        result = llm_rank(user_intent, candidates)
        
        return {
            selected_model_id,
            reasoning,
            suggested_hyperparams,
            all_params,
            candidates
        }
```

#### `services/trainer.py` (128 lines)
**Purpose**: Orchestrate model training with WebSocket streaming

**Flow:**
```python
async def start_training(config, dataset_path, client_id):
    # 1. Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # 2. Load dataset
    dataset = load_dataset("glue", "sst2", split="train[:100]")  # TODO: Use real dataset
    
    # 3. Create callback
    ws_callback = WebSocketCallback(client_id, socket_manager)
    
    # 4. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[ws_callback]
    )
    trainer.train()
    
    # 5. Save
    trainer.save_model(output_path)
```

#### `services/socket_manager.py` (88 lines)
**Purpose**: Manage WebSocket connections

**Key Methods:**
```python
class ConnectionManager:
    active_connections: Dict[client_id, Set[WebSocket]]
    
    async def connect(client_id, websocket)
    async def disconnect(client_id, websocket)
    async def broadcast_json(client_id, data)
```

#### `services/callbacks.py` (125 lines)
**Purpose**: Custom HuggingFace callback for streaming

**Hooks:**
```python
class WebSocketCallback(TrainerCallback):
    def on_log(self, logs):
        # Capture: loss, learning_rate, eval_loss
        socket_manager.broadcast_json(client_id, {
            type: "metrics",
            loss: logs["loss"],
            step: state.global_step
        })
```

### Frontend

#### `components/ModelSelection.tsx` (347 lines)
**Purpose**: 2-column UI for model recommendation

**Logic:**
```typescript
const handleFindModel = async () => {
    await recommend({
        intent: userIntent,
        task_type: datasetInfo?.intent_analysis?.detected_task
    })
}

// If training started, show dashboard
if (isTraining) {
    return <TrainingDashboard clientId={jobId} />
}
```

#### `components/TrainingDashboard.tsx` (310 lines)
**Purpose**: Live training monitor with charts

**Features:**
- 4 metric cards (epoch, LR, time, loss)
- 2 Recharts (loss curve, LR schedule)
- Log stream
- Status panel

---

## 🐛 THE BUG & FIX

### Problem
```python
# Frontend (ModelSelection.tsx)
const detected_task = "Text Classification"  # From Gemini analysis

# POST to /api/recommend
fetch('/api/recommend', {
    body: JSON.stringify({
        intent: "...",
        task_type: "Text Classification"  # ❌ Wrong format
    })
})
```

```python
# Backend (services/recommender.py)
def _filter_by_task(self, task_type: str):
    for model in models:
        if model['task'] == task_type:  # ❌ "text-classification" != "Text Classification"
            filtered.append(model)
```

### Solution
**Option 1: Fix in Recommender (Backend)**
```python
# services/recommender.py, line ~129
def _filter_by_task(self, task_type: str) -> List[int]:
    # Normalize task type
    task_type = task_type.lower().replace(" ", "-")  # ✅ Add this
    
    filtered_indices = []
    for idx, model_id in enumerate(self.index_data['model_ids']):
        model = self._get_model_details(model_id)
        if model and model['task'] == task_type:
            filtered_indices.append(idx)
    return filtered_indices
```

**Option 2: Fix in Frontend (Alternative)**
```typescript
// hooks/useModelRecommendation.ts
const normalizeTaskType = (task: string) => {
    return task.toLowerCase().replace(/\s+/g, '-')
}

const recommend = async (request: RecommendationRequest) => {
    const response = await fetch('/api/recommend', {
        body: JSON.stringify({
            ...request,
            task_type: request.task_type ? normalizeTaskType(request.task_type) : undefined
        })
    })
}
```

---

## 📊 DATABASE CONTENTS

**Task Types in DB:**
```
text-classification   (50 models)
text-generation       (43 models)  
image-classification  (50 models)
```

**Sample Models:**
```
distilbert/distilbert-base-uncased-finetuned-sst-2-english
facebook/bart-large-mnli
microsoft/Phi-3-mini-4k-instruct
google/vit-base-patch16-224
```

---

## 🔧 IMMEDIATE FIX

**File**: `backend/services/recommender.py`
**Line**: ~129
**Change**:
```python
def _filter_by_task(self, task_type: str) -> List[int]:
    """Filter model indices by task type."""
    # Normalize task type to match DB format
    if task_type:
        task_type = task_type.lower().replace(" ", "-")
    
    filtered_indices = []
    for idx, model_id in enumerate(self.index_data['model_ids']):
        model = self._get_model_details(model_id)
        if model and model['task'] == task_type:
            filtered_indices.append(idx)
    return filtered_indices
```

This will fix the "No models found" error! 🎯
