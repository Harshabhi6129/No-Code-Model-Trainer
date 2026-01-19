# 🤖 AI Model Training Platform

A **clean, modern, and minimal** platform for training ML models with zero code complexity.

## ✨ What We Built

A streamlined 4-step workflow:

1. **📤 Upload Dataset** → Drag & drop CSV files with smart analysis
2. **📊 Data Preview** → Instant stats, task detection, and data insights  
3. **🧠 AI Model Selection** → Smart recommendations with dynamic parameters
4. **⚡ Live Training** → Real-time metrics, charts, and progress tracking
5. **📦 Export Results** → Download trained models with inference code

## 🎨 Modern UI Features

- **Glassmorphism Design** - Transparent, blurred glass effects
- **Dark Theme** - Easy on the eyes with purple/pink gradients
- **Smooth Animations** - Framer Motion powered transitions
- **Real-time Updates** - WebSocket live training metrics
- **Responsive Layout** - Works on all screen sizes

## 🚀 Quick Start

```bash
# One command to start everything
./start.sh
```

This will:
- Start backend API on `http://localhost:8000`
- Start frontend UI on `http://localhost:5173`
- Install all dependencies automatically

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Frontend (React)              │
│  - Modern UI with Glassmorphism         │
│  - Real-time WebSocket updates          │
│  - Interactive charts and controls      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│          Backend (FastAPI)              │
│  - Training orchestration               │
│  - LLM-powered recommendations          │
│  - Real-time metrics streaming          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ML Engine (HuggingFace)            │
│  - Model training and fine-tuning       │
│  - Checkpoint management                │
│  - Distributed training support         │
└─────────────────────────────────────────┘
```

## 📁 Project Structure

```
Agent/
├── backend/
│   ├── main.py          # Single backend file (200 lines)
│   ├── requirements.txt # Minimal dependencies
│   └── uploads/         # Dataset storage
├── nlp-finetune-ui/
│   └── src/
│       ├── App.tsx           # Main app with step flow
│       ├── components/
│       │   ├── UploadStep.tsx      # Step 1: Upload
│       │   ├── DataPreview.tsx     # Step 2: Preview  
│       │   ├── ModelSelection.tsx  # Step 3: Models
│       │   ├── TrainingDashboard.tsx # Step 4: Training
│       │   └── ExportResults.tsx   # Step 5: Export
│       └── App.css       # Glassmorphism styles
└── start.sh             # One-command startup
```

## 🎯 Key Improvements

### ❌ What We Removed
- Complex state management (Zustand)
- Multiple routing pages
- Overly complex components
- Unnecessary API abstractions
- Heavy dependencies

### ✅ What We Kept Simple
- **Single backend file** - Everything in `main.py`
- **Step-by-step flow** - Linear, intuitive progression
- **Real-time updates** - WebSocket for live metrics
- **Modern UI** - Clean, animated, responsive
- **Smart AI features** - Model recommendations, parameter tuning

## 🛠️ Tech Stack

**Minimal & Modern:**
- **Backend**: FastAPI + Pandas + WebSockets
- **Frontend**: React + TypeScript + Framer Motion + Chart.js
- **Styling**: TailwindCSS + Custom glassmorphism
- **Icons**: Lucide React (lightweight)

## 📊 Features

- ✅ **Drag & Drop Upload** - Instant CSV analysis
- ✅ **AI Task Detection** - Auto-detect classification/generation
- ✅ **Smart Model Recommendations** - Best models for your data
- ✅ **Dynamic Parameters** - Sliders, dropdowns, real-time updates
- ✅ **Live Training Metrics** - Loss, accuracy, progress charts
- ✅ **Real-time WebSocket** - 100ms update frequency
- ✅ **Model Export** - Download with inference code
- ✅ **Modern Animations** - Smooth, professional transitions

## 🎨 Design Philosophy

**Less is More:**
- Minimal code, maximum impact
- Clean, flowing user experience  
- Modern glassmorphism aesthetics
- Intuitive step-by-step workflow
- Real-time feedback everywhere

**Performance First:**
- Single backend file
- Lightweight frontend components
- Efficient WebSocket updates
- Fast CSV processing
- Smooth 60fps animations

---

**Built for simplicity, designed for beauty** ✨