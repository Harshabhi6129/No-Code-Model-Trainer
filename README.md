# 🤖 No-Code ML Training Platform

A professional, enterprise-grade platform for fine-tuning machine learning models without writing code.

## ✨ Features

- 🎯 **No-Code Training** - Fine-tune models through intuitive UI
- 🔄 **Real-time Control** - Pause, resume, and adjust training live
- 🤖 **AI-Powered** - Intelligent model recommendations and analysis
- 📊 **Advanced Visualizations** - Real-time metrics and interactive charts
- 🎨 **Modern UI** - Glassmorphism design with dark mode
- 📦 **Complete Export** - Download models with inference code
- 🔧 **Hyperparameter Optimization** - Automated parameter tuning
- 📈 **Comprehensive Reports** - Professional training documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- 8GB RAM minimum
- Google API Key (for LLM features)

### One-Command Setup

```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:
1. Set up Python virtual environment
2. Install all dependencies
3. Create test dataset
4. Start backend (port 8000)
5. Start frontend (port 5173)

### Manual Setup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

**Frontend:**
```bash
cd nlp-finetune-ui
npm install
npm run dev
```

## 🧪 Testing

### Quick API Test
```bash
chmod +x test_api.sh
./test_api.sh
```

### Full Test Suite
```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd nlp-finetune-ui
npm test
```

## 📖 Documentation

- [Setup Guide](SETUP_GUIDE.md) - Detailed installation instructions
- [Testing Guide](TESTING_GUIDE.md) - Comprehensive testing procedures
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment steps

## 🎯 Quick Test Workflow

1. **Open** http://localhost:5173
2. **Upload** `backend/uploads/test_sentiment.csv`
3. **Select** Text Classification task
4. **Choose** DistilBERT model
5. **Configure** 2 epochs for quick test
6. **Start** training and watch real-time metrics
7. **Test** pause/resume controls
8. **Export** trained model

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

## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- PyTorch - Deep learning framework
- HuggingFace Transformers - Pre-trained models
- Google Gemini - LLM integration
- Weights & Biases - Experiment tracking

**Frontend:**
- React 18 - UI framework
- TypeScript - Type safety
- Vite - Build tool
- Framer Motion - Animations
- Chart.js - Visualizations
- TailwindCSS - Styling

## 📊 Platform Capabilities

### Training Features
- ✅ Pause/Resume/Stop training
- ✅ Live parameter adjustment
- ✅ Checkpoint system
- ✅ Multi-GPU support
- ✅ Distributed training
- ✅ Early stopping
- ✅ Learning rate scheduling

### Intelligence Features
- ✅ AI model recommendations
- ✅ Automated hyperparameter tuning
- ✅ Training progress analysis
- ✅ Overfitting detection
- ✅ Performance optimization suggestions
- ✅ Resource monitoring

### Export Features
- ✅ Trained model weights
- ✅ Inference code generation
- ✅ Training reports with charts
- ✅ Comprehensive documentation
- ✅ Ready-to-deploy packages

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```env
GOOGLE_API_KEY=your_key_here
WANDB_API_KEY=your_wandb_key
HOST=0.0.0.0
PORT=8000
```

**Frontend (.env):**
```env
VITE_API_URL=http://localhost:8000
```

## 📈 Performance

- **Training Speed**: Optimized for both CPU and GPU
- **Real-time Updates**: <100ms latency
- **API Response**: <200ms average
- **Memory Efficient**: Checkpoint-based training
- **Scalable**: Supports distributed training

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed
kill -9 <PID>
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Training fails
- Reduce batch size
- Use smaller model (DistilBERT)
- Check GPU memory
- Verify dataset format

### API connection issues
- Ensure backend is running
- Check CORS settings
- Verify API URL in frontend .env

## 🚀 Deployment

### Production Deployment

**Frontend (Vercel):**
```bash
cd nlp-finetune-ui
vercel --prod
```

**Backend (Railway):**
```bash
cd backend
railway up
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📝 API Documentation

### Key Endpoints

- `GET /` - Health check
- `POST /validate-dataset` - Validate uploaded dataset
- `GET /model-candidates` - Get model recommendations
- `POST /train` - Start training
- `POST /api/training/{id}/pause` - Pause training
- `POST /api/training/{id}/resume` - Resume training
- `GET /api/training/{id}/status` - Get training status
- `GET /api/training/{id}/report` - Generate report
- `GET /export/{id}` - Download trained model

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community]
- 📖 Docs: [Full documentation]
- 🐛 Issues: [GitHub Issues]

## 🎉 Acknowledgments

- HuggingFace for transformers library
- FastAPI for excellent web framework
- Vercel for hosting platform
- Google for Gemini API

---

**Built with ❤️ for the ML community**

⭐ Star us on GitHub if you find this useful!