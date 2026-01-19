# 🚀 Deployment Guide - Vercel + Railway

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├──────────────────┐
                     │                  │
                     ▼                  ▼
         ┌───────────────────┐  ┌──────────────────┐
         │  VERCEL (Frontend)│  │ RAILWAY (Backend)│
         │  - React App      │  │ - FastAPI        │
         │  - Static Assets  │  │ - WebSocket      │
         │  - CDN Cached     │  │ - ML Training    │
         └───────────────────┘  └──────────────────┘
                                         │
                                         ▼
                                 ┌──────────────────┐
                                 │  STORAGE         │
                                 │  - Models        │
                                 │  - Datasets      │
                                 │  - Checkpoints   │
                                 └──────────────────┘
```

---

## 📦 Part 1: Backend Deployment (Railway)

### Why Railway?
- ✅ Supports long-running processes (training)
- ✅ WebSocket support
- ✅ Persistent storage
- ✅ Easy Python deployment
- ✅ Free tier available

### Step 1: Prepare Backend for Deployment

Create `backend/railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

Create `backend/Procfile`:
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2
```

Update `backend/requirements.txt` (ensure all deps listed):
```txt
fastapi==0.104.1
pydantic==1.10.13
uvicorn[standard]==0.24.0
python-multipart==0.0.6
torch==2.1.0
transformers==4.35.0
datasets==2.14.6
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
psutil==5.9.6
GPUtil==1.4.0
python-dotenv==1.0.0
requests==2.31.0
wandb==0.16.0
```

### Step 2: Deploy to Railway

1. **Create Railway Account:**
   - Visit https://railway.app
   - Sign up with GitHub

2. **Create New Project:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login
   railway login
   
   # Initialize project
   cd backend
   railway init
   ```

3. **Configure Environment Variables:**
   ```bash
   railway variables set GOOGLE_API_KEY=your_key_here
   railway variables set WANDB_API_KEY=your_wandb_key
   railway variables set PYTHONUNBUFFERED=1
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

5. **Get Deployment URL:**
   ```bash
   railway domain
   # Example: https://your-app.railway.app
   ```

### Step 3: Configure Storage (Railway Volumes)

```bash
# Create persistent volume for models
railway volume create models-storage --mount /app/runs

# Create volume for uploads
railway volume create uploads-storage --mount /app/uploads
```

---

## 🎨 Part 2: Frontend Deployment (Vercel)

### Why Vercel?
- ✅ Optimized for React/Vite
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Zero-config deployment
- ✅ Free tier generous

### Step 1: Prepare Frontend for Deployment

Update `nlp-finetune-ui/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'chart-vendor': ['chart.js', 'react-chartjs-2'],
          'ui-vendor': ['@mui/material', 'framer-motion']
        }
      }
    }
  },
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
```

Create `nlp-finetune-ui/vercel.json`:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://your-backend.railway.app/api/:path*"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ]
}
```

Create `nlp-finetune-ui/.env.production`:
```env
VITE_API_URL=https://your-backend.railway.app
VITE_WS_URL=wss://your-backend.railway.app
```

### Step 2: Deploy to Vercel

**Option A: Vercel CLI**
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd nlp-finetune-ui
vercel --prod
```

**Option B: GitHub Integration (Recommended)**

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/ml-platform.git
   git push -u origin main
   ```

2. **Connect to Vercel:**
   - Visit https://vercel.com
   - Click "Import Project"
   - Select your GitHub repository
   - Configure:
     - Framework: Vite
     - Root Directory: `nlp-finetune-ui`
     - Build Command: `npm run build`
     - Output Directory: `dist`

3. **Set Environment Variables:**
   - Go to Project Settings → Environment Variables
   - Add:
     - `VITE_API_URL`: Your Railway backend URL
     - `VITE_WS_URL`: Your Railway WebSocket URL

4. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete
   - Get deployment URL: `https://your-app.vercel.app`

---

## 🔧 Part 3: Post-Deployment Configuration

### Update Backend CORS

Edit `backend/app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",
        "http://localhost:5173",  # Keep for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Redeploy backend:
```bash
cd backend
railway up
```

### Configure Custom Domain (Optional)

**For Frontend (Vercel):**
1. Go to Project Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

**For Backend (Railway):**
1. Go to Project Settings → Domains
2. Add custom domain
3. Update DNS CNAME record

---

## 🧪 Part 4: Deployment Testing

### Test Checklist

```bash
# 1. Test backend health
curl https://your-backend.railway.app/

# 2. Test API endpoint
curl https://your-backend.railway.app/model-candidates?task=classification

# 3. Test frontend loads
open https://your-app.vercel.app

# 4. Test WebSocket connection
# Open browser console and check for WebSocket connection

# 5. Test full workflow
# Upload dataset → Select model → Start training
```

### Monitor Deployment

**Railway Monitoring:**
```bash
# View logs
railway logs

# Check metrics
railway status
```

**Vercel Monitoring:**
- Visit Vercel Dashboard
- Check Analytics tab
- Monitor build logs

---

## 📊 Part 5: Performance Optimization

### Frontend Optimization

1. **Enable Compression:**
   Already handled by Vercel automatically

2. **Optimize Images:**
   ```bash
   npm install -D vite-plugin-imagemin
   ```

3. **Code Splitting:**
   Already configured in `vite.config.ts`

### Backend Optimization

1. **Add Caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def get_model_candidates(task: str):
       # Cached for repeated calls
       pass
   ```

2. **Enable Compression:**
   ```python
   from fastapi.middleware.gzip import GZipMiddleware
   app.add_middleware(GZipMiddleware, minimum_size=1000)
   ```

3. **Database Connection Pooling:**
   (If using database in future)

---

## 🔒 Part 6: Security Hardening

### Environment Variables Security

**Never commit:**
- `.env` files
- API keys
- Secrets

**Add to `.gitignore`:**
```
.env
.env.local
.env.production
*.key
*.pem
```

### API Security

Add rate limiting:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/training/list")
@limiter.limit("10/minute")
async def list_training_runs():
    pass
```

---

## 📈 Part 7: Monitoring & Logging

### Setup Logging

Create `backend/logging_config.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('app.log', maxBytes=10485760, backupCount=5),
            logging.StreamHandler()
        ]
    )
```

### Error Tracking

Consider integrating:
- **Sentry** for error tracking
- **LogRocket** for session replay
- **Datadog** for APM

---

## 🚨 Part 8: Rollback Plan

### If Deployment Fails

**Vercel Rollback:**
```bash
# List deployments
vercel ls

# Rollback to previous
vercel rollback [deployment-url]
```

**Railway Rollback:**
```bash
# View deployments
railway status --json

# Rollback
railway rollback [deployment-id]
```

---

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] All tests passing locally
- [ ] Environment variables configured
- [ ] CORS settings updated
- [ ] API keys secured
- [ ] Build succeeds locally

### Deployment
- [ ] Backend deployed to Railway
- [ ] Frontend deployed to Vercel
- [ ] Environment variables set
- [ ] Custom domains configured (if any)
- [ ] SSL certificates active

### Post-Deployment
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] WebSocket connections working
- [ ] Training workflow functional
- [ ] Monitoring setup
- [ ] Error tracking active

### Documentation
- [ ] Deployment URLs documented
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Team notified

---

## 🎉 Success!

Your platform is now live at:
- **Frontend:** https://your-app.vercel.app
- **Backend:** https://your-backend.railway.app

**Next Steps:**
1. Monitor initial usage
2. Gather user feedback
3. Iterate and improve
4. Scale as needed

---

## 🆘 Troubleshooting

### Common Issues

**Build Fails:**
- Check build logs
- Verify all dependencies in package.json/requirements.txt
- Test build locally first

**API Not Connecting:**
- Verify CORS settings
- Check environment variables
- Ensure backend is running

**WebSocket Issues:**
- Check WSS protocol (not WS)
- Verify Railway supports WebSocket
- Check firewall settings

**Out of Memory:**
- Upgrade Railway plan
- Optimize model loading
- Reduce batch sizes

---

## 📞 Support Resources

- **Railway Docs:** https://docs.railway.app
- **Vercel Docs:** https://vercel.com/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Vite Docs:** https://vitejs.dev
