#!/bin/bash

# Quick Start Script for No-Code ML Training Platform
# This script sets up and starts both backend and frontend

set -e  # Exit on error

echo "🚀 No-Code ML Training Platform - Quick Start"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"
echo ""

# Setup Backend
echo "📦 Setting up Backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create required directories
mkdir -p uploads runs checkpoints exports reports

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating template...${NC}"
    cat > .env << EOF
# Google Gemini API Key (Required for LLM features)
GOOGLE_API_KEY=your_google_api_key_here

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
EOF
    echo -e "${YELLOW}⚠️  Please edit backend/.env and add your API keys${NC}"
fi

echo -e "${GREEN}✓ Backend setup complete${NC}"
echo ""

# Setup Frontend
echo "📦 Setting up Frontend..."
cd ../nlp-finetune-ui

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << EOF
VITE_API_URL=http://localhost:8000
EOF
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"
echo ""

# Create test dataset
echo "📝 Creating test dataset..."
cd ../backend
cat > uploads/test_sentiment.csv << EOF
text,label
"This product is amazing! I love it.",1
"Great quality and fast shipping.",1
"Excellent customer service.",1
"Best purchase I've made this year.",1
"Highly recommend to everyone.",1
"Terrible product, waste of money.",0
"Very disappointed with the quality.",0
"Worst experience ever.",0
"Do not buy this product.",0
"Complete waste of time and money.",0
EOF

echo -e "${GREEN}✓ Test dataset created: backend/uploads/test_sentiment.csv${NC}"
echo ""

# Start servers
echo "🚀 Starting servers..."
echo ""
echo -e "${GREEN}Backend will start on: http://localhost:8000${NC}"
echo -e "${GREEN}Frontend will start on: http://localhost:5173${NC}"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend in background
cd /Users/harshaabhinavkusampudi/Documents/Agent/backend
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Test backend health
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✓ Backend is running${NC}"
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Start frontend
cd /Users/harshaabhinavkusampudi/Documents/Agent/nlp-finetune-ui
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo "Waiting for frontend to start..."
sleep 5

echo ""
echo "=============================================="
echo -e "${GREEN}✅ Platform is running!${NC}"
echo ""
echo "📱 Open your browser to: http://localhost:5173"
echo ""
echo "🧪 Quick Test Steps:"
echo "  1. Upload test_sentiment.csv from backend/uploads/"
echo "  2. Select 'Text Classification' task"
echo "  3. Choose 'DistilBERT' model (fastest)"
echo "  4. Set epochs to 2 for quick test"
echo "  5. Click 'Start Training'"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=============================================="

# Wait for user interrupt
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait