#!/bin/bash

echo "🚀 Starting ML Training Platform..."

# Start backend
echo "📡 Starting backend server..."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
cd ../nlp-finetune-ui
npm install
npm run dev &
FRONTEND_PID=$!

echo "✅ Platform started!"
echo "🌐 Frontend: http://localhost:5173"
echo "📡 Backend: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait