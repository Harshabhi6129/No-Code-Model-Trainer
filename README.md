---
title: ModelForge Backend
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ModelForge
> Describe what you want to build. We train the model.

ModelForge is an **agent-native** AI model training platform. You tell it your problem in plain English and an orchestrated team of Claude agents handles dataset analysis, model selection, training, evaluation, and deployment.

## Repo layout
```
apps/web/    Next.js 16 + shadcn/ui + Tailwind
backend/     FastAPI — SSE streaming, training job API
agents/      Claude Agent SDK — orchestrated training pipeline
```

## Quickstart
```bash
pnpm install
cd apps/web && pnpm dev           # port 3456
cd backend && uvicorn main:app --reload --port 8000
cd agents && python -m pytest tests/ -v
```

## Tech stack
| Layer | Choice |
|---|---|
| Frontend | Next.js 16 App Router + shadcn/ui + Tailwind v4 |
| Backend | FastAPI + SSE streaming + asyncio |
| Agents | Claude Agent SDK (Python), claude-sonnet-4-6 |
| ML | HuggingFace Transformers + LoRA/QLoRA (PEFT) |
| DB | Supabase (Postgres + Auth) |
| GPU | Modal (serverless H100 dispatch) |
| Hosting | Vercel (frontend) + HuggingFace Spaces (backend) |

## Configuration
```bash
cp backend/.env.example backend/.env
cp apps/web/.env.example apps/web/.env.local
```

Required: `ANTHROPIC_API_KEY`, `HUGGINGFACE_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
