# ModelForge
> Describe what you want to build. We train the model.

ModelForge is an **agent-native** AI model training platform. You tell it your problem in plain English and an orchestrated team of Claude agents handles dataset analysis, model selection, training, evaluation, and deployment.

## Repo layout
```
apps/web/    Next.js 16 + shadcn/ui + Tailwind
backend/     FastAPI — WebSocket streaming, training job API
agents/      Claude Agent SDK — orchestrated training pipeline
```

## Quickstart
```bash
pnpm install
cd apps/web && pnpm dev           # port 3000
cd backend && uvicorn main:app --reload --port 8000
cd agents && uv pip install -e ".[dev]" && pytest
```

## Tech stack
| Layer | Choice |
|---|---|
| Frontend | Next.js 15 App Router + shadcn/ui + Tailwind v4 |
| Backend | FastAPI + WebSocket + asyncio |
| Agents | Claude Agent SDK (Python), claude-sonnet-4-6 |
| ML | HuggingFace Transformers + Unsloth (LoRA/QLoRA) + vLLM |
| DB | Postgres (Neon) — coming soon |
| GPU | Modal — coming soon |
| Auth | Clerk — coming soon |
| Billing | Stripe — coming soon |

## Configuration
```bash
cp backend/.env.example backend/.env
cp apps/web/.env.example apps/web/.env.local
```

Required: `ANTHROPIC_API_KEY`, `HUGGINGFACE_TOKEN`
