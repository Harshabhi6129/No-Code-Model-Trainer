# ModelForge — No-Code AI Model Trainer

## Product Vision

A conversational, agent-native platform that turns a plain-English problem description into a deployed, production-ready AI model. The user says *"I want to classify customer support tickets by urgency"* — and an orchestrated team of Claude agents handles dataset analysis, model selection, training, evaluation, and one-click deployment.

**Not** another form-filling AutoML tool. **Yes** to: transparent AI, forkable experiments, real-time streaming, and models that go straight to production.

## Architecture
apps/web/ Next.js 15 App Router — conversational UI + dashboard
backend/ FastAPI — job queue API, WebSocket streaming
agents/ Claude Agent SDK — orchestrated training pipeline
intent_agent Translates user intent → formal task spec
data_agent Ingest, profile, validate, clean datasets
model_agent Selects base model + training recipe
train_agent Runs training, monitors, auto-recovers
eval_agent Generates eval suite, reports calibrated metrics
deploy_agent Publishes to HF Hub / Modal / Replicate endpoint
packages/
ml/ Core ML: HuggingFace + Unsloth LoRA/QLoRA + vLLM

## Tech Stack
| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 15 App Router + shadcn/ui + Tailwind | SSR, file-based routing, one styling system |
| Backend | FastAPI + async job queue (Modal or Celery+Redis) | Long training jobs need proper queuing |
| Agents | Claude Agent SDK (Python), claude-sonnet-4-6 by default | Native tool use, streaming, MCP integration |
| ML Core | transformers + Unsloth (LoRA/QLoRA) + vLLM | 2-5x faster LoRA, production-grade inference |
| Database | Postgres (Neon) + S3/R2 for model artifacts | Experiments, users, run history |
| Auth | Clerk | Skip rolling your own |
| Billing | Stripe (usage-based: GPU-minutes) | Models cost real money to train |
| GPU Compute | Modal (serverless GPU) | No infra to manage, scales to zero |
| Observability | Langfuse (agent traces) + Sentry + W&B (training runs) | Full visibility |
| Deploy | Vercel (frontend) + Modal (backend + GPU) | Unified, scales to zero |

## Supported ML Tasks (v0 → v1 roadmap)
- **v0.1** Text classification with LoRA (encoder models — BERT, RoBERTa, DeBERTa)
- **v0.2** Token classification (NER), text generation fine-tuning
- **v0.3** Small-LLM fine-tuning (LoRA/QLoRA on Llama, Qwen, Mistral via Unsloth)
- **v1.0** Vision classification (timm), sentence embeddings, audio (Whisper)
- **v1.1** Multimodal (LLaVA-style), reranker training

## Key Files
- `apps/web/app/` — Next.js pages and layouts
- `apps/web/components/` — shadcn/ui components + custom components
- `backend/main.py` — FastAPI entrypoint
- `backend/services/trainer.py` — Real HuggingFace training pipeline (preserved from MVP)
- `agents/pipeline.py` — Top-level agent orchestrator
- `agents/` — Individual agent implementations

## Agent Autonomy
Claude agents make autonomous decisions about:
- Which base model to use given dataset size and task type
- Training recipe (full FT vs LoRA vs QLoRA vs embed+classify)
- Hyperparameter ranges (via `hparam_suggester.py` + LLM suggestions)
- When to checkpoint vs continue vs abort a run
- Which eval metrics matter for the stated task

Users can override any agent decision via natural language in the conversation.

## Development Commands
```bash
# Frontend
cd apps/web && pnpm dev
cd apps/web && pnpm build
cd apps/web && pnpm type-check

# Backend
cd backend && uvicorn main:app --reload --port 8000

# Agents
cd agents && python -m pytest tests/ -v

# Full stack
pnpm dev
```

## Environment Variables
```
# apps/web/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=

# backend/.env
DATABASE_URL=
ANTHROPIC_API_KEY=
HUGGINGFACE_TOKEN=
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
WANDB_API_KEY=
STRIPE_SECRET_KEY=
SENTRY_DSN=
```

## Coding Conventions
- Frontend: Server Components by default; "use client" only when needed.
- Backend: Pydantic models for every request/response. No bare except blocks.
- Agents: Each agent is a class with a `run(context) -> AgentResult` interface.
- ML: All training runs saved to `runs/{run_id}/`.
- No comments explaining what code does — only comments explaining non-obvious WHY.

## MCP Servers in Use
- context7 — up-to-date docs for Next.js, FastAPI, Claude SDK, Unsloth.
- filesystem — structured file access across the monorepo
- postgres — direct DB queries for debugging
