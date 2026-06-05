# ─────────────────────────────────────────────────────────────────────────────
# ModelForge Backend — HuggingFace Spaces (Docker SDK)
# Port 7860 is required by HF Spaces.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: CPU-only PyTorch (~250 MB vs 2.5 GB CUDA build) ─────────────────
# Installed separately so it is cached even when requirements-hf.txt changes.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# ── Layer 2: All other Python dependencies ────────────────────────────────────
COPY backend/requirements-hf.txt /tmp/requirements-hf.txt
RUN pip install --no-cache-dir -r /tmp/requirements-hf.txt

# ── Layer 3: Application code ─────────────────────────────────────────────────
COPY backend/ /app/backend/
COPY agents/  /app/agents/

# Ephemeral runtime directories (overridable via UPLOAD_DIR / RUNS_DIR env vars)
RUN mkdir -p /tmp/uploads /tmp/runs

# Environment
ENV PYTHONPATH=/app
ENV PORT=7860
ENV UPLOAD_DIR=/tmp/uploads
ENV RUNS_DIR=/tmp/runs

WORKDIR /app/backend

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
