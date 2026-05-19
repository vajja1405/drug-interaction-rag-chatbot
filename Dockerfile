# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Drug Interaction AI — Backend Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System deps (FAISS needs libgomp) ─────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps (cacheable layer) ────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY api/           api/
COPY chatbot/       chatbot/
COPY models/        models/
COPY rag_pipeline/  rag_pipeline/
COPY data_pipeline/ data_pipeline/
COPY config.py      .
COPY build_index.py .

# ── Data directory (will be overridden by volume mount in docker-compose) ────
# Create the directory so the app doesn't fail if no volume is mounted
RUN mkdir -p data/vectorstore data/raw data/processed

EXPOSE 8000

# ── Default command ──────────────────────────────────────────────────────────
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
