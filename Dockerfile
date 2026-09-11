# syntax=docker/dockerfile:1
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model-cache OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    MAX_DRUGS_PER_REQUEST=5 RATE_LIMIT_PER_MINUTE=5 \
    MAX_CONCURRENT_ANALYSES=1 MAX_UNCACHED_ANALYSES_PER_DAY=200 \
    LLM_MAX_TOKENS=1800 LLM_TIMEOUT_SECONDS=25 PORT=7860
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-runtime.txt
COPY api/ api/
COPY chatbot/ chatbot/
COPY models/ models/
COPY rag_pipeline/ rag_pipeline/
COPY data_pipeline/ data_pipeline/
COPY config.py cache.py cache_identity.py hosting_limits.py build_index.py ./
# Build only repository fixtures. No user data or API secret enters the image.
RUN python build_index.py --all-seed && \
    python -c "from models.severity_classifier import get_trained_classifier; get_trained_classifier()"
COPY --from=frontend /frontend/dist/ /app/frontend/dist/
RUN useradd --uid 1000 --create-home appuser && chown -R appuser:appuser /app
USER appuser
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/v1/ready')"
CMD ["sh", "-c", "exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --no-proxy-headers"]
