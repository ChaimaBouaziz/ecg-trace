# ECG-Trace API + dashboard (Python pipeline, agents, FastAPI, UI)
# Same Python + scikit-learn versions as the training environment, so the saved model loads identically.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. dependencies first (cached layer)
COPY requirements.txt .
ARG SKLEARN_VERSION=1.9.1
RUN pip install -r requirements.txt "scikit-learn==${SKLEARN_VERSION}"

# 2. application code + trained model (data/ is mounted at runtime, never baked into the image)
COPY pipeline/ pipeline/
COPY ledger/ ledger/
COPY agents/ agents/
COPY api/ api/
COPY ui/ ui/
COPY scripts/ scripts/
COPY models/ models/

# 3. non-root user (uid configurable so files written to the mounted data/ belong to you)
ARG APP_UID=1000
RUN useradd --create-home --uid ${APP_UID} app && mkdir -p data/offchain && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]