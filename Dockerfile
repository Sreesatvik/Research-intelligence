# API image: FastAPI + retrieval + agent. The UI has its own image (ui/Dockerfile).
FROM python:3.13-slim

WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel bundles GPU libraries (several GB) that a CPU host never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. There is no built-in corpus: every user uploads their own documents. Secrets are NOT copied: see .dockerignore.
COPY apps ./apps
COPY agent ./agent
COPY auth ./auth
COPY chats ./chats
COPY library ./library
COPY cache ./cache
COPY classification ./classification
COPY clustering ./clustering
COPY core ./core
COPY generation ./generation
COPY ingestion ./ingestion
COPY quota ./quota
COPY rag ./rag
COPY representations ./representations
COPY retrieval ./retrieval
RUN mkdir -p data/logs data/db

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "apps.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
