# Ask PanDA FastAPI service container image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# System packages required for faiss-cpu, sentence-transformers, and HTTPS downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure data directories exist even if volumes are mounted later
RUN mkdir -p cache chromadb resources

VOLUME ["/app/cache", "/app/chromadb"]

EXPOSE 8000

CMD ["uvicorn", "ask_panda_server:app", "--host", "0.0.0.0", "--port", "8000"]
