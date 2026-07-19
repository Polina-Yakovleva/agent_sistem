FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY checkenv.py .

RUN mkdir -p /cache/huggingface

EXPOSE 8000

# Defaults for compose network; overridden by env_file / environment.
ENV PG_HOST=postgres \
    QDRANT_HOST=qdrant \
    LLM_BASE_URL=http://host.docker.internal:11434/v1

CMD ["sh", "-c", "python checkenv.py && exec uvicorn app.api:app --host 0.0.0.0 --port 8000"]
