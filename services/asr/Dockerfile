FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    uvicorn[standard]>=0.34 \
    redis>=5.0 \
    httpx>=0.28 \
    aiofiles>=24.1 \
    pydantic-settings>=2.7 \
    torch>=2.5 \
    torchaudio>=2.5 \
    transformers>=4.47 \
    silero-vad>=5.1

# App code
COPY src/ ./src/

# Dirs
RUN mkdir -p uploads .cache/huggingface

EXPOSE 8001

# Default: run API (override for worker)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
