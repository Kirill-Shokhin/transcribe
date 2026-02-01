FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (rarely changes)
RUN pip install --no-cache-dir \
    fastapi>=0.115 \
    uvicorn[standard]>=0.34 \
    python-multipart>=0.0.9 \
    redis>=5.0 \
    httpx>=0.28 \
    aiofiles>=24.1 \
    aiosqlite>=0.20 \
    pydantic-settings>=2.7 \
    silero-vad>=5.1 \
    git+https://github.com/salute-developers/GigaAM.git

# App code (changes often)
COPY src/ ./src/

# Runtime config
ENV PYTHONUNBUFFERED=1
RUN mkdir -p uploads

EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
