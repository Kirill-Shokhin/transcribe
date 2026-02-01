FROM python:3.11-slim

WORKDIR /app

# Install docker CLI + compose plugin (static binaries)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL https://github.com/docker/compose/releases/download/v2.35.0/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose \
    && apt-get remove -y curl && apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn httpx python-multipart

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
