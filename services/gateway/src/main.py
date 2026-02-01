import os
import asyncio
import subprocess
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from pathlib import Path

# === Config ===
ASR_URL = os.getenv("ASR_URL", "http://asr-api:8001")
LLM_URL = os.getenv("LLM_URL", "http://llm:8080")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "120"))
COMPOSE_PROJECT = os.getenv("COMPOSE_PROJECT_NAME", "transcribe")

# LLM parameters
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ASR parameters
ASR_TIMEOUT = int(os.getenv("ASR_TIMEOUT", "300"))  # upload timeout for large files

# Track last activity for GPU services
last_activity = {"asr-worker": 0, "llm": 0}
gpu_services_running = {"asr-worker": False, "llm": False}


def docker_compose(*args):
    cmd = ["docker", "compose", "-p", COMPOSE_PROJECT] + list(args)
    subprocess.run(cmd, capture_output=True)


def start_service(name: str):
    if not gpu_services_running.get(name):
        print(f"Starting {name}...")
        docker_compose("start", name)
        gpu_services_running[name] = True


def stop_service(name: str):
    if gpu_services_running.get(name):
        print(f"Stopping {name} (idle)...")
        docker_compose("stop", name)
        gpu_services_running[name] = False


async def wait_for_service(url: str, timeout: float = 120):
    """Wait for service to be ready"""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(f"{url}/health", timeout=2)
                if resp.status_code == 200:
                    return True
            except:
                pass
            await asyncio.sleep(1)
    return False


async def ensure_asr_worker():
    start_service("asr-worker")
    last_activity["asr-worker"] = time.time()


async def ensure_llm():
    start_service("llm")
    last_activity["llm"] = time.time()
    # Wait for LLM to be ready
    ready = await wait_for_service(LLM_URL, timeout=180)
    if not ready:
        raise HTTPException(status_code=503, detail="LLM failed to start")


async def has_pending_jobs() -> bool:
    """Check if there are pending/processing jobs"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ASR_URL}/v1/jobs")
            jobs = resp.json()
            return any(j["status"] in ("pending", "processing") for j in jobs)
    except:
        return False


async def idle_checker():
    """Background task to stop idle GPU services"""
    while True:
        await asyncio.sleep(60)
        now = time.time()

        # Check asr-worker
        if gpu_services_running.get("asr-worker"):
            last = last_activity.get("asr-worker", 0)
            if last > 0 and (now - last) > IDLE_TIMEOUT:
                if not await has_pending_jobs():
                    stop_service("asr-worker")

        # Check llm
        if gpu_services_running.get("llm"):
            last = last_activity.get("llm", 0)
            if last > 0 and (now - last) > IDLE_TIMEOUT:
                stop_service("llm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check initial state
    result = subprocess.run(
        ["docker", "compose", "-p", COMPOSE_PROJECT, "ps", "--format", "{{.Service}}:{{.State}}"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            svc, state = line.split(":", 1)
            if svc in gpu_services_running:
                gpu_services_running[svc] = "running" in state.lower()

    # Start idle checker
    task = asyncio.create_task(idle_checker())
    yield
    task.cancel()


app = FastAPI(title="Gateway", lifespan=lifespan)
static_dir = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 2000


class SummarizeRequest(BaseModel):
    text: str
    prompt: str | None = None


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    await ensure_asr_worker()
    async with httpx.AsyncClient(timeout=ASR_TIMEOUT) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        resp = await client.post(f"{ASR_URL}/v1/transcribe", files=files)
        return resp.json()


@app.get("/api/jobs")
async def list_jobs():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{ASR_URL}/v1/jobs")
        return resp.json()


@app.get("/api/jobs/stream")
async def stream_jobs():
    from starlette.responses import StreamingResponse

    async def proxy_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{ASR_URL}/v1/jobs/stream") as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        proxy_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{ASR_URL}/v1/jobs/{job_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        return resp.json()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{ASR_URL}/v1/jobs/{job_id}")
        return resp.json()


@app.post("/api/chat")
async def chat(req: ChatRequest):
    await ensure_llm()
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        payload = {
            "model": "local",
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": LLM_TEMPERATURE
        }
        try:
            resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
            data = resp.json()
            return {"text": data["choices"][0]["message"]["content"]}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    await ensure_llm()
    default_prompt = """Проанализируй транскрипт встречи и выдели:
1. Краткое содержание (2-3 предложения)
2. Ключевые решения
3. Задачи и ответственные (если упоминаются)
4. Открытые вопросы

Транскрипт:
"""
    prompt = req.prompt or default_prompt
    messages = [{"role": "user", "content": prompt + req.text}]

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE
        }
        try:
            resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
            data = resp.json()
            return {"summary": data["choices"][0]["message"]["content"]}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/gpu-status")
async def gpu_status():
    return {
        "services": gpu_services_running,
        "last_activity": {k: int(time.time() - v) if v > 0 else None for k, v in last_activity.items()},
        "idle_timeout": IDLE_TIMEOUT
    }
