import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from pathlib import Path

app = FastAPI(title="Gateway")

ASR_URL = "http://asr-api:8001"
LLM_URL = os.getenv("LLM_URL", "http://llm:8000")


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 2000


class SummarizeRequest(BaseModel):
    text: str
    prompt: str | None = None

static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=60) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        resp = await client.post(f"{ASR_URL}/v1/transcribe", files=files)
        return resp.json()


@app.get("/api/jobs")
async def list_jobs():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{ASR_URL}/v1/jobs")
        return resp.json()


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
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "model": "local",
            "messages": req.messages,
            "max_tokens": req.max_tokens
        }
        try:
            resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
            data = resp.json()
            return {"text": data["choices"][0]["message"]["content"]}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    default_prompt = """Проанализируй транскрипт встречи и выдели:
1. Краткое содержание (2-3 предложения)
2. Ключевые решения
3. Задачи и ответственные (если упоминаются)
4. Открытые вопросы

Транскрипт:
"""
    prompt = req.prompt or default_prompt
    messages = [{"role": "user", "content": prompt + req.text}]

    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": 2000
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
