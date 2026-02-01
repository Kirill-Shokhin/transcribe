from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
from pathlib import Path

app = FastAPI(title="Gateway")

ASR_URL = "http://asr-api:8001"

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


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{ASR_URL}/v1/jobs/{job_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        return resp.json()


@app.get("/health")
async def health():
    return {"status": "ok"}
