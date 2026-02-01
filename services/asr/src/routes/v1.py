import json
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from redis import asyncio as aioredis

from ..config import settings
from ..models import TranscribeRequest, JobCreate, JobState, JobStatus
from ..service import asr_service


router = APIRouter(prefix="/v1", tags=["v1"])

QUEUE_KEY = "asr:queue"
JOBS_KEY = "asr:jobs"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def get_redis():
    return await aioredis.from_url(settings.redis_url)


@router.post("/transcribe", response_model=JobCreate)
async def transcribe(
    file: UploadFile = File(...),
    callback_url: str | None = None
):
    job_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    async with aiofiles.open(audio_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    job_data = {
        "job_id": job_id,
        "audio_path": str(audio_path),
        "callback_url": callback_url
    }

    redis = await get_redis()

    initial_state = JobState(job_id=job_id, status=JobStatus.pending)
    await redis.hset(JOBS_KEY, job_id, initial_state.model_dump_json())
    await redis.lpush(QUEUE_KEY, json.dumps(job_data))

    return JobCreate(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str):
    redis = await get_redis()
    job_json = await redis.hget(JOBS_KEY, job_id)

    if not job_json:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobState.model_validate_json(job_json)
