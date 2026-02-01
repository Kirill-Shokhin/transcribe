import json
import uuid
import asyncio
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from redis import asyncio as aioredis

from ..config import settings
from ..models import JobCreate, JobState
from .. import database as db


router = APIRouter(prefix="/v1", tags=["v1"])
settings.upload_dir.mkdir(exist_ok=True)


async def get_redis():
    return await aioredis.from_url(settings.redis_url)


@router.post("/transcribe", response_model=JobCreate)
async def transcribe(
    file: UploadFile = File(...),
    callback_url: str | None = None
):
    job_id = str(uuid.uuid4())
    audio_path = settings.upload_dir / f"{job_id}_{file.filename}"

    async with aiofiles.open(audio_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Save to SQLite
    await db.create_job(job_id, file.filename)

    # Push to Redis queue for worker
    redis = await get_redis()
    job_data = {
        "job_id": job_id,
        "audio_path": str(audio_path),
        "callback_url": callback_url
    }
    await redis.lpush(settings.redis_queue_key, json.dumps(job_data))

    return JobCreate(job_id=job_id)


@router.get("/jobs", response_model=list[JobState])
async def list_jobs():
    return await db.get_all_jobs()


@router.get("/jobs/stream")
async def stream_jobs():
    """SSE endpoint for job updates"""
    async def event_generator():
        last_states = {}
        while True:
            all_jobs = await db.get_all_jobs()
            updates = []
            for job in all_jobs:
                key = job.job_id
                state = (job.status, job.progress)
                if last_states.get(key) != state:
                    last_states[key] = state
                    updates.append(job.model_dump())

            if updates:
                yield f"data: {json.dumps(updates)}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str):
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    await db.delete_job(job_id)
    return {"status": "deleted"}
