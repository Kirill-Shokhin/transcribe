import json
import asyncio
import httpx
from redis import asyncio as aioredis
from pathlib import Path

from .config import settings
from .service import asr_service
from .models import JobStatus, JobState


QUEUE_KEY = "asr:queue"
JOBS_KEY = "asr:jobs"


async def get_redis():
    return await aioredis.from_url(settings.redis_url)


async def update_job(redis, job_id: str, status: JobStatus, result=None, error=None):
    state = JobState(
        job_id=job_id,
        status=status,
        result=result,
        error=error
    )
    await redis.hset(JOBS_KEY, job_id, state.model_dump_json())
    return state


async def send_webhook(callback_url: str, state: JobState):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(callback_url, json=state.model_dump())
    except Exception as e:
        print(f"Webhook failed: {e}")


async def process_job(redis, job_data: dict):
    job_id = job_data["job_id"]
    audio_path = job_data["audio_path"]
    callback_url = job_data.get("callback_url")

    await update_job(redis, job_id, JobStatus.processing)

    try:
        result = await asyncio.to_thread(asr_service.transcribe, audio_path)
        state = await update_job(redis, job_id, JobStatus.done, result=result)
    except Exception as e:
        state = await update_job(redis, job_id, JobStatus.error, error=str(e))
    finally:
        Path(audio_path).unlink(missing_ok=True)

    if callback_url:
        await send_webhook(callback_url, state)


async def worker_loop():
    print("Loading models...")
    asr_service.load_models()
    print("Models loaded. Worker ready.")

    redis = await get_redis()

    while True:
        _, job_json = await redis.brpop(QUEUE_KEY)
        job_data = json.loads(job_json)
        print(f"Processing job: {job_data['job_id']}")
        await process_job(redis, job_data)


def run_worker():
    asyncio.run(worker_loop())


if __name__ == "__main__":
    run_worker()
