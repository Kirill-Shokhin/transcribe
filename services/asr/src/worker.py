import json
import asyncio
import httpx
from redis import asyncio as aioredis
from pathlib import Path

from .config import settings
from .service import asr_service
from .models import JobStatus, JobState
from . import database as db


async def get_redis():
    return await aioredis.from_url(settings.redis_url)


async def send_webhook(callback_url: str, state: JobState):
    try:
        async with httpx.AsyncClient(timeout=settings.webhook_timeout) as client:
            await client.post(callback_url, json=state.model_dump())
    except Exception as e:
        print(f"Webhook failed: {e}")


async def process_job(job_data: dict):
    job_id = job_data["job_id"]
    audio_path = job_data["audio_path"]
    callback_url = job_data.get("callback_url")

    await db.update_job(job_id, JobStatus.processing, progress=0)

    loop = asyncio.get_event_loop()
    last_progress = [0]  # mutable container for closure

    def on_progress(pct: int):
        if pct > last_progress[0]:
            last_progress[0] = pct
            asyncio.run_coroutine_threadsafe(
                db.update_job(job_id, JobStatus.processing, progress=pct),
                loop
            )

    try:
        result = await asyncio.to_thread(asr_service.transcribe, audio_path, on_progress)
        await db.update_job(job_id, JobStatus.done, result=result.model_dump(), progress=100)
        state = await db.get_job(job_id)
    except Exception as e:
        await db.update_job(job_id, JobStatus.error, error=str(e))
        state = await db.get_job(job_id)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    if callback_url and state:
        await send_webhook(callback_url, state)


async def worker_loop():
    print("Initializing database...")
    await db.init_db()

    print("Loading models...")
    asr_service.load_models()
    print("Models loaded. Worker ready.")

    redis = await get_redis()

    while True:
        _, job_json = await redis.brpop(settings.redis_queue_key)
        job_data = json.loads(job_json)
        print(f"Processing job: {job_data['job_id']}")
        await process_job(job_data)


def run_worker():
    asyncio.run(worker_loop())


if __name__ == "__main__":
    run_worker()
