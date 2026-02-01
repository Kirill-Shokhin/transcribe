import json
import aiosqlite
from pathlib import Path
from .config import settings
from .models import JobState, JobStatus


async def init_db():
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def create_job(job_id: str, filename: str) -> JobState:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO jobs (job_id, filename, status) VALUES (?, ?, ?)",
            (job_id, filename, JobStatus.pending.value)
        )
        await db.commit()
    return JobState(job_id=job_id, status=JobStatus.pending)


async def update_job(job_id: str, status: JobStatus, result: dict | None = None, error: str | None = None, progress: int | None = None):
    async with aiosqlite.connect(settings.db_path) as db:
        if progress is not None:
            await db.execute(
                """UPDATE jobs SET status = ?, progress = ?, result = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE job_id = ?""",
                (status.value, progress, json.dumps(result) if result else None, error, job_id)
            )
        else:
            await db.execute(
                """UPDATE jobs SET status = ?, result = ?, error = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE job_id = ?""",
                (status.value, json.dumps(result) if result else None, error, job_id)
            )
        await db.commit()


async def get_job(job_id: str) -> JobState | None:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return _row_to_job(row)


async def get_all_jobs() -> list[JobState]:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
            return [_row_to_job(row) for row in rows]


async def delete_job(job_id: str):
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        await db.commit()


def _row_to_job(row) -> JobState:
    return JobState(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        progress=row["progress"] or 0,
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        filename=row["filename"]
    )
