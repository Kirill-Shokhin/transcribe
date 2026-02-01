from pydantic import BaseModel, HttpUrl
from enum import Enum


class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    done = "done"
    error = "error"


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscribeResult(BaseModel):
    text: str
    segments: list[Segment]
    duration: float


class TranscribeRequest(BaseModel):
    callback_url: HttpUrl | None = None


class JobCreate(BaseModel):
    job_id: str


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    result: TranscribeResult | None = None
    error: str | None = None
