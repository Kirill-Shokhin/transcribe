from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_key: str = "asr:queue"
    redis_jobs_key: str = "asr:jobs"

    # Model
    model_type: str = "v3_e2e_rnnt"

    # VAD
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 300

    # Processing
    sample_rate: int = 16000
    max_chunk_duration: float = 25.0
    max_gap_duration: float = 1.5
    short_audio_threshold: float = 30.0

    # Paths
    upload_dir: Path = Path("uploads")
    db_path: Path = Path("data/jobs.db")

    # Webhook
    webhook_timeout: int = 30

    class Config:
        env_prefix = "ASR_"


settings = Settings()
