from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    model_name: str = "ai-sage/GigaAM-v3"
    model_revision: str = "e2e_rnnt"

    vad_threshold: float = 0.5
    max_chunk_duration: float = 25.0
    max_gap_duration: float = 1.5
    sample_rate: int = 16000

    class Config:
        env_prefix = "ASR_"


settings = Settings()
