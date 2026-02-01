from contextlib import asynccontextmanager
from fastapi import FastAPI

from .routes import v1
from .service import asr_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: models loaded in worker, not in API
    yield
    # Shutdown


app = FastAPI(
    title="ASR Service",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(v1.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    # API is always ready, worker loads models
    return {"ready": True}
