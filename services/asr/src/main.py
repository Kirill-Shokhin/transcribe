from contextlib import asynccontextmanager
from fastapi import FastAPI

from .routes import v1
from . import database as db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield


app = FastAPI(
    title="ASR Service",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(v1.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
