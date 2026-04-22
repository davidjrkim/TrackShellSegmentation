import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.routes.jobs import router as jobs_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        app.state.db_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    except Exception as exc:
        logger.error("Database pool failed to initialise: %s", exc)
        app.state.db_pool = None
    yield
    if app.state.db_pool is not None:
        await app.state.db_pool.close()


app = FastAPI(title="TrackShell Segmentation API", lifespan=lifespan)
app.include_router(jobs_router, prefix="/jobs")


@app.get("/health")
async def health():
    db_ok = app.state.db_pool is not None
    status = "ok" if db_ok else "degraded"
    return JSONResponse(
        {"status": status, "db": db_ok, "version": "1.0.0"},
        status_code=200 if db_ok else 503,
    )
