import asyncio
import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.routes.jobs import router as jobs_router
from pipeline.worker import run_worker

logger = logging.getLogger(__name__)

WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        app.state.db_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    except Exception as exc:
        logger.error("Database pool failed to initialise: %s", exc)
        app.state.db_pool = None

    stop_event = asyncio.Event()
    worker_task: asyncio.Task | None = None
    if app.state.db_pool is not None:
        worker_task = asyncio.create_task(run_worker(app.state.db_pool, stop_event))

    try:
        yield
    finally:
        stop_event.set()
        if worker_task is not None:
            try:
                await asyncio.wait_for(worker_task, timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                worker_task.cancel()
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
